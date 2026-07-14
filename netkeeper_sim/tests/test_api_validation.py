from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest

from netkeeper_sim.api import API_REGISTRY, ApiRequest, ApiResponse, export_json_schema, validate_request
from netkeeper_sim.dataset.scenarios import scenario_from_record
from netkeeper_sim.simulator import UnifiedNetworkEnvironment


@pytest.fixture(scope="module")
def snapshot():
    root = Path(__file__).resolve().parents[2] / "data" / "netkeeper_lite"
    record = json.loads((root / "scenarios" / "test.jsonl").read_text(encoding="utf-8").splitlines()[0])
    scenario = scenario_from_record(root, record)
    return UnifiedNetworkEnvironment().reset(scenario)[0]


def request(*calls, **more):
    return {"api_version": "v1", "request_id": "api-test", "calls": list(calls), **more}


def call(api, **arguments): return {"api": api, "arguments": arguments}


def _safe_pair(snapshot):
    isolated = {(p.fields.get("first_source"), p.fields.get("first_destination")) for p in snapshot.policies if p.kind == "isolation"}
    isolated |= {(p.fields.get("second_source"), p.fields.get("second_destination")) for p in snapshot.policies if p.kind == "isolation"}
    nodes = [node.node_id for node in snapshot.topology.nodes]
    return next((a, b) for a in nodes for b in nodes if a != b and (a, b) not in isolated)


def test_every_registry_api_has_a_valid_real_test_split_request(snapshot):
    src, dst = _safe_pair(snapshot); link = snapshot.topology.links[0]; demand = next(d for d in snapshot.traffic.demands if d.destination is not None)
    graph = nx.Graph((l.source, l.target) for l in snapshot.topology.links)
    a, waypoint, b = next((a, w, b) for w in graph.nodes for a in graph.neighbors(w) for b in graph.neighbors(w) if a != b and (a, b) != (src, dst))
    nodes = [n.node_id for n in snapshot.topology.nodes]
    first_src, first_dst, second_src, second_dst = nodes[:4]
    route = snapshot.configuration.bgp.routes[0]
    legal = {
        "add_reachable_policy": call("add_reachable_policy", src=src, dst=dst),
        "add_forward_policy": call("add_forward_policy", src=a, dst=b, pass_node=waypoint),
        "add_avoid_policy": call("add_avoid_policy", src=a, dst=b, avoid_node=waypoint),
        "add_isolation_policy": call("add_isolation_policy", first_src=first_src, first_dst=first_dst, second_src=second_src, second_dst=second_dst),
        "remove_policy": call("remove_policy", policy_id=snapshot.policies[0].policy_id),
        "set_traffic_demand": call("set_traffic_demand", src=demand.source, dst=demand.destination, demand_bps=0),
        "scale_traffic_demand": call("scale_traffic_demand", src=demand.source, dst=demand.destination, factor=1.0),
        "set_traffic_hotspot": call("set_traffic_hotspot", src=demand.source, dst=demand.destination, demand_bps=1.0),
        "set_link_state": call("set_link_state", link_id=link.link_id, state="down"),
        "set_node_state": call("set_node_state", node_id=nodes[0], state="down"),
        "set_ospf_weight": call("set_ospf_weight", link_id=link.link_id, weight=1),
        "set_bgp_local_pref": call("set_bgp_local_pref", router_id=route.router_id, prefix=route.prefix, next_hop=route.next_hop, value=1),
        "set_bgp_as_path_length": call("set_bgp_as_path_length", router_id=route.router_id, prefix=route.prefix, next_hop=route.next_hop, length=1),
        "set_bgp_med": call("set_bgp_med", router_id=route.router_id, prefix=route.prefix, next_hop=route.next_hop, value=1),
        "set_link_bandwidth": call("set_link_bandwidth", link_id=link.link_id, bandwidth_bps=snapshot.configuration.performance[link.link_id].bandwidth_bps),
        "set_link_capacity": call("set_link_capacity", link_id=link.link_id, capacity_bps=snapshot.configuration.performance[link.link_id].capacity_bps),
        "set_queue_length": call("set_queue_length", link_id=link.link_id, queue_packets=0),
        "get_network_state": call("get_network_state"),
        "optimize_network": call("optimize_network", objectives=["mlu"]),
    }
    assert set(legal) == set(API_REGISTRY)
    for name, item in legal.items():
        result = validate_request(request(item), snapshot)
        assert result.valid, (name, [error.to_dict() for error in result.errors])


@pytest.mark.parametrize("payload,code", [
    (request(call("nope")), "UNSUPPORTED_API"),
    (request(call("add_reachable_policy", src="R0")), "MISSING_ARGUMENT"),
    (request(call("get_network_state", nope=True)), "ADDITIONAL_PROPERTY"),
    ({"api_version": "v1", "request_id": "x", "calls": []}, "INVALID_CALLS"),
    ({"api_version": "v2", "request_id": "x", "calls": [call("get_network_state")]}, "UNSUPPORTED_VERSION"),
])
def test_schema_errors(snapshot, payload, code):
    assert code in {error.code for error in validate_request(payload, snapshot).errors}


def test_entity_policy_and_state_errors(snapshot):
    link = snapshot.topology.links[0]
    cases = [
        (call("add_reachable_policy", src="R999", dst="R1"), "UNKNOWN_NODE"),
        (call("set_ospf_weight", link_id="missing", weight=1), "UNKNOWN_LINK"),
        (call("remove_policy", policy_id="missing"), "POLICY_NOT_FOUND"),
        (call("set_bgp_med", router_id="R0", prefix="missing", next_hop="R1", value=1), "UNKNOWN_BGP_ROUTE"),
        (call("add_reachable_policy", src="R0", dst="R0"), "INVALID_ENDPOINTS"),
        (call("add_forward_policy", src="R0", dst="R1", pass_node="R0"), "INVALID_WAYPOINT"),
        (call("add_avoid_policy", src="R0", dst="R1", avoid_node="R1"), "INVALID_WAYPOINT"),
        (call("add_isolation_policy", first_src="R0", first_dst="R1", second_src="R0", second_dst="R2"), "OVERLAPPING_ISOLATION"),
    ]
    for item, code in cases:
        assert code in {error.code for error in validate_request(request(item), snapshot).errors}
    result = validate_request(request(call("set_link_state", link_id=link.link_id, state="down"), call("set_ospf_weight", link_id=link.link_id, weight=1)), snapshot)
    assert "OBJECT_DOWN" in {error.code for error in result.errors}


@pytest.mark.parametrize("api,args", [
    ("set_ospf_weight", {"weight": 0}), ("set_ospf_weight", {"weight": 65}),
    ("set_bgp_local_pref", {"value": True}), ("set_queue_length", {"queue_packets": -1}),
])
def test_ranges_and_bool_are_rejected(snapshot, api, args):
    link = snapshot.topology.links[0]; route = snapshot.configuration.bgp.routes[0]
    base = {"set_ospf_weight": {"link_id": link.link_id}, "set_bgp_local_pref": {"router_id": route.router_id, "prefix": route.prefix, "next_hop": route.next_hop}, "set_queue_length": {"link_id": link.link_id}}[api]
    result = validate_request(request(call(api, **base, **args)), snapshot)
    assert result.errors and {e.code for e in result.errors} & {"INVALID_TYPE", "OUT_OF_RANGE"}
    demand = next(d for d in snapshot.traffic.demands if d.destination is not None)
    for value in (float("nan"), float("inf"), True):
        result = validate_request(request(call("set_traffic_demand", src=demand.source, dst=demand.destination, demand_bps=value)), snapshot)
        assert "INVALID_TYPE" in {e.code for e in result.errors}


def test_conflicts_order_stale_and_dry_run_are_pure(snapshot):
    link = snapshot.topology.links[0]; src, dst = _safe_pair(snapshot); node = next(n.node_id for n in snapshot.topology.nodes if n.node_id not in {src, dst})
    conflict = validate_request(request(call("add_forward_policy", src=src, dst=dst, pass_node=node), call("add_avoid_policy", src=src, dst=dst, avoid_node=node)), snapshot)
    assert "POLICY_CONFLICT" in {e.code for e in conflict.errors}
    demand = next(d for d in snapshot.traffic.demands if d.destination is not None)
    traffic = validate_request(request(call("set_traffic_demand", src=demand.source, dst=demand.destination, demand_bps=1), call("scale_traffic_demand", src=demand.source, dst=demand.destination, factor=2)), snapshot)
    assert "TRAFFIC_OPERATION_CONFLICT" in {e.code for e in traffic.errors}
    ordered = validate_request(request(call("set_ospf_weight", link_id=link.link_id, weight=1), call("add_reachable_policy", src=src, dst=dst)), snapshot)
    assert "ORDER_VIOLATION" in {e.code for e in ordered.errors}
    stale = validate_request(request(call("get_network_state"), expected_snapshot_id="SN:old"), snapshot)
    assert stale.errors[0].code == "STALE_SNAPSHOT"
    before = snapshot.to_json(); result = validate_request(request(call("set_ospf_weight", link_id=link.link_id, weight=64), dry_run=True), snapshot)
    assert result.valid and result.plan and snapshot.to_json() == before and result.plan.predicted_diff["configuration_version_delta"] == 1


def test_models_and_exported_schema_round_trip(snapshot):
    request_model = ApiRequest.from_dict(request(call("get_network_state"), dry_run=True))
    assert ApiRequest.from_json(request_model.to_json()) == request_model
    result = validate_request(request_model, snapshot); response = ApiResponse(True, "api-test", plan=result.plan)
    assert ApiResponse.from_json(response.to_json()).to_dict() == response.to_dict()
    schema = export_json_schema()
    assert schema["$schema"].endswith("2020-12/schema") and schema["additionalProperties"] is False
    definitions = schema["$defs"]
    assert set(definitions) == set(API_REGISTRY)
    for definition in API_REGISTRY.values():
        assert definition.description and definition.arguments_schema["additionalProperties"] is False
