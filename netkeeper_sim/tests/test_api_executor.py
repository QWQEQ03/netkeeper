from __future__ import annotations

import json
from pathlib import Path

import pytest

from netkeeper_sim.api import ApiRequest, OptimizationRequest, execute
from netkeeper_sim.dataset.dynamic_sequences import dynamic_scenario
from netkeeper_sim.dataset.scenarios import scenario_from_record
from netkeeper_sim.schemas import AtomicAction, JointAction
from netkeeper_sim.schemas import BGPConfiguration, BGPRoute, Link, LinkAttributes, NetworkConfiguration, NetworkScenario, Node, Topology, TrafficDemand, TrafficMatrix
from netkeeper_sim.simulator import UnifiedNetworkEnvironment


ROOT = Path(__file__).resolve().parents[2] / "data" / "netkeeper_lite"


def _record(split: str):
    return json.loads((ROOT / "scenarios" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()[0])


def _environment(split="test"):
    env = UnifiedNetworkEnvironment(); snapshot, _ = env.reset(scenario_from_record(ROOT, _record(split)))
    return env, snapshot


def _request(*calls, **extra): return {"api_version": "v1", "request_id": "exec-test", "calls": list(calls), **extra}
def _call(api, **arguments): return {"api": api, "arguments": arguments}


def _values(snapshot):
    link = snapshot.topology.links[0]; demand = next(item for item in snapshot.traffic.demands if item.destination is not None); route = snapshot.configuration.bgp.routes[0]
    nodes = [item.node_id for item in snapshot.topology.nodes]
    return link, demand, route, nodes


@pytest.mark.parametrize("api", [
    "add_reachable_policy", "add_forward_policy", "add_avoid_policy", "add_isolation_policy", "remove_policy",
    "set_traffic_demand", "scale_traffic_demand", "set_traffic_hotspot", "set_link_state", "set_node_state",
    "set_ospf_weight", "set_bgp_local_pref", "set_bgp_as_path_length", "set_bgp_med", "set_link_bandwidth",
    "set_link_capacity", "set_queue_length", "get_network_state",
])
def test_every_non_optimizer_api_commits_through_unified_environment(api):
    env, snapshot = _environment(); link, demand, route, nodes = _values(snapshot)
    arguments = {
        "add_reachable_policy": dict(src=nodes[0], dst=nodes[1]),
        "add_forward_policy": dict(src=nodes[0], dst=nodes[2], pass_node=nodes[1]),
        "add_avoid_policy": dict(src=nodes[0], dst=nodes[2], avoid_node=nodes[1]),
        "add_isolation_policy": dict(first_src=nodes[0], first_dst=nodes[1], second_src=nodes[2], second_dst=nodes[3]),
        "remove_policy": dict(policy_id=snapshot.policies[0].policy_id),
        "set_traffic_demand": dict(src=demand.source, dst=demand.destination, demand_bps=0),
        "scale_traffic_demand": dict(src=demand.source, dst=demand.destination, factor=2),
        "set_traffic_hotspot": dict(src=demand.source, dst=demand.destination, demand_bps=1),
        "set_link_state": dict(link_id=link.link_id, state="down"),
        "set_node_state": dict(node_id=nodes[-1], state="down"),
        "set_ospf_weight": dict(link_id=link.link_id, weight=64),
        "set_bgp_local_pref": dict(router_id=route.router_id, prefix=route.prefix, next_hop=route.next_hop, value=64),
        "set_bgp_as_path_length": dict(router_id=route.router_id, prefix=route.prefix, next_hop=route.next_hop, length=64),
        "set_bgp_med": dict(router_id=route.router_id, prefix=route.prefix, next_hop=route.next_hop, value=64),
        "set_link_bandwidth": dict(link_id=link.link_id, bandwidth_bps=snapshot.configuration.performance[link.link_id].bandwidth_bps),
        "set_link_capacity": dict(link_id=link.link_id, capacity_bps=snapshot.configuration.performance[link.link_id].capacity_bps),
        "set_queue_length": dict(link_id=link.link_id, queue_packets=0),
        "get_network_state": {},
    }[api]
    response = execute(env, snapshot, _request(_call(api, **arguments)))
    assert response.success, (api, [error.to_dict() for error in response.errors])
    assert response.before_snapshot_id == snapshot.snapshot_id and env.current_snapshot.snapshot_id == response.after_snapshot_id
    if api == "get_network_state": assert response.state["snapshot_id"] == response.after_snapshot_id == snapshot.snapshot_id
    if api == "remove_policy": assert len(env.current_snapshot.policies) == len(snapshot.policies) - 1
    if api.startswith("add_"): assert len(env.current_snapshot.policies) == len(snapshot.policies) + 1


def test_combined_events_actions_are_atomic_and_do_not_touch_dataset_files():
    env, snapshot = _environment(); link, demand, route, nodes = _values(snapshot)
    matrix_file = ROOT / _record("test")["traffic"]["base_matrix_file"]; before_file = matrix_file.read_bytes(); before_json = snapshot.to_json()
    other_link = snapshot.topology.links[1]
    response = execute(env, snapshot, _request(
        _call("add_reachable_policy", src=nodes[0], dst=nodes[1]),
        _call("set_traffic_demand", src=demand.source, dst=demand.destination, demand_bps=0),
        _call("set_link_state", link_id=link.link_id, state="down"),
        _call("set_ospf_weight", link_id=other_link.link_id, weight=64),
        _call("set_bgp_med", router_id=route.router_id, prefix=route.prefix, next_hop=route.next_hop, value=64),
    ))
    assert response.success and matrix_file.read_bytes() == before_file and snapshot.to_json() == before_json
    current = env.current_snapshot
    assert current.configuration.link_states[link.link_id] == "down" and current.configuration.ospf_weights[other_link.link_id] == 64
    assert current.traffic != snapshot.traffic and len(current.policies) == len(snapshot.policies) + 1
    assert response.configuration_diff and response.event_diff["traffic_matrix_changed"]


def test_read_is_post_commit_and_dry_run_is_identical_plan_without_side_effects():
    env, snapshot = _environment(); link, _, _, _ = _values(snapshot)
    calls = (_call("get_network_state"), _call("set_ospf_weight", link_id=link.link_id, weight=64))
    dry = execute(env, snapshot, _request(*calls, dry_run=True))
    assert dry.success and dry.after_snapshot_id == snapshot.snapshot_id and env.current_snapshot == snapshot
    real = execute(env, snapshot, _request(*calls))
    assert real.success and dry.plan.events == real.plan.events and dry.plan.actions == real.plan.actions
    assert real.state["snapshot_id"] == real.after_snapshot_id and real.state["configuration_version"] == env.current_snapshot.configuration.version


def test_link_and_node_recovery_recompute_from_new_snapshots():
    env, snapshot = _environment(); link, _, _, nodes = _values(snapshot)
    down = execute(env, snapshot, _request(_call("set_link_state", link_id=link.link_id, state="down")))
    assert down.success and env.current_snapshot.configuration.link_states[link.link_id] == "down"
    up = execute(env, env.current_snapshot, _request(_call("set_link_state", link_id=link.link_id, state="up")))
    assert up.success and env.current_snapshot.configuration.link_states[link.link_id] == "up"
    node_down = execute(env, env.current_snapshot, _request(_call("set_node_state", node_id=nodes[-1], state="down")))
    node_up = execute(env, env.current_snapshot, _request(_call("set_node_state", node_id=nodes[-1], state="up")))
    assert node_down.success and node_up.success and env.current_snapshot.configuration.node_states[nodes[-1]] == "up"


def test_failures_and_stale_snapshot_have_zero_side_effects(monkeypatch):
    env, snapshot = _environment(); link, _, _, _ = _values(snapshot); before = snapshot.to_json()
    invalid = execute(env, snapshot, _request(_call("set_ospf_weight", link_id=link.link_id, weight=64), _call("set_ospf_weight", link_id="missing", weight=1)))
    assert not invalid.success and not invalid.applied_calls and invalid.after_snapshot_id is None and env.current_snapshot.to_json() == before
    stale = execute(env, snapshot, _request(_call("get_network_state"), expected_snapshot_id="SN:old"))
    assert stale.errors[0].code == "STALE_SNAPSHOT" and env.current_snapshot.to_json() == before
    monkeypatch.setattr(env, "step", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("reject")))
    rejected = execute(env, snapshot, _request(_call("set_ospf_weight", link_id=link.link_id, weight=64)))
    assert rejected.errors[0].code == "ENVIRONMENT_REJECTED" and env.current_snapshot.to_json() == before


def test_mapping_failure_is_precommit_and_has_no_side_effects(monkeypatch):
    env, snapshot = _environment(); link, _, _, _ = _values(snapshot); before = snapshot.to_json()
    monkeypatch.setattr("netkeeper_sim.api.executor._action_for", lambda *args: (_ for _ in ()).throw(ValueError("mapping reject")))
    response = execute(env, snapshot, _request(_call("set_ospf_weight", link_id=link.link_id, weight=64)))
    assert not response.success and response.errors[0].code == "MAPPING_FAILED" and env.current_snapshot.to_json() == before


class _Dispatcher:
    def __init__(self, action): self.action, self.seen = action, None
    def dispatch(self, snapshot, request: OptimizationRequest): self.seen = (snapshot.snapshot_id, request); return self.action


def test_optimizer_is_explicitly_unavailable_or_dispatched_once():
    env, snapshot = _environment(); link, _, _, _ = _values(snapshot); request = _request(_call("optimize_network", objectives=["mlu"]))
    unavailable = execute(env, snapshot, request)
    assert not unavailable.success and unavailable.errors[0].code == "OPTIMIZER_UNAVAILABLE" and env.current_snapshot == snapshot
    dispatcher = _Dispatcher(JointAction((AtomicAction("ospf", "ospf_weight", {"link_id": link.link_id}, "set", 64),), snapshot_id=snapshot.snapshot_id))
    response = execute(env, snapshot, request, dispatcher=dispatcher)
    assert response.success and response.optimization_status == "dispatched" and dispatcher.seen[0] == snapshot.snapshot_id
    dry = execute(env, env.current_snapshot, _request(_call("optimize_network", objectives=["mlu"]), dry_run=True))
    assert dry.success and dry.optimization_status == "pending" and env.current_snapshot.snapshot_id == response.after_snapshot_id


def test_train_validation_test_and_dynamic_sequence_end_to_end():
    for split in ("train", "validation", "test"):
        env, snapshot = _environment(split); response = execute(env, snapshot, _request(_call("get_network_state")))
        assert response.success and response.metrics["total_input_bps"] >= 0
    sequence = json.loads((ROOT / "dynamic_sequences" / "test.jsonl").read_text(encoding="utf-8").splitlines()[0])
    scenario = dynamic_scenario(ROOT, sequence); env = UnifiedNetworkEnvironment(); snapshot, _ = env.reset(scenario)
    demand = next(item for item in snapshot.traffic.demands if item.destination is not None)
    response = execute(env, snapshot, _request(_call("get_network_state"), _call("set_traffic_demand", src=demand.source, dst=demand.destination, demand_bps=0)))
    assert response.success and response.after_snapshot_id != snapshot.snapshot_id and response.state["policy_count"] == len(snapshot.policies) + 1


def test_api_ospf_bgp_and_performance_changes_have_kernel_effects():
    nodes = tuple(Node(f"R{i}", str(i)) for i in range(4))
    edges = (("R0", "R1"), ("R1", "R3"), ("R0", "R2"), ("R2", "R3"))
    links = tuple(Link(f"L:{a}-{b}", a, b, 0, LinkAttributes(physical_bandwidth_bps=10_000_000, bandwidth_bps=10_000_000, capacity_max_bps=10_000_000, capacity_bps=10_000_000)) for a, b in edges)
    topology = Topology("T:api-effects", "api-effects", "synthetic", "effects", nodes, links)
    config = NetworkConfiguration.initial(topology).with_updates(bgp=BGPConfiguration((
        BGPRoute("R0", "203.0.113.0/24", "R1", 64, (64501,), 1), BGPRoute("R0", "203.0.113.0/24", "R2", 64, (64501,), 1),
    )))
    traffic = TrafficMatrix("TM:api-effects", tuple(node.node_id for node in nodes), (TrafficDemand("R0", 2_000_000, destination="R3"), TrafficDemand("R0", 1_000_000, prefix="203.0.113.0/24")))
    scenario = NetworkScenario("S:api-effects", topology, traffic, configuration=config)
    env = UnifiedNetworkEnvironment(); snapshot, _ = env.reset(scenario)
    ospf = execute(env, snapshot, _request(_call("set_ospf_weight", link_id=links[0].link_id, weight=64)))
    route = next(item for item in env.current_snapshot.routing_state if item.router_id == "R0" and item.destination == "R3")
    assert ospf.success and route.next_hops == ("R2",)
    bgp = execute(env, env.current_snapshot, _request(_call("set_bgp_local_pref", router_id="R0", prefix="203.0.113.0/24", next_hop="R1", value=1)))
    assert bgp.success and bgp.metrics["traffic_shift_step_project_v1"] is not None
    before_mlu = env.current_snapshot.metrics.maximum_link_utilization
    performance = execute(env, env.current_snapshot, _request(_call("set_link_capacity", link_id=links[2].link_id, capacity_bps=1_000_000)))
    assert performance.success and env.current_snapshot.metrics.maximum_link_utilization > before_mlu
