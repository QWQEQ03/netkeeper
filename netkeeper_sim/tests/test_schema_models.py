from __future__ import annotations

import networkx as nx
import pytest

from netkeeper_sim.schemas import (
    BGPConfiguration,
    Event,
    LinkAttributes,
    Metrics,
    NetworkConfiguration,
    NetworkSnapshot,
    Policy,
    TrafficDemand,
    TrafficMatrix,
    legacy_topology_from_schema,
)
from netkeeper_sim.routing.ospf import compute_ospf_routes
from netkeeper_sim.schemas.loader import normalize_schema_topology


def _parallel_raw_graph() -> nx.MultiGraph:
    graph = nx.MultiGraph()
    graph.add_node("10", label="Ten")
    graph.add_node("2", label="Two")
    graph.add_edge("10", "2", key="b", LinkSpeedRaw="100", LinkSpeedUnits="Mbps")
    graph.add_edge("10", "2", key="a", LinkSpeedRaw="1", LinkSpeedUnits="Gbps")
    return graph


def _topology():
    return normalize_schema_topology(_parallel_raw_graph(), "Parallel Test", "synthetic")


def test_schema_topology_is_stable_and_parallel_links_are_distinct():
    first = _topology()
    second = _topology()

    assert first.topology_id == second.topology_id
    assert [node.node_id for node in first.nodes] == ["R0", "R1"]
    assert first.nodes[0].original_id == "2"
    assert first.nodes[1].original_label == "Ten"
    assert [link.link_id for link in first.links] == [link.link_id for link in second.links]
    assert len(first.links) == 2
    assert len({link.link_id for link in first.links}) == 2
    assert len(first.arcs) == 4


def test_schema_units_conversion_and_defaults():
    topology = _topology()
    values = {link.attributes.physical_bandwidth_bps for link in topology.links}
    assert values == {100_000_000, 1_000_000_000}
    assert all(link.attributes.delay_ms == 1.0 for link in topology.links)
    assert all(link.attributes.queue_packets == 1000 for link in topology.links)

    raw = nx.Graph()
    raw.add_node("a")
    raw.add_node("b")
    raw.add_edge("a", "b", LinkSpeedRaw="100")
    fallback = normalize_schema_topology(raw, "Fallback", "synthetic")
    assert fallback.links[0].attributes.bandwidth_bps == 100_000_000
    assert fallback.links[0].attributes.value_source == "default_missing_unit"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"loss_rate": 1.1},
        {"ospf_weight": 0},
        {"queue_packets": -1},
        {"packet_size_bytes": 63},
        {"capacity_bps": 100_000_001},
    ],
)
def test_schema_rejects_invalid_link_attribute_ranges(kwargs):
    with pytest.raises(ValueError):
        LinkAttributes(**kwargs)


def test_configuration_updates_are_versioned_immutable_and_diffable():
    topology = _topology()
    original = NetworkConfiguration.initial(topology)
    link_id = topology.links[0].link_id
    updated = original.with_updates(ospf_weights={link_id: 7}, step=1)

    assert original.version == 0
    assert original.ospf_weights[link_id] == 1
    assert updated.version == 1
    assert updated.parent_version == 0
    assert updated.ospf_weights[link_id] == 7
    assert original.diff(updated)[f"ospf_weights.{link_id}"] == (1, 7)
    with pytest.raises(TypeError):
        updated.ospf_weights[link_id] = 9  # type: ignore[index]


def test_schema_configuration_has_a_legacy_routing_adapter():
    topology = _topology()
    configuration = NetworkConfiguration.initial(topology)
    legacy = legacy_topology_from_schema(topology, configuration)

    table = compute_ospf_routes(legacy)
    assert table["R0"]["R1"].reachable is True


def test_snapshot_does_not_modify_history_and_round_trips_json():
    topology = _topology()
    config = NetworkConfiguration.initial(topology)
    traffic = TrafficMatrix("TM:test", tuple(node.node_id for node in topology.nodes), (TrafficDemand("R0", 1_000_000, destination="R1"),))
    policy = Policy("P:test:0", "reachable", {"source": "R0", "destination": "R1"})
    initial = NetworkSnapshot(0, topology, config, traffic, (policy,), metrics=Metrics(total_input_bps=1_000_000))
    updated = initial.next(configuration=config.with_updates(ospf_weights={topology.links[0].link_id: 5}, step=1))

    assert initial.snapshot_id != updated.snapshot_id
    assert initial.configuration.ospf_weights[topology.links[0].link_id] == 1
    assert updated.configuration.ospf_weights[topology.links[0].link_id] == 5
    recovered = NetworkSnapshot.from_json(updated.to_json())
    assert recovered == updated
    assert recovered.topology.links[0].attributes.bandwidth_bps == 1_000_000_000


def test_policy_event_and_traffic_matrix_round_trip():
    traffic = TrafficMatrix("TM:round", ("R0", "R1"), (TrafficDemand("R0", 10.0, prefix="203.0.113.0/24"),), seed=7)
    policy = Policy("P:round:0", "forward_avoid", {"source": "R0", "destination": "R1", "forbidden_node": "R2"})
    event = Event("E:round:2:0", 2, "link_down", target_id="L:R0--R1:0")

    assert TrafficMatrix.from_dict(traffic.to_dict()) == traffic
    assert Policy.from_dict(policy.to_dict()) == policy
    assert Event.from_dict(event.to_dict()) == event
