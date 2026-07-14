from __future__ import annotations

from netkeeper_sim.schemas import (
    AtomicAction, BGPConfiguration, BGPRoute, Event, JointAction, Link,
    LinkAttributes, NetworkConfiguration, NetworkScenario, Node, Policy,
    Topology, TrafficDemand, TrafficMatrix,
)
from netkeeper_sim.simulator import UnifiedNetworkEnvironment


def _topology(edges, name="unified"):
    nodes = tuple(Node(node, node) for node in sorted({item for edge in edges for item in edge[:2]}))
    links = tuple(Link(f"L:{a}--{b}:{i}", a, b, 0, LinkAttributes(physical_bandwidth_bps=10_000_000, bandwidth_bps=10_000_000, capacity_max_bps=10_000_000, capacity_bps=10_000_000, ospf_weight=w)) for i, (a, b, w) in enumerate(edges))
    return Topology(f"T:{name}", name, "synthetic", name, nodes, links)


def _scenario(topology, *, config=None, events=(), policies=(), demands=None, max_steps=5):
    traffic = TrafficMatrix("TM:unified", tuple(node.node_id for node in topology.nodes), tuple(demands or (TrafficDemand("R0", 2_000_000, destination="R3"),)))
    return NetworkScenario("S:unified", topology, traffic, tuple(policies), config, tuple(events), max_steps=max_steps)


def test_reset_is_immutable_deterministic_and_no_update_is_noop():
    topology = _topology([("R0", "R1", 1), ("R1", "R3", 1)])
    scenario = _scenario(topology)
    env = UnifiedNetworkEnvironment()
    first, first_observation = env.reset(scenario, seed=7)
    second, second_observation = env.reset(scenario, seed=7)
    assert first == second and first_observation == second_observation
    result = env.step(first, JointAction((AtomicAction("ospf", "ospf_weight", {"link_id": topology.links[0].link_id}),), snapshot_id=first.snapshot_id))
    assert result.next_snapshot.configuration.version == first.configuration.version
    assert result.next_snapshot.metrics.traffic_shift_step_project_v1 == 0.0
    assert first.configuration == second.configuration and result.next_snapshot.snapshot_id != first.snapshot_id


def test_ospf_bgp_and_performance_actions_have_end_to_end_effects():
    topology = _topology([("R0", "R1", 1), ("R1", "R3", 1), ("R0", "R2", 2), ("R2", "R3", 1)], "actions")
    config = NetworkConfiguration.initial(topology)
    scenario = _scenario(topology, config=config)
    env = UnifiedNetworkEnvironment(); initial, _ = env.reset(scenario)
    result = env.step(initial, JointAction((AtomicAction("ospf", "ospf_weight", {"link_id": topology.links[0].link_id}, "set", 10),), snapshot_id=initial.snapshot_id))
    route = next(item for item in result.next_snapshot.routing_state if item.router_id == "R0" and item.destination == "R3")
    assert route.next_hops == ("R2",)
    exit_link = next(link for link in topology.links if {link.source, link.target} == {"R0", "R2"})
    perf = env.step(result.next_snapshot, JointAction((AtomicAction("performance", "capacity_bps", {"link_id": exit_link.link_id}, "set", 500_000),), snapshot_id=result.next_snapshot.snapshot_id))
    assert perf.metrics.maximum_link_utilization > result.metrics.maximum_link_utilization
    bgp_config = config.with_updates(bgp=BGPConfiguration((
        BGPRoute("R0", "203.0.113.0/24", "R1", 200, (64501,), 1),
        BGPRoute("R0", "203.0.113.0/24", "R2", 100, (64501,), 1),
    )), performance={exit_link.link_id: LinkAttributes(physical_bandwidth_bps=10_000_000, bandwidth_bps=500_000, capacity_max_bps=10_000_000, capacity_bps=500_000)})
    prefix_scenario = _scenario(topology, config=bgp_config, demands=(TrafficDemand("R0", 2_000_000, prefix="203.0.113.0/24"),))
    prefix, _ = env.reset(prefix_scenario)
    changed = env.step(prefix, JointAction((AtomicAction("bgp", "local_preference", {"router_id": "R0", "prefix": "203.0.113.0/24", "next_hop": "R2"}, "set", 300),), snapshot_id=prefix.snapshot_id))
    assert changed.metrics.traffic_shift_step_project_v1 is not None
    assert changed.metrics.maximum_link_utilization != prefix.metrics.maximum_link_utilization


def test_events_errors_history_and_policy_metrics():
    topology = _topology([("R0", "R1", 1), ("R1", "R3", 1)], "events")
    policy = Policy("P:reach", "reachable", {"source": "R0", "destination": "R3"})
    scenario = _scenario(topology, events=(
        Event("E:scale", 0, "traffic_scale", {"factor": 2}),
        Event("E:policy", 0, "policy_add", {"policy": policy.to_dict()}),
        Event("E:down", 1, "node_down", target_id="R1"),
    ))
    env = UnifiedNetworkEnvironment(); initial, _ = env.reset(scenario)
    after_events = env.step(initial, JointAction(()))
    assert after_events.metrics.maximum_link_utilization > initial.metrics.maximum_link_utilization
    assert after_events.metrics.policy_denominator == 1
    failed = env.step(after_events.next_snapshot, JointAction(()))
    assert failed.metrics.unreachable_bps > 0
    invalid = env.step(failed.next_snapshot, JointAction((AtomicAction("ospf", "ospf_weight", {"link_id": "L:missing"}, "set", 3),), snapshot_id=failed.next_snapshot.snapshot_id))
    assert invalid.errors and invalid.next_snapshot.configuration.version == failed.next_snapshot.configuration.version
    assert invalid.next_snapshot.configuration.ospf_weights == failed.next_snapshot.configuration.ospf_weights
    replay = env.step(after_events.next_snapshot, JointAction(()))
    assert replay.next_snapshot.metrics == failed.next_snapshot.metrics
