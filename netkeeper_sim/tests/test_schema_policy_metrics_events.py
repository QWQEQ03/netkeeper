from __future__ import annotations

from netkeeper_sim.metrics.traffic_shift import (
    ForwardingKey, ForwardingPlaneSnapshot, ForwardingState, calculate_versioned_traffic_shift,
)
from netkeeper_sim.policies.schema_evaluator import evaluate_schema_policies
from netkeeper_sim.schemas import (
    Event, Link, LinkAttributes, NetworkConfiguration, NetworkSnapshot, Node, Policy,
    Topology, TrafficDemand, TrafficMatrix,
)
from netkeeper_sim.simulator.deterministic import simulate_deterministic
from netkeeper_sim.simulator.events import apply_scheduled_events
from netkeeper_sim.simulator.schema_evaluation import evaluate_schema_snapshot


def _topology(edges, name="schema-policy"):
    nodes = tuple(Node(node, node) for node in sorted({x for edge in edges for x in edge[:2]}))
    links = tuple(Link(f"L:{a}--{b}:{i}", a, b, 0, LinkAttributes(physical_bandwidth_bps=10_000_000, bandwidth_bps=10_000_000, capacity_max_bps=10_000_000, capacity_bps=10_000_000, ospf_weight=w)) for i, (a, b, w) in enumerate(edges))
    return Topology(f"T:{name}", name, "synthetic", name, nodes, links)


def _snapshot(topology, policies=(), demands=None):
    config = NetworkConfiguration.initial(topology)
    traffic = TrafficMatrix("TM:policy", tuple(node.node_id for node in topology.nodes), tuple(demands or (TrafficDemand("R0", 1000, destination="R3"),)))
    return NetworkSnapshot(0, topology, config, traffic, tuple(policies))


def _routing(topology, config=None):
    config = config or NetworkConfiguration.initial(topology)
    return simulate_deterministic(topology, config, TrafficMatrix("TM:r", tuple(n.node_id for n in topology.nodes), ())).routing_table


def test_reachable_and_forward_ecmp_all_any_and_avoid():
    topology = _topology([("R0", "R1", 1), ("R1", "R3", 1), ("R0", "R2", 1), ("R2", "R3", 1)], "diamond")
    policies = (
        Policy("P:reachable", "reachable", {"source": "R0", "destination": "R3"}),
        Policy("P:pass-all", "forward_pass", {"source": "R0", "destination": "R3", "waypoint": "R1"}),
        Policy("P:pass-any", "forward_pass", {"source": "R0", "destination": "R3", "waypoint": "R1", "path_mode": "any_path"}),
        Policy("P:avoid-all", "forward_avoid", {"source": "R0", "destination": "R3", "forbidden_node": "R2"}),
    )
    report = evaluate_schema_policies(policies, _routing(topology), NetworkConfiguration.initial(topology), topology)
    assert [item.status for item in report.evaluations] == ["satisfied", "unsatisfied", "satisfied", "unsatisfied"]
    down = NetworkConfiguration.initial(topology).with_updates(link_states={link.link_id: "down" for link in topology.links})
    failed = evaluate_schema_policies((policies[0],), _routing(topology, down), down, topology)
    assert failed.evaluations[0].status == "infeasible_due_to_failure"


def test_isolation_conflicts_and_consistency_denominators():
    topology = _topology([("R0", "R4", 1), ("R4", "R3", 1), ("R1", "R5", 1), ("R5", "R2", 1)], "isolation")
    isolated = Policy("P:isolate", "isolation", {"first_source": "R0", "first_destination": "R3", "second_source": "R1", "second_destination": "R2"})
    conflicting = (
        Policy("P:pass", "forward_pass", {"source": "R0", "destination": "R3", "waypoint": "R4"}),
        Policy("P:avoid", "forward_avoid", {"source": "R0", "destination": "R3", "forbidden_node": "R4"}),
    )
    report = evaluate_schema_policies((isolated, *conflicting), _routing(topology), NetworkConfiguration.initial(topology), topology)
    assert report.evaluations[0].status == "satisfied"
    assert {item.status for item in report.evaluations[1:]} == {"conflict"}
    assert (report.numerator, report.denominator, report.feasible_numerator, report.feasible_denominator, report.excluded_count) == (1, 3, 1, 1, 2)
    shared = _topology([("R0", "R4", 1), ("R4", "R3", 1), ("R1", "R4", 1), ("R4", "R2", 1)], "isolation-shared")
    failed = evaluate_schema_policies((isolated,), _routing(shared), NetworkConfiguration.initial(shared), shared)
    assert failed.evaluations[0].status == "unsatisfied"


def test_directed_mlu_and_versioned_step_total_traffic_shift():
    topology = _topology([("R0", "R1", 1)], "mlu")
    snapshot = _snapshot(topology, demands=(TrafficDemand("R0", 2_000_000, destination="R1"),))
    constrained = snapshot.next(configuration=snapshot.configuration.with_updates(performance={topology.links[0].link_id: LinkAttributes(physical_bandwidth_bps=10_000_000, bandwidth_bps=1_000_000, capacity_max_bps=10_000_000, capacity_bps=1_000_000, queue_packets=0)}))
    assert evaluate_schema_snapshot(constrained).snapshot.metrics.maximum_link_utilization == 2.0
    prefix = ForwardingKey("R0", "203.0.113.0/24", "prefix")
    node = ForwardingKey("R0", "R1", "node")
    previous = ForwardingPlaneSnapshot({prefix: ForwardingState(True, frozenset(("R1",))), node: ForwardingState(True, frozenset(("R1",)))})
    current = ForwardingPlaneSnapshot({prefix: ForwardingState(True, frozenset(("R2",))), node: ForwardingState(False, frozenset())})
    paper = calculate_versioned_traffic_shift(previous, current, "paper_v1")
    project = calculate_versioned_traffic_shift(previous, current, "project_v1")
    assert (paper.numerator, paper.denominator, paper.shift_ratio) == (1, 1, 1.0)
    assert (project.numerator, project.denominator, project.shift_ratio) == (2, 2, 1.0)
    diamond = _topology([("R0", "R1", 1), ("R1", "R3", 1), ("R0", "R2", 1), ("R2", "R3", 1)], "shift-snapshots")
    initial = _snapshot(diamond)
    old = evaluate_schema_snapshot(initial).snapshot
    changed = initial.next(configuration=initial.configuration.with_updates(link_states={diamond.links[0].link_id: "down"}))
    evaluated = evaluate_schema_snapshot(changed, previous=old, initial=old).snapshot.metrics
    assert evaluated.traffic_shift_step_project_v1 is not None
    assert evaluated.traffic_shift_total_project_v1 == evaluated.traffic_shift_step_project_v1


def test_events_are_ordered_atomic_and_restore_link_node_traffic_policy():
    topology = _topology([("R0", "R1", 1), ("R1", "R3", 1)], "events")
    snapshot = _snapshot(topology)
    link = topology.links[0]
    policy = Policy("P:new", "reachable", {"source": "R0", "destination": "R3"})
    events = (
        Event("E:2", 0, "traffic_scale", {"factor": 2}),
        Event("E:1", 0, "policy_add", {"policy": policy.to_dict()}),
        Event("E:bad", 0, "link_down", target_id="L:missing"),
        Event("E:3", 0, "link_down", target_id=link.link_id),
    )
    applied = apply_scheduled_events(snapshot, events)
    assert applied.applied_event_ids == ("E:1", "E:2", "E:3") and len(applied.errors) == 1
    assert applied.snapshot.traffic.load_multiplier == 2 and applied.snapshot.policies[0].policy_id == "P:new"
    assert applied.snapshot.configuration.link_states[link.link_id] == "down"
    assert snapshot.configuration.link_states[link.link_id] == "up"  # immutable history
    recovered = apply_scheduled_events(applied.snapshot, (Event("E:up", 0, "link_up", target_id=link.link_id), Event("E:node-down", 0, "node_down", target_id="R1")))
    assert recovered.snapshot.configuration.link_states[link.link_id] == "up"
    assert recovered.snapshot.configuration.node_states["R1"] == "down"
    restored = apply_scheduled_events(recovered.snapshot, (Event("E:node-up", 0, "node_up", target_id="R1"), Event("E:remove", 0, "policy_remove", target_id="P:new")))
    assert restored.snapshot.configuration.node_states["R1"] == "up" and not restored.snapshot.policies
