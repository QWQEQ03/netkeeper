from __future__ import annotations

from netkeeper_sim.schemas import (
    BGPConfiguration, BGPRoute, Link, LinkAttributes, NetworkConfiguration,
    Node, Topology, TrafficDemand, TrafficMatrix,
)
from netkeeper_sim.simulator import simulate_deterministic
from netkeeper_sim.simulator.environment import NetworkSimulationEnvironment


def _topology(edges: list[tuple[str, str, int]], name: str = "T") -> Topology:
    node_ids = sorted({node for edge in edges for node in edge[:2]})
    nodes = tuple(Node(node, node, node) for node in node_ids)
    links = tuple(
        Link(f"L:{left}--{right}:{index}", left, right, 0, LinkAttributes(
            physical_bandwidth_bps=10_000_000, bandwidth_bps=10_000_000,
            capacity_max_bps=10_000_000, capacity_bps=10_000_000, ospf_weight=weight,
        ))
        for index, (left, right, weight) in enumerate(edges)
    )
    return Topology(f"T:{name}", name, "synthetic", name, nodes, links)


def _traffic(topology: Topology, *demands: TrafficDemand) -> TrafficMatrix:
    return TrafficMatrix("TM:test", tuple(node.node_id for node in topology.nodes), tuple(demands))


def _assert_conserved(result):
    for outcome in result.demand_outcomes:
        assert abs(outcome.offered_bps - outcome.delivered_bps - outcome.dropped_bps - outcome.unreachable_bps) < 1e-5


def test_ospf_single_path_and_ecmp_are_complete():
    single = _topology([("R0", "R1", 2), ("R1", "R2", 3)], "single")
    result = simulate_deterministic(single, NetworkConfiguration.initial(single), _traffic(single, TrafficDemand("R0", 1000, destination="R2")))
    route = next(item for item in result.routing_table if item.router_id == "R0" and item.destination == "R2")
    assert route.reachable and route.next_hops == ("R1",) and route.cost == 5
    diamond = _topology([("R0", "R1", 1), ("R1", "R3", 1), ("R0", "R2", 1), ("R2", "R3", 1)], "ecmp")
    ecmp = simulate_deterministic(diamond, NetworkConfiguration.initial(diamond), _traffic(diamond, TrafficDemand("R0", 1000, destination="R3")))
    route = next(item for item in ecmp.routing_table if item.router_id == "R0" and item.destination == "R3")
    assert route.next_hops == ("R1", "R2")
    first_arcs = [load.load_bps for load in ecmp.directed_link_loads if load.arc_id.endswith(":R0->R1") or load.arc_id.endswith(":R0->R2")]
    assert first_arcs == [500.0, 500.0]
    _assert_conserved(ecmp)
    assert NetworkSimulationEnvironment().simulate_schema(diamond, NetworkConfiguration.initial(diamond), _traffic(diamond, TrafficDemand("R0", 1, destination="R3"))).delivered_bps == 1


def test_parallel_link_weight_and_single_link_failure_are_independent():
    topology = _topology([("R0", "R1", 10), ("R0", "R1", 5)], "parallel")
    base = NetworkConfiguration.initial(topology)
    traffic = _traffic(topology, TrafficDemand("R0", 1000, destination="R1"))
    result = simulate_deterministic(topology, base, traffic)
    low, high = topology.links[1], topology.links[0]
    assert next(load.load_bps for load in result.directed_link_loads if f":{low.link_id}:R0->R1" in load.arc_id) == 1000
    assert next(load.load_bps for load in result.directed_link_loads if f":{high.link_id}:R0->R1" in load.arc_id) == 0
    failed = base.with_updates(link_states={low.link_id: "down"}, step=1)
    recovered = simulate_deterministic(topology, failed, traffic)
    assert recovered.delivered_bps == 1000
    assert next(load.load_bps for load in recovered.directed_link_loads if f":{high.link_id}:R0->R1" in load.arc_id) == 1000
    _assert_conserved(recovered)


def test_bgp_localpref_med_and_exit_change_affect_forwarding_mlu_and_shift():
    topology = _topology([("R0", "R1", 1), ("R1", "R3", 1), ("R0", "R2", 1), ("R2", "R3", 1)], "bgp")
    base = NetworkConfiguration.initial(topology)
    routes_a = BGPConfiguration((
        BGPRoute("R0", "203.0.113.0/24", "R1", 200, (64501,), 100),
        BGPRoute("R0", "203.0.113.0/24", "R2", 100, (64501,), 1),
    ))
    traffic = _traffic(topology, TrafficDemand("R0", 2_000_000, prefix="203.0.113.0/24"))
    first = simulate_deterministic(topology, base.with_updates(bgp=routes_a), traffic)
    assert first.selected_bgp_routes[0].next_hop == "R1"
    routes_b = BGPConfiguration((
        BGPRoute("R0", "203.0.113.0/24", "R1", 200, (64501,), 200),
        BGPRoute("R0", "203.0.113.0/24", "R2", 200, (64501,), 1),
    ))
    second = simulate_deterministic(topology, base.with_updates(bgp=routes_b), traffic, previous=first)
    assert second.selected_bgp_routes[0].next_hop == "R2"  # MED ascending after equal LocalPref/AS path
    assert second.metrics.traffic_shift_project_v1 == 1.0
    # Constrain only the R0->R2 exit: the BGP selection changes MLU too.
    r2_link = next(link for link in topology.links if {link.source, link.target} == {"R0", "R2"})
    constrained = base.with_updates(performance={r2_link.link_id: LinkAttributes(
        physical_bandwidth_bps=10_000_000, bandwidth_bps=500_000, capacity_max_bps=10_000_000, capacity_bps=500_000,
    )}, bgp=routes_b)
    third = simulate_deterministic(topology, constrained, traffic, previous=first)
    assert third.metrics.maximum_link_utilization > first.metrics.maximum_link_utilization
    _assert_conserved(third)


def test_performance_parameters_and_failures_are_observable_and_conserved():
    topology = _topology([("R0", "R1", 1)], "performance")
    link = topology.links[0]
    base = NetworkConfiguration.initial(topology)
    traffic = _traffic(topology, TrafficDemand("R0", 2_000_000, destination="R1"))
    zero_queue = LinkAttributes(physical_bandwidth_bps=10_000_000, bandwidth_bps=1_000_000, capacity_max_bps=10_000_000, capacity_bps=1_000_000, queue_packets=0)
    queued = LinkAttributes(physical_bandwidth_bps=10_000_000, bandwidth_bps=1_000_000, capacity_max_bps=10_000_000, capacity_bps=1_000_000, queue_packets=1000)
    loss = LinkAttributes(physical_bandwidth_bps=10_000_000, bandwidth_bps=1_000_000, capacity_max_bps=10_000_000, capacity_bps=1_000_000, queue_packets=1000, loss_rate=0.25)
    a = simulate_deterministic(topology, base.with_updates(performance={link.link_id: zero_queue}), traffic)
    b = simulate_deterministic(topology, base.with_updates(performance={link.link_id: queued}), traffic)
    c = simulate_deterministic(topology, base.with_updates(performance={link.link_id: loss}), traffic)
    assert a.dropped_bps > b.dropped_bps  # finite queue is a burst allowance in this static window
    assert c.delivered_bps < b.delivered_bps and c.directed_link_loads[0].loss_dropped_bps > 0
    capacity_limited = LinkAttributes(physical_bandwidth_bps=10_000_000, bandwidth_bps=1_000_000, capacity_max_bps=10_000_000, capacity_bps=500_000, queue_packets=0)
    assert simulate_deterministic(topology, base.with_updates(performance={link.link_id: capacity_limited}), traffic).delivered_bps < a.delivered_bps
    small_packets = LinkAttributes(physical_bandwidth_bps=10_000_000, bandwidth_bps=1_000_000, capacity_max_bps=10_000_000, capacity_bps=1_000_000, queue_packets=100, packet_size_bytes=500)
    large_packets = LinkAttributes(physical_bandwidth_bps=10_000_000, bandwidth_bps=1_000_000, capacity_max_bps=10_000_000, capacity_bps=1_000_000, queue_packets=100, packet_size_bytes=1500)
    assert simulate_deterministic(topology, base.with_updates(performance={link.link_id: small_packets}), traffic).dropped_bps > simulate_deterministic(topology, base.with_updates(performance={link.link_id: large_packets}), traffic).dropped_bps
    down = simulate_deterministic(topology, base.with_updates(link_states={link.link_id: "down"}), traffic)
    node_down = simulate_deterministic(topology, base.with_updates(node_states={"R1": "down"}), traffic)
    assert down.unreachable_bps == 2_000_000 and node_down.unreachable_bps == 2_000_000
    for result in (a, b, c, down, node_down):
        _assert_conserved(result)
