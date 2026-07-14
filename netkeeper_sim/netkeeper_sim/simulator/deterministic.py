"""Schema-driven deterministic routing, forwarding, and performance kernel.

This is deliberately a static-window fluid model, not a device protocol
emulator.  It is the authoritative path for new environments; legacy classes
remain available for compatibility with the original tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Mapping

import networkx as nx

from netkeeper_sim.schemas import (
    BGPRoute,
    DirectedLinkLoad,
    Metrics,
    NetworkConfiguration,
    RoutingEntry,
    Topology,
    TrafficMatrix,
)


FLOW_TOLERANCE_BPS = 1e-6
MAX_FIXED_POINT_ITERATIONS = 80


@dataclass(frozen=True)
class SelectedBGPRoute:
    router_id: str
    prefix: str
    next_hop: str
    local_preference: int
    as_path_length: int
    med: int
    igp_cost: float


@dataclass(frozen=True)
class DemandOutcome:
    demand_index: int
    source: str
    destination: str | None
    prefix: str | None
    offered_bps: float
    delivered_bps: float
    dropped_bps: float
    unreachable_bps: float

    def __post_init__(self) -> None:
        if abs(self.offered_bps - self.delivered_bps - self.dropped_bps - self.unreachable_bps) > FLOW_TOLERANCE_BPS:
            raise ValueError("demand traffic is not conserved")


@dataclass(frozen=True)
class CongestionInfo:
    arc_id: str
    service_rate_bps: float
    queue_capacity_bits: float
    queue_occupancy_bits: float
    queue_dropped_bps: float
    loss_dropped_bps: float


@dataclass(frozen=True)
class DeterministicSimulationResult:
    routing_table: tuple[RoutingEntry, ...]
    selected_bgp_routes: tuple[SelectedBGPRoute, ...]
    directed_link_loads: tuple[DirectedLinkLoad, ...]
    demand_outcomes: tuple[DemandOutcome, ...]
    metrics: Metrics
    congestion: tuple[CongestionInfo, ...]

    @property
    def offered_bps(self) -> float:
        return self.metrics.total_input_bps

    @property
    def delivered_bps(self) -> float:
        return self.metrics.delivered_bps

    @property
    def dropped_bps(self) -> float:
        return self.metrics.dropped_bps

    @property
    def unreachable_bps(self) -> float:
        return self.metrics.unreachable_bps


def simulate_deterministic(
    topology: Topology,
    configuration: NetworkConfiguration,
    traffic: TrafficMatrix,
    *,
    previous: DeterministicSimulationResult | None = None,
) -> DeterministicSimulationResult:
    """Recompute all route and load state for one immutable configuration.

    Any configuration/version change calls this stateless function again, so
    no OSPF/BGP cache can survive a weight or failure change.
    """
    if configuration.topology_id != topology.topology_id:
        raise ValueError("configuration does not belong to topology")
    routing, distances, eligible, next_hop_map = _compute_ospf(topology, configuration)
    selected = _select_bgp(topology, configuration, routing)
    resolved, initial_unreachable = _resolve_demands(topology, configuration, traffic, routing, selected)
    scales: dict[str, float] = {arc.arc_id: 1.0 for arc in topology.arcs}
    offers: dict[str, float] = {}
    delivered: dict[int, float] = {}
    for _ in range(MAX_FIXED_POINT_ITERATIONS):
        offers, delivered, dynamic_unreachable = _propagate(
            topology, traffic, resolved, distances, eligible, next_hop_map, scales
        )
        updated_scales, _loads, _congestion = _performance(
            topology, configuration, traffic.interval_seconds, offers
        )
        delta = max((abs(updated_scales[key] - scales.get(key, 1.0)) for key in updated_scales), default=0.0)
        scales = updated_scales
        if delta <= 1e-10:
            break
    # One final pass makes the outcome correspond to the returned arc loads.
    offers, delivered, dynamic_unreachable = _propagate(topology, traffic, resolved, distances, eligible, next_hop_map, scales)
    _scales, loads, congestion = _performance(topology, configuration, traffic.interval_seconds, offers)
    outcomes: list[DemandOutcome] = []
    offered_total = 0.0
    for index, demand in enumerate(traffic.demands):
        offered = demand.traffic_rate_bps * traffic.load_multiplier
        offered_total += offered
        unreachable = initial_unreachable.get(index, 0.0) + dynamic_unreachable.get(index, 0.0)
        arrived = delivered.get(index, 0.0)
        # A disconnected path is classified as unreachable; all other loss is
        # congestion/loss-rate drop.  Clamp protects floating-point iteration.
        arrived = min(max(arrived, 0.0), max(offered - unreachable, 0.0))
        dropped = max(offered - unreachable - arrived, 0.0)
        outcomes.append(DemandOutcome(index, demand.source, demand.destination, demand.prefix, offered, arrived, dropped, unreachable))
    delivered_total = sum(item.delivered_bps for item in outcomes)
    dropped_total = sum(item.dropped_bps for item in outcomes)
    unreachable_total = sum(item.unreachable_bps for item in outcomes)
    mlu = max((item.utilization for item in loads), default=0.0)
    shift = _traffic_shift(previous, selected)
    metrics = Metrics(
        maximum_link_utilization=mlu,
        traffic_shift_paper_v1=shift,
        traffic_shift_project_v1=shift,
        total_input_bps=offered_total,
        admitted_bps=sum(item.admitted_bps for item in loads),
        delivered_bps=delivered_total,
        dropped_bps=dropped_total,
        unreachable_bps=unreachable_total,
        congestion_arc_count=sum(item.congested for item in loads),
    )
    return DeterministicSimulationResult(tuple(routing), tuple(selected), tuple(loads), tuple(outcomes), metrics, tuple(congestion))


def _active_link(topology: Topology, config: NetworkConfiguration, link_id: str) -> bool:
    link = next(link for link in topology.links if link.link_id == link_id)
    return (
        config.link_states.get(link_id, link.attributes.state) == "up"
        and config.node_states.get(link.source, "up") == "up"
        and config.node_states.get(link.target, "up") == "up"
    )


def _compute_ospf(topology: Topology, config: NetworkConfiguration):
    graph = nx.Graph()
    graph.add_nodes_from(node.node_id for node in topology.nodes if config.node_states.get(node.node_id, "up") == "up")
    by_pair: dict[tuple[str, str], list[str]] = {}
    for link in topology.links:
        if not _active_link(topology, config, link.link_id):
            continue
        key = tuple(sorted((link.source, link.target)))
        by_pair.setdefault(key, []).append(link.link_id)
    eligible: dict[tuple[str, str], tuple[str, ...]] = {}
    for (left, right), link_ids in by_pair.items():
        minimum = min(config.ospf_weights.get(link_id, 1) for link_id in link_ids)
        chosen = tuple(sorted(link_id for link_id in link_ids if config.ospf_weights.get(link_id, 1) == minimum))
        graph.add_edge(left, right, weight=minimum)
        eligible[left, right] = eligible[right, left] = chosen
    entries: list[RoutingEntry] = []
    next_hop_map: dict[tuple[str, str], tuple[str, ...]] = {}
    distances: dict[str, dict[str, float]] = {}
    for destination in sorted(graph.nodes):
        dist = nx.single_source_dijkstra_path_length(graph, destination, weight="weight")
        distances[destination] = {node: float(value) for node, value in dist.items()}
        for router in sorted(node.node_id for node in topology.nodes):
            if router not in graph or router not in dist:
                entries.append(RoutingEntry(router, destination, "node", False, ()))
                continue
            if router == destination:
                entries.append(RoutingEntry(router, destination, "node", True, (), 0.0))
                continue
            cost = float(dist[router])
            hops = tuple(sorted(
                neighbor for neighbor in graph.neighbors(router)
                if neighbor in dist and graph[router][neighbor]["weight"] + dist[neighbor] == cost
            ))
            next_hop_map[router, destination] = hops
            entries.append(RoutingEntry(router, destination, "node", bool(hops), hops, cost))
    return entries, distances, eligible, next_hop_map


def _select_bgp(topology: Topology, config: NetworkConfiguration, routing: list[RoutingEntry]) -> list[SelectedBGPRoute]:
    route_map = {(item.router_id, item.destination): item for item in routing}
    candidates: dict[tuple[str, str], list[BGPRoute]] = {}
    for route in config.bgp.routes:
        if route.enabled and route.router_id in {node.node_id for node in topology.nodes}:
            candidates.setdefault((route.router_id, route.prefix), []).append(route)
    selected: list[SelectedBGPRoute] = []
    for (router, prefix), routes in sorted(candidates.items()):
        def key(route: BGPRoute):
            entry = route_map.get((router, route.next_hop))
            igp_cost = entry.cost if entry is not None and entry.reachable and entry.cost is not None else inf
            return (-route.local_preference, len(route.as_path), route.med, igp_cost, route.next_hop, route.router_id, prefix)
        best = min(routes, key=key)
        entry = route_map.get((router, best.next_hop))
        igp = entry.cost if entry is not None and entry.reachable and entry.cost is not None else inf
        if igp != inf:
            selected.append(SelectedBGPRoute(router, prefix, best.next_hop, best.local_preference, len(best.as_path), best.med, igp))
    return selected


def _resolve_demands(topology, config, traffic, routing, selected):
    nodes = {node.node_id for node in topology.nodes}
    routes = {(item.router_id, item.destination): item for item in routing}
    bgp = {(item.router_id, item.prefix): item for item in selected}
    resolved: dict[int, str] = {}
    unreachable: dict[int, float] = {}
    for index, demand in enumerate(traffic.demands):
        amount = demand.traffic_rate_bps * traffic.load_multiplier
        destination = demand.destination
        if demand.prefix is not None:
            selected_route = bgp.get((demand.source, demand.prefix))
            destination = selected_route.next_hop if selected_route else None
        valid = (
            demand.source in nodes and destination in nodes
            and config.node_states.get(demand.source, "up") == "up"
            and config.node_states.get(destination, "up") == "up"
            and (demand.source == destination or routes.get((demand.source, destination), RoutingEntry("", "", "node", False, ())).reachable)
        )
        if valid:
            resolved[index] = destination  # type: ignore[assignment]
        else:
            unreachable[index] = amount
    return resolved, unreachable


def _propagate(topology, traffic, resolved, distances, eligible, next_hop_map, scales):
    offers: dict[str, float] = {}
    arrivals: dict[int, float] = {}
    unreachable: dict[int, float] = {}
    arcs = {(arc.source, arc.target, arc.link_id): arc.arc_id for arc in topology.arcs}
    by_destination: dict[str, list[int]] = {}
    for index, destination in resolved.items():
        by_destination.setdefault(destination, []).append(index)
    for destination, indexes in by_destination.items():
        dist = distances.get(destination, {})
        flow: dict[str, dict[int, float]] = {}
        for index in indexes:
            source = traffic.demands[index].source
            flow.setdefault(source, {})[index] = traffic.demands[index].traffic_rate_bps * traffic.load_multiplier
        # Positive integral OSPF weights make this a destination-directed DAG.
        for router in sorted(flow.keys() | set(dist), key=lambda node: (-dist.get(node, -1.0), node)):
            values = flow.get(router, {})
            if not values:
                continue
            if router == destination:
                for index, amount in values.items():
                    arrivals[index] = arrivals.get(index, 0.0) + amount
                continue
            next_hops = next_hop_map.get((router, destination), ())
            branch_arcs = [(neighbor, link_id) for neighbor in next_hops for link_id in eligible.get((router, neighbor), ())]
            if not branch_arcs:
                for index, amount in values.items():
                    unreachable[index] = unreachable.get(index, 0.0) + amount
                continue
            for index, amount in values.items():
                share = amount / len(branch_arcs)
                for neighbor, link_id in branch_arcs:
                    arc_id = arcs[router, neighbor, link_id]
                    offers[arc_id] = offers.get(arc_id, 0.0) + share
                    flow.setdefault(neighbor, {})[index] = flow.setdefault(neighbor, {}).get(index, 0.0) + share * scales.get(arc_id, 1.0)
    return offers, arrivals, unreachable


def _performance(topology, config, interval, offers):
    loads: list[DirectedLinkLoad] = []
    congestion: list[CongestionInfo] = []
    scales: dict[str, float] = {}
    links = {link.link_id: link for link in topology.links}
    for arc in sorted(topology.arcs, key=lambda item: item.arc_id):
        offered = offers.get(arc.arc_id, 0.0)
        attributes = config.performance.get(arc.link_id, links[arc.link_id].attributes)
        active = _active_link(topology, config, arc.link_id)
        mu = min(attributes.bandwidth_bps, attributes.capacity_bps) if active else 0.0
        queue_bits = attributes.queue_packets * attributes.packet_size_bytes * 8
        # Static-window convention: finite queue is a one-window burst
        # allowance, so no unaccounted "backlog" term violates conservation.
        admitted = min(offered, mu + queue_bits / interval) if mu > 0 else 0.0
        queue_drop = max(offered - admitted, 0.0)
        loss_drop = admitted * attributes.loss_rate
        delivered = max(admitted - loss_drop, 0.0)
        occupancy = min(max(admitted - mu, 0.0) * interval, queue_bits)
        utilization = offered / mu if mu > 0 else (inf if offered > 0 else 0.0)
        congested = queue_drop > FLOW_TOLERANCE_BPS or occupancy > FLOW_TOLERANCE_BPS
        delay_ms = attributes.delay_ms + (occupancy / mu * 1000.0 if mu > 0 else 0.0)
        loads.append(DirectedLinkLoad(arc.arc_id, offered, delivered, queue_drop + loss_drop, utilization, admitted, queue_drop, loss_drop, occupancy, delay_ms, congested))
        scales[arc.arc_id] = delivered / offered if offered > FLOW_TOLERANCE_BPS else 1.0
        if congested or loss_drop > FLOW_TOLERANCE_BPS:
            congestion.append(CongestionInfo(arc.arc_id, mu, queue_bits, occupancy, queue_drop, loss_drop))
    return scales, loads, congestion


def _traffic_shift(previous, current):
    if previous is None:
        return None
    before = {(route.router_id, route.prefix): route.next_hop for route in previous.selected_bgp_routes}
    after = {(route.router_id, route.prefix): route.next_hop for route in current}
    keys = set(before) | set(after)
    return sum(before.get(key) != after.get(key) for key in keys) / len(keys) if keys else 0.0
