from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import isclose

from netkeeper_sim.routing.bgp import BGPRouteTable, best_route_for_prefix
from netkeeper_sim.routing.ospf import ForwardingTable
from netkeeper_sim.topology.model import Topology
from netkeeper_sim.traffic.matrix import (
    PrefixTrafficMatrix,
    TrafficDemand,
    TrafficMatrix,
)


@dataclass(frozen=True)
class UnreachableDemand:
    source: str
    destination: str
    demand: float
    reason: str


@dataclass
class PropagationResult:
    flow_paths: dict[tuple[str, str], dict[tuple[str, str], float]]
    link_loads: dict[str, float]
    unreachable_demands: list[UnreachableDemand] = field(default_factory=list)
    delivered_traffic: float = 0.0
    dropped_traffic: float = 0.0
    total_input_traffic: float = 0.0

    def is_flow_conserved(self, tolerance: float = 1e-9) -> bool:
        return isclose(
            self.total_input_traffic,
            self.delivered_traffic + self.dropped_traffic,
            rel_tol=tolerance,
            abs_tol=tolerance,
        )


def propagate_traffic(
    topology: Topology,
    forwarding_table: ForwardingTable,
    traffic_matrix: TrafficMatrix,
    tolerance: float = 1e-9,
) -> PropagationResult:
    flow_paths: dict[tuple[str, str], dict[tuple[str, str], float]] = {}
    link_loads: dict[str, float] = {link_id: 0.0 for link_id in topology.links}
    unreachable: list[UnreachableDemand] = []
    delivered = 0.0
    dropped = 0.0
    total_input = 0.0

    for demand in traffic_matrix:
        total_input += demand.demand
        demand_edges: defaultdict[tuple[str, str], float] = defaultdict(float)
        delivered_part, dropped_part, unreachable_part = _propagate_one_demand(
            topology,
            forwarding_table,
            demand,
            demand_edges,
            link_loads,
            tolerance,
        )
        delivered += delivered_part
        dropped += dropped_part
        unreachable.extend(unreachable_part)
        flow_key = (demand.source, demand.destination)
        aggregate_edges = flow_paths.setdefault(flow_key, {})
        for edge, amount in demand_edges.items():
            aggregate_edges[edge] = aggregate_edges.get(edge, 0.0) + amount

    return PropagationResult(
        flow_paths=flow_paths,
        link_loads=link_loads,
        unreachable_demands=unreachable,
        delivered_traffic=delivered,
        dropped_traffic=dropped,
        total_input_traffic=total_input,
    )


def propagate_bgp_traffic(
    topology: Topology,
    forwarding_table: ForwardingTable,
    bgp_routes: BGPRouteTable,
    traffic_matrix: PrefixTrafficMatrix,
    tolerance: float = 1e-9,
) -> PropagationResult:
    flow_paths: dict[tuple[str, str], dict[tuple[str, str], float]] = {}
    link_loads: dict[str, float] = {link_id: 0.0 for link_id in topology.links}
    unreachable: list[UnreachableDemand] = []
    delivered = 0.0
    dropped = 0.0
    total_input = 0.0

    for demand in traffic_matrix:
        total_input += demand.demand
        route = best_route_for_prefix(bgp_routes, demand.source, demand.prefix)
        if route is None:
            dropped += demand.demand
            unreachable.append(
                UnreachableDemand(
                    demand.source,
                    demand.prefix,
                    demand.demand,
                    "no_bgp_route",
                )
            )
            continue
        node_result = propagate_traffic(
            topology,
            forwarding_table,
            TrafficMatrix(
                (
                    TrafficDemand(
                        source=demand.source,
                        destination=route.next_hop,
                        demand=demand.demand,
                    ),
                )
            ),
            tolerance=tolerance,
        )
        delivered += node_result.delivered_traffic
        dropped += node_result.dropped_traffic
        for link_id, load in node_result.link_loads.items():
            link_loads[link_id] += load
        for flow_key, edges in node_result.flow_paths.items():
            aggregate_edges = flow_paths.setdefault(flow_key, {})
            for edge, amount in edges.items():
                aggregate_edges[edge] = aggregate_edges.get(edge, 0.0) + amount
        for unreachable_demand in node_result.unreachable_demands:
            unreachable.append(
                UnreachableDemand(
                    demand.source,
                    demand.prefix,
                    unreachable_demand.demand,
                    unreachable_demand.reason,
                )
            )

    return PropagationResult(
        flow_paths=flow_paths,
        link_loads=link_loads,
        unreachable_demands=unreachable,
        delivered_traffic=delivered,
        dropped_traffic=dropped,
        total_input_traffic=total_input,
    )


def _propagate_one_demand(
    topology: Topology,
    forwarding_table: ForwardingTable,
    demand: TrafficDemand,
    demand_edges: defaultdict[tuple[str, str], float],
    link_loads: dict[str, float],
    tolerance: float,
) -> tuple[float, float, list[UnreachableDemand]]:
    delivered = 0.0
    dropped = 0.0
    unreachable: list[UnreachableDemand] = []
    stack: list[tuple[str, float, tuple[str, ...]]] = [
        (demand.source, demand.demand, (demand.source,))
    ]
    max_hops = max(1, len(topology.nodes) * 2)

    while stack:
        current, amount, path = stack.pop()
        if amount <= tolerance:
            continue
        if current == demand.destination:
            delivered += amount
            continue
        if len(path) > max_hops:
            dropped += amount
            unreachable.append(
                UnreachableDemand(
                    demand.source,
                    demand.destination,
                    amount,
                    "max_hops_exceeded",
                )
            )
            continue

        entry = forwarding_table.get(current, {}).get(demand.destination)
        if entry is None or not entry.reachable or not entry.next_hops:
            dropped += amount
            unreachable.append(
                UnreachableDemand(
                    demand.source,
                    demand.destination,
                    amount,
                    "unreachable",
                )
            )
            continue

        next_hops = list(entry.next_hops)
        split_amount = amount / len(next_hops)
        for next_hop in next_hops:
            if next_hop in path:
                dropped += split_amount
                unreachable.append(
                    UnreachableDemand(
                        demand.source,
                        demand.destination,
                        split_amount,
                        "routing_loop",
                    )
                )
                continue
            link_ids = topology.active_routing_link_ids_between(current, next_hop)
            if not link_ids:
                dropped += split_amount
                unreachable.append(
                    UnreachableDemand(
                        demand.source,
                        demand.destination,
                        split_amount,
                        "missing_active_link",
                    )
                )
                continue

            demand_edges[(current, next_hop)] += split_amount
            per_physical_link = split_amount / len(link_ids)
            for link_id in link_ids:
                link_loads[link_id] += per_physical_link
            stack.append((next_hop, split_amount, (*path, next_hop)))

    return delivered, dropped, unreachable
