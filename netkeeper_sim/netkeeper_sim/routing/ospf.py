from __future__ import annotations

from dataclasses import dataclass
from math import inf, isclose

import networkx as nx

from netkeeper_sim.topology.model import Topology


@dataclass(frozen=True)
class ForwardingEntry:
    cost: float
    next_hops: list[str]
    reachable: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "cost": self.cost,
            "next_hops": list(self.next_hops),
            "reachable": self.reachable,
        }


ForwardingTable = dict[str, dict[str, ForwardingEntry]]


def compute_ospf_routes(topology: Topology, tolerance: float = 1e-9) -> ForwardingTable:
    routing_graph = topology.to_routing_graph()
    nodes = sorted(topology.nodes)
    all_lengths = _all_pairs_lengths(routing_graph)
    table: ForwardingTable = {}

    for source in nodes:
        source_entries: dict[str, ForwardingEntry] = {}
        source_lengths = all_lengths.get(source, {})
        for destination in nodes:
            if source == destination:
                source_entries[destination] = ForwardingEntry(
                    cost=0.0,
                    next_hops=[],
                    reachable=True,
                )
                continue

            cost = source_lengths.get(destination, inf)
            if cost == inf:
                source_entries[destination] = ForwardingEntry(
                    cost=inf,
                    next_hops=[],
                    reachable=False,
                )
                continue

            next_hops = _find_equal_cost_next_hops(
                routing_graph,
                all_lengths,
                source,
                destination,
                cost,
                tolerance,
            )
            source_entries[destination] = ForwardingEntry(
                cost=cost,
                next_hops=next_hops,
                reachable=bool(next_hops),
            )
        table[source] = source_entries

    return table


def forwarding_table_to_dict(table: ForwardingTable) -> dict[str, dict[str, dict[str, object]]]:
    return {
        source: {
            destination: entry.to_dict()
            for destination, entry in destinations.items()
        }
        for source, destinations in table.items()
    }


def _all_pairs_lengths(graph: nx.Graph) -> dict[str, dict[str, float]]:
    return {
        source: dict(lengths)
        for source, lengths in nx.all_pairs_dijkstra_path_length(
            graph,
            weight="ospf_weight",
        )
    }


def _find_equal_cost_next_hops(
    graph: nx.Graph,
    all_lengths: dict[str, dict[str, float]],
    source: str,
    destination: str,
    target_cost: float,
    tolerance: float,
) -> list[str]:
    next_hops: list[str] = []
    for neighbor in sorted(graph.neighbors(source)):
        edge_weight = graph[source][neighbor]["ospf_weight"]
        neighbor_cost = all_lengths.get(neighbor, {}).get(destination, inf)
        total_cost = edge_weight + neighbor_cost
        if isclose(total_cost, target_cost, rel_tol=tolerance, abs_tol=tolerance):
            next_hops.append(neighbor)
    return next_hops
