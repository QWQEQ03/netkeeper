from __future__ import annotations

from netkeeper_sim.routing.ospf import ForwardingTable


def ecmp_next_hops(table: ForwardingTable, source: str, destination: str) -> list[str]:
    return list(table[source][destination].next_hops)


def has_ecmp(table: ForwardingTable, source: str, destination: str) -> bool:
    return len(ecmp_next_hops(table, source, destination)) > 1
