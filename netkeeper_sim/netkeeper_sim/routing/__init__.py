"""Routing algorithms."""

from netkeeper_sim.routing.bgp import (
    BGPRoute,
    BGPRouteTable,
    best_route_for_prefix,
    select_best_route,
    select_best_routes,
)
from netkeeper_sim.routing.ecmp import ecmp_next_hops, has_ecmp
from netkeeper_sim.routing.ospf import ForwardingEntry, ForwardingTable, compute_ospf_routes

__all__ = [
    "BGPRoute",
    "BGPRouteTable",
    "ForwardingEntry",
    "ForwardingTable",
    "best_route_for_prefix",
    "compute_ospf_routes",
    "ecmp_next_hops",
    "has_ecmp",
    "select_best_route",
    "select_best_routes",
]
