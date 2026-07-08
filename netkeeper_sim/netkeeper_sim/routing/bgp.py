from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Iterable

from netkeeper_sim.routing.ospf import ForwardingTable


@dataclass(frozen=True)
class BGPRoute:
    prefix: str
    next_hop: str
    local_preference: int
    as_path: tuple[int, ...]
    med: int
    origin_router: str
    learned_from: str
    igp_cost_to_next_hop: float | None = None


BGPRouteTable = dict[str, dict[str, BGPRoute]]


def select_best_route(routes: Iterable[BGPRoute]) -> BGPRoute:
    route_list = list(routes)
    if not route_list:
        raise ValueError("At least one BGP route is required")
    return min(route_list, key=_route_sort_key)


def select_best_routes(
    candidate_routes: dict[str, dict[str, Iterable[BGPRoute]]],
    forwarding_table: ForwardingTable | None = None,
) -> BGPRouteTable:
    selected: BGPRouteTable = {}
    for router, prefixes in candidate_routes.items():
        selected[router] = {}
        for prefix, routes in prefixes.items():
            enriched_routes = [
                _with_igp_cost(route, router, forwarding_table)
                for route in routes
            ]
            selected[router][prefix] = select_best_route(enriched_routes)
    return selected


def best_route_for_prefix(
    route_table: BGPRouteTable,
    router: str,
    prefix: str,
) -> BGPRoute | None:
    return route_table.get(router, {}).get(prefix)


def _route_sort_key(route: BGPRoute) -> tuple[object, ...]:
    igp_cost = inf if route.igp_cost_to_next_hop is None else route.igp_cost_to_next_hop
    return (
        -route.local_preference,
        len(route.as_path),
        route.med,
        igp_cost,
        route.origin_router,
        route.next_hop,
        route.learned_from,
        route.prefix,
    )


def _with_igp_cost(
    route: BGPRoute,
    router: str,
    forwarding_table: ForwardingTable | None,
) -> BGPRoute:
    if route.igp_cost_to_next_hop is not None or forwarding_table is None:
        return route
    entry = forwarding_table.get(router, {}).get(route.next_hop)
    igp_cost = entry.cost if entry is not None and entry.reachable else inf
    return BGPRoute(
        prefix=route.prefix,
        next_hop=route.next_hop,
        local_preference=route.local_preference,
        as_path=route.as_path,
        med=route.med,
        origin_router=route.origin_router,
        learned_from=route.learned_from,
        igp_cost_to_next_hop=igp_cost,
    )
