from __future__ import annotations

from netkeeper_sim.routing.bgp import BGPRoute, select_best_route, select_best_routes
from netkeeper_sim.routing.ospf import compute_ospf_routes
from netkeeper_sim.traffic.matrix import PrefixTrafficDemand, PrefixTrafficMatrix
from netkeeper_sim.traffic.propagation import propagate_bgp_traffic


def route(
    next_hop: str,
    local_preference: int = 100,
    as_path: tuple[int, ...] = (65001,),
    med: int = 100,
    origin_router: str = "R9",
    learned_from: str = "peer",
    igp_cost_to_next_hop: float | None = 1,
) -> BGPRoute:
    return BGPRoute(
        prefix="203.0.113.0/24",
        next_hop=next_hop,
        local_preference=local_preference,
        as_path=as_path,
        med=med,
        origin_router=origin_router,
        learned_from=learned_from,
        igp_cost_to_next_hop=igp_cost_to_next_hop,
    )


def test_local_preference_wins_first():
    best = select_best_route(
        [
            route("R2", local_preference=100, as_path=(1,)),
            route("R3", local_preference=200, as_path=(1, 2, 3), med=999),
        ]
    )

    assert best.next_hop == "R3"


def test_shorter_as_path_wins_after_local_preference_tie():
    best = select_best_route(
        [
            route("R2", as_path=(1, 2, 3)),
            route("R3", as_path=(1, 2)),
        ]
    )

    assert best.next_hop == "R3"


def test_lower_med_wins_after_as_path_tie():
    best = select_best_route(
        [
            route("R2", med=50),
            route("R3", med=10),
        ]
    )

    assert best.next_hop == "R3"


def test_lower_igp_cost_to_next_hop_wins_after_med_tie():
    best = select_best_route(
        [
            route("R2", igp_cost_to_next_hop=20),
            route("R3", igp_cost_to_next_hop=5),
        ]
    )

    assert best.next_hop == "R3"


def test_router_id_breaks_final_tie_deterministically():
    best = select_best_route(
        [
            route("R3", origin_router="R3", learned_from="peer-b"),
            route("R2", origin_router="R2", learned_from="peer-a"),
        ]
    )

    assert best.origin_router == "R2"
    assert best.next_hop == "R2"


def test_select_best_routes_for_each_router_prefix():
    candidates = {
        "R1": {
            "203.0.113.0/24": [
                route("R2", local_preference=100),
                route("R3", local_preference=120),
            ],
            "198.51.100.0/24": [
                BGPRoute(
                    prefix="198.51.100.0/24",
                    next_hop="R2",
                    local_preference=100,
                    as_path=(65001,),
                    med=100,
                    origin_router="R2",
                    learned_from="peer",
                    igp_cost_to_next_hop=1,
                )
            ],
        }
    }

    selected = select_best_routes(candidates)

    assert selected["R1"]["203.0.113.0/24"].next_hop == "R3"
    assert selected["R1"]["198.51.100.0/24"].next_hop == "R2"


def test_bgp_uses_ospf_cost_when_route_cost_is_missing(diamond_topology):
    diamond_topology.update_ospf_weight("R1", "R3", 5)
    forwarding_table = compute_ospf_routes(diamond_topology)
    candidates = {
        "R1": {
            "203.0.113.0/24": [
                route("R2", igp_cost_to_next_hop=None),
                route("R3", igp_cost_to_next_hop=None),
            ]
        }
    }

    selected = select_best_routes(candidates, forwarding_table)

    assert selected["R1"]["203.0.113.0/24"].next_hop == "R2"
    assert selected["R1"]["203.0.113.0/24"].igp_cost_to_next_hop == 1


def test_bgp_prefix_traffic_uses_ospf_ecmp_to_selected_exit(diamond_topology):
    forwarding_table = compute_ospf_routes(diamond_topology)
    selected = {
        "R1": {
            "203.0.113.0/24": route(
                "R4",
                local_preference=200,
                igp_cost_to_next_hop=2,
            )
        }
    }
    traffic = PrefixTrafficMatrix(
        (PrefixTrafficDemand("R1", "203.0.113.0/24", 100.0),)
    )

    result = propagate_bgp_traffic(
        diamond_topology,
        forwarding_table,
        selected,
        traffic,
    )

    assert result.delivered_traffic == 100.0
    assert result.dropped_traffic == 0.0
    assert result.is_flow_conserved()
    assert result.flow_paths[("R1", "R4")] == {
        ("R1", "R2"): 50.0,
        ("R2", "R4"): 50.0,
        ("R1", "R3"): 50.0,
        ("R3", "R4"): 50.0,
    }
