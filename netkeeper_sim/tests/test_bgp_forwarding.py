from __future__ import annotations

from netkeeper_sim.routing.bgp import BGPRoute
from netkeeper_sim.routing.ospf import compute_ospf_routes
from netkeeper_sim.traffic.matrix import PrefixTrafficDemand, PrefixTrafficMatrix
from netkeeper_sim.traffic.propagation import propagate_bgp_traffic


def test_bgp_prefix_without_route_is_recorded_unreachable(diamond_topology):
    forwarding_table = compute_ospf_routes(diamond_topology)
    traffic = PrefixTrafficMatrix(
        (PrefixTrafficDemand("R1", "203.0.113.0/24", 25.0),)
    )

    result = propagate_bgp_traffic(
        diamond_topology,
        forwarding_table,
        bgp_routes={},
        traffic_matrix=traffic,
    )

    assert result.delivered_traffic == 0.0
    assert result.dropped_traffic == 25.0
    assert result.is_flow_conserved()
    assert len(result.unreachable_demands) == 1
    assert result.unreachable_demands[0].destination == "203.0.113.0/24"
    assert result.unreachable_demands[0].reason == "no_bgp_route"


def test_bgp_exit_can_be_unreachable_by_igp(single_path_topology):
    single_path_topology.fail_link("R2", "R3")
    forwarding_table = compute_ospf_routes(single_path_topology)
    bgp_routes = {
        "R1": {
            "203.0.113.0/24": BGPRoute(
                prefix="203.0.113.0/24",
                next_hop="R3",
                local_preference=100,
                as_path=(65001,),
                med=100,
                origin_router="R3",
                learned_from="peer",
                igp_cost_to_next_hop=None,
            )
        }
    }
    traffic = PrefixTrafficMatrix(
        (PrefixTrafficDemand("R1", "203.0.113.0/24", 25.0),)
    )

    result = propagate_bgp_traffic(
        single_path_topology,
        forwarding_table,
        bgp_routes,
        traffic,
    )

    assert result.delivered_traffic == 0.0
    assert result.dropped_traffic == 25.0
    assert result.is_flow_conserved()
    assert len(result.unreachable_demands) == 1
    assert result.unreachable_demands[0].destination == "203.0.113.0/24"
    assert result.unreachable_demands[0].reason == "unreachable"
