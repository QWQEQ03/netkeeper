from __future__ import annotations

from math import inf

from netkeeper_sim.routing.ospf import compute_ospf_routes


def test_single_path_shortest_route(single_path_topology):
    table = compute_ospf_routes(single_path_topology)

    assert table["R1"]["R3"].reachable is True
    assert table["R1"]["R3"].cost == 2
    assert table["R1"]["R3"].next_hops == ["R2"]


def test_unreachable_after_link_failure(single_path_topology):
    single_path_topology.fail_link("R2", "R3")

    table = compute_ospf_routes(single_path_topology)

    assert table["R1"]["R3"].reachable is False
    assert table["R1"]["R3"].cost == inf
    assert table["R1"]["R3"].next_hops == []
