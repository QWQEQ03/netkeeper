from __future__ import annotations

from netkeeper_sim.routing.ospf import compute_ospf_routes


def test_link_failure_recomputes_alternate_path(diamond_topology):
    diamond_topology.fail_link("R1", "R2")

    table = compute_ospf_routes(diamond_topology)

    assert table["R1"]["R4"].reachable is True
    assert table["R1"]["R4"].cost == 2
    assert table["R1"]["R4"].next_hops == ["R3"]


def test_link_restore_recovers_ecmp(diamond_topology):
    diamond_topology.fail_link("R1", "R2")
    diamond_topology.restore_link("R1", "R2")

    table = compute_ospf_routes(diamond_topology)

    assert table["R1"]["R4"].next_hops == ["R2", "R3"]


def test_ospf_weight_update_changes_next_hop(diamond_topology):
    diamond_topology.update_ospf_weight("R2", "R4", 10)

    table = compute_ospf_routes(diamond_topology)

    assert table["R1"]["R4"].cost == 2
    assert table["R1"]["R4"].next_hops == ["R3"]
