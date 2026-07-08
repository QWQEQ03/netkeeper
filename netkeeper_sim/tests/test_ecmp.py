from __future__ import annotations

from netkeeper_sim.routing.ecmp import ecmp_next_hops, has_ecmp
from netkeeper_sim.routing.ospf import compute_ospf_routes
from netkeeper_sim.topology.model import build_topology_from_edges


def test_diamond_topology_preserves_equal_cost_next_hops(diamond_topology):
    table = compute_ospf_routes(diamond_topology)

    assert table["R1"]["R4"].cost == 2
    assert ecmp_next_hops(table, "R1", "R4") == ["R2", "R3"]
    assert has_ecmp(table, "R1", "R4") is True


def test_non_equal_path_does_not_split(diamond_topology):
    diamond_topology.update_ospf_weight("R1", "R3", 5)

    table = compute_ospf_routes(diamond_topology)

    assert table["R1"]["R4"].cost == 2
    assert ecmp_next_hops(table, "R1", "R4") == ["R2"]
    assert has_ecmp(table, "R1", "R4") is False


def test_ospf_preserves_all_equal_cost_next_hops():
    topology = build_topology_from_edges(
        [
            ("R1", "R2", 1),
            ("R2", "R4", 1),
            ("R1", "R3", 1),
            ("R3", "R4", 1),
            ("R1", "R5", 1),
            ("R5", "R4", 1),
        ]
    )

    table = compute_ospf_routes(topology)

    assert table["R1"]["R4"].cost == 2
    assert ecmp_next_hops(table, "R1", "R4") == ["R2", "R3", "R5"]
