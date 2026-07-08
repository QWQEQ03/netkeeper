from __future__ import annotations

from pathlib import Path

import pytest

from netkeeper_sim.topology.model import Topology, build_topology_from_edges


@pytest.fixture
def single_path_topology() -> Topology:
    return build_topology_from_edges(
        [
            ("R1", "R2", 1),
            ("R2", "R3", 1),
        ]
    )


@pytest.fixture
def diamond_topology() -> Topology:
    return build_topology_from_edges(
        [
            ("R1", "R2", 1),
            ("R2", "R4", 1),
            ("R1", "R3", 1),
            ("R3", "R4", 1),
        ]
    )


@pytest.fixture
def topology_zoo_root() -> Path:
    return Path(__file__).resolve().parents[2] / "InternetTopologyZoo"
