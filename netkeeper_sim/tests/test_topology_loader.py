from __future__ import annotations

from netkeeper_sim.topology.loader import load_topology


def test_loads_real_topology_zoo_graphml(topology_zoo_root):
    topology = load_topology(topology_zoo_root / "graphml" / "Abilene.graphml")

    assert topology.node_count == 11
    assert topology.edge_count == 14
    assert topology.nodes["0"].name == "New York"
    assert topology.nodes["0"].latitude == 40.71427
    assert topology.nodes["0"].longitude == -74.00597

    first_link = next(iter(topology.links.values()))
    assert first_link.ospf_weight == 1.0
    assert first_link.capacity == 100.0
    assert first_link.queue_length == 100


def test_loads_real_topology_zoo_gml(topology_zoo_root):
    topology = load_topology(topology_zoo_root / "gml" / "Abilene.gml")

    assert topology.node_count == 11
    assert topology.edge_count == 14
    assert topology.nodes["0"].name == "New York"


def test_missing_topology_zoo_link_attributes_use_config_defaults(
    topology_zoo_root,
    tmp_path,
):
    config = tmp_path / "defaults.yaml"
    config.write_text(
        "\n".join(
            [
                "topology_defaults:",
                "  ospf_weight: 7",
                "  bandwidth: 123.0",
                "  capacity: 456.0",
                "  queue_length: 321",
                "  loss_rate: 0.25",
                "  propagation_delay: 9.5",
            ]
        ),
        encoding="utf-8",
    )

    topology = load_topology(
        topology_zoo_root / "graphml" / "Abilene.graphml",
        config_path=config,
    )
    first_link = next(iter(topology.links.values()))

    assert first_link.ospf_weight == 7.0
    assert first_link.bandwidth == 123.0
    assert first_link.capacity == 456.0
    assert first_link.queue_length == 321
    assert first_link.loss_rate == 0.25
    assert first_link.propagation_delay == 9.5
