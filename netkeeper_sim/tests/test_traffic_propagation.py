from __future__ import annotations

from math import isclose

from netkeeper_sim.metrics.load import calculate_link_load_metrics
from netkeeper_sim.routing.ospf import compute_ospf_routes
from netkeeper_sim.traffic.matrix import TrafficDemand, TrafficMatrix
from netkeeper_sim.traffic.propagation import propagate_traffic


def test_single_path_traffic_loads_and_utilization(single_path_topology):
    table = compute_ospf_routes(single_path_topology)
    traffic = TrafficMatrix((TrafficDemand("R1", "R3", 100.0),))

    result = propagate_traffic(single_path_topology, table, traffic)
    metrics = calculate_link_load_metrics(single_path_topology, result.link_loads)

    assert result.delivered_traffic == 100.0
    assert result.dropped_traffic == 0.0
    assert result.is_flow_conserved()
    assert result.flow_paths[("R1", "R3")] == {
        ("R1", "R2"): 100.0,
        ("R2", "R3"): 100.0,
    }
    assert sorted(metrics.total_load.values()) == [100.0, 100.0]
    assert sorted(metrics.utilization.values()) == [1.0, 1.0]
    assert metrics.maximum_link_utilization == 1.0


def test_ecmp_traffic_splits_equally(diamond_topology):
    table = compute_ospf_routes(diamond_topology)
    traffic = TrafficMatrix((TrafficDemand("R1", "R4", 100.0),))

    result = propagate_traffic(diamond_topology, table, traffic)
    metrics = calculate_link_load_metrics(diamond_topology, result.link_loads)

    assert result.delivered_traffic == 100.0
    assert result.dropped_traffic == 0.0
    assert result.is_flow_conserved()
    assert result.flow_paths[("R1", "R4")] == {
        ("R1", "R2"): 50.0,
        ("R2", "R4"): 50.0,
        ("R1", "R3"): 50.0,
        ("R3", "R4"): 50.0,
    }
    assert sorted(metrics.total_load.values()) == [50.0, 50.0, 50.0, 50.0]
    assert metrics.maximum_link_utilization == 0.5


def test_non_equal_path_traffic_uses_single_path(diamond_topology):
    diamond_topology.update_ospf_weight("R1", "R3", 5)
    table = compute_ospf_routes(diamond_topology)
    traffic = TrafficMatrix((TrafficDemand("R1", "R4", 100.0),))

    result = propagate_traffic(diamond_topology, table, traffic)

    assert result.flow_paths[("R1", "R4")] == {
        ("R1", "R2"): 100.0,
        ("R2", "R4"): 100.0,
    }
    assert result.is_flow_conserved()


def test_unreachable_demand_is_detected(single_path_topology):
    single_path_topology.fail_link("R2", "R3")
    table = compute_ospf_routes(single_path_topology)
    traffic = TrafficMatrix((TrafficDemand("R1", "R3", 100.0),))

    result = propagate_traffic(single_path_topology, table, traffic)

    assert result.delivered_traffic == 0.0
    assert result.dropped_traffic == 100.0
    assert len(result.unreachable_demands) == 1
    assert result.unreachable_demands[0].reason == "unreachable"
    assert result.is_flow_conserved()


def test_multiple_demands_preserve_flow(diamond_topology):
    table = compute_ospf_routes(diamond_topology)
    traffic = TrafficMatrix(
        (
            TrafficDemand("R1", "R4", 100.0),
            TrafficDemand("R4", "R1", 20.0),
            TrafficDemand("R2", "R3", 10.0),
        )
    )

    result = propagate_traffic(diamond_topology, table, traffic)

    assert isclose(result.total_input_traffic, 130.0)
    assert isclose(result.delivered_traffic, 130.0)
    assert result.dropped_traffic == 0.0
    assert result.is_flow_conserved()


def test_duplicate_demands_aggregate_flow_paths_and_loads(single_path_topology):
    table = compute_ospf_routes(single_path_topology)
    traffic = TrafficMatrix(
        (
            TrafficDemand("R1", "R3", 40.0),
            TrafficDemand("R1", "R3", 60.0),
        )
    )

    result = propagate_traffic(single_path_topology, table, traffic)
    metrics = calculate_link_load_metrics(single_path_topology, result.link_loads)

    assert result.flow_paths[("R1", "R3")] == {
        ("R1", "R2"): 100.0,
        ("R2", "R3"): 100.0,
    }
    assert sorted(metrics.total_load.values()) == [100.0, 100.0]
    assert result.is_flow_conserved()


def test_random_traffic_on_real_topology_zoo_graphml(topology_zoo_root):
    from netkeeper_sim.topology.loader import load_topology

    topology = load_topology(topology_zoo_root / "graphml" / "Abilene.graphml")
    table = compute_ospf_routes(topology)
    traffic = TrafficMatrix.random(
        sorted(topology.nodes),
        seed=7,
        max_demand=20.0,
        density=0.25,
    )

    result = propagate_traffic(topology, table, traffic)
    metrics = calculate_link_load_metrics(topology, result.link_loads)

    assert topology.node_count == 11
    assert topology.edge_count == 14
    assert len(traffic) > 0
    assert result.dropped_traffic == 0.0
    assert result.is_flow_conserved()
    assert metrics.maximum_link_utilization >= 0.0
