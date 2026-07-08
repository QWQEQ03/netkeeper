from __future__ import annotations

from netkeeper_sim.simulator.environment import NetworkSimulationEnvironment
from netkeeper_sim.traffic.matrix import TrafficDemand, TrafficMatrix


def test_environment_loads_topology_and_computes_routes(topology_zoo_root):
    env = NetworkSimulationEnvironment()
    topology = env.load_topology(topology_zoo_root / "graphml" / "Abilene.graphml")
    table = env.compute_ospf_routes()

    assert topology.node_count == 11
    assert topology.edge_count == 14
    assert table["0"]["10"].reachable is True


def test_environment_propagates_traffic(single_path_topology):
    env = NetworkSimulationEnvironment()
    env.topology = single_path_topology
    env.set_traffic_matrix(TrafficMatrix((TrafficDemand("R1", "R3", 100.0),)))

    env.compute_ospf_routes()
    propagation = env.propagate_traffic()
    metrics = env.calculate_metrics()

    assert propagation.delivered_traffic == 100.0
    assert propagation.is_flow_conserved()
    assert metrics.maximum_link_utilization == 1.0
