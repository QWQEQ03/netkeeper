from __future__ import annotations

from math import inf

from netkeeper_sim.metrics.evaluation import evaluate_netkeeper_metrics
from netkeeper_sim.metrics.load import calculate_link_load_metrics
from netkeeper_sim.metrics.traffic_shift import capture_forwarding_plane_snapshot
from netkeeper_sim.policies import ForwardPolicy
from netkeeper_sim.routing.ospf import compute_ospf_routes
from netkeeper_sim.simulator.environment import NetworkSimulationEnvironment
from netkeeper_sim.topology.model import build_topology_from_edges
from netkeeper_sim.traffic.matrix import TrafficDemand, TrafficMatrix


def test_evaluation_reuses_existing_maximum_link_utilization(single_path_topology):
    link_loads = {link_id: 50.0 for link_id in single_path_topology.links}
    table = compute_ospf_routes(single_path_topology)

    load_metrics = calculate_link_load_metrics(single_path_topology, link_loads)
    evaluation = evaluate_netkeeper_metrics(
        single_path_topology,
        link_loads,
        forwarding_table=table,
    )

    assert evaluation.maximum_link_utilization == load_metrics.maximum_link_utilization
    assert evaluation.link_utilizations == load_metrics.utilization


def test_evaluation_reports_zero_utilization_without_traffic(single_path_topology):
    table = compute_ospf_routes(single_path_topology)

    evaluation = evaluate_netkeeper_metrics(
        single_path_topology,
        {},
        forwarding_table=table,
    )

    assert evaluation.maximum_link_utilization == 0.0
    assert evaluation.average_link_utilization == 0.0
    assert evaluation.overloaded_links == ()


def test_evaluation_preserves_capacity_zero_behavior(single_path_topology):
    first_link = next(iter(single_path_topology.links.values()))
    first_link.capacity = 0.0
    table = compute_ospf_routes(single_path_topology)

    evaluation = evaluate_netkeeper_metrics(
        single_path_topology,
        {first_link.link_id: 10.0},
        forwarding_table=table,
    )

    assert evaluation.link_utilizations[first_link.link_id] == inf
    assert evaluation.maximum_link_utilization == inf
    assert first_link.link_id in evaluation.overloaded_links


def test_evaluation_captures_ecmp_maximum_utilization(diamond_topology):
    table = compute_ospf_routes(diamond_topology)
    link_loads = {link_id: 50.0 for link_id in diamond_topology.links}

    evaluation = evaluate_netkeeper_metrics(
        diamond_topology,
        link_loads,
        forwarding_table=table,
    )

    assert evaluation.maximum_link_utilization == 0.5
    assert evaluation.average_link_utilization == 0.5


def test_environment_evaluates_netkeeper_metrics_end_to_end(diamond_topology):
    env = NetworkSimulationEnvironment()
    env.topology = diamond_topology
    env.set_traffic_matrix(TrafficMatrix((TrafficDemand("R1", "R4", 100.0),)))
    env.set_policies([ForwardPolicy("must-use-r3", "R1", "R4", "R3")])

    env.compute_ospf_routes()
    env.propagate_traffic()
    before_snapshot = env.capture_forwarding_snapshot()
    before_metrics = env.evaluate_metrics()

    env.update_ospf_weight("R1", "R3", 5)
    env.compute_ospf_routes()
    env.propagate_traffic()
    after_metrics = env.evaluate_metrics(previous_snapshot=before_snapshot)

    assert before_metrics.policy_consistency.consistency == 1.0
    assert after_metrics.policy_consistency.consistency == 0.0
    assert after_metrics.maximum_link_utilization >= 0.0
    assert after_metrics.traffic_shift is not None
    assert after_metrics.traffic_shift.shift_ratio > 0.0


def test_environment_evaluate_metrics_without_traffic_uses_zero_loads(single_path_topology):
    env = NetworkSimulationEnvironment()
    env.topology = single_path_topology
    env.compute_ospf_routes()

    result = env.evaluate_metrics()

    assert result.maximum_link_utilization == 0.0
    assert result.policy_consistency.total == 0


def test_environment_can_recompute_stored_bgp_candidates(single_path_topology):
    from netkeeper_sim.routing.bgp import BGPRoute

    env = NetworkSimulationEnvironment()
    env.topology = single_path_topology
    prefix = "203.0.113.0/24"
    candidates = {
        "R1": {
            prefix: [
                BGPRoute(prefix, "R2", 100, (65001,), 100, "R2", "peer"),
            ]
        }
    }

    first = env.compute_bgp_routes(candidates)
    env.update_ospf_weight("R1", "R2", 2)
    second = env.compute_bgp_routes()

    assert first["R1"][prefix].next_hop == "R2"
    assert second["R1"][prefix].next_hop == "R2"
