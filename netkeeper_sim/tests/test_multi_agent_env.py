from __future__ import annotations

from netkeeper_sim.policies import ForwardPolicy
from netkeeper_sim.rl import MultiAgentNetworkEnvironment, RLConfig
from netkeeper_sim.routing.bgp import BGPRoute
from netkeeper_sim.traffic.matrix import TrafficDemand, TrafficMatrix


def test_invalid_masked_action_does_not_change_environment(diamond_topology):
    env = MultiAgentNetworkEnvironment(topology=diamond_topology)
    state, _ = env.reset()
    first_link_id = state.link_ids[0]
    first_link = env.env.topology.links[first_link_id]
    env.env.fail_link(first_link.source, first_link.target)
    original_weight = env.env.topology.links[first_link_id].ospf_weight
    action = _zero_action(len(state.link_ids), len(state.bgp_route_targets))
    action["ospf"]["ospf_weight"][0] = 9

    env.step(action)

    assert env.env.topology.links[first_link_id].ospf_weight == original_weight


def test_ospf_action_modifies_ospf_weight(diamond_topology):
    env = MultiAgentNetworkEnvironment(topology=diamond_topology)
    state, _ = env.reset()
    action = _zero_action(len(state.link_ids), len(state.bgp_route_targets))
    action["ospf"]["ospf_weight"][0] = 7

    env.step(action)

    assert env.env.topology.links[state.link_ids[0]].ospf_weight == 7.0


def test_bgp_action_modifies_candidate_route_parameters(single_path_topology):
    prefix = "203.0.113.0/24"
    candidates = {
        "R1": {
            prefix: [BGPRoute(prefix, "R2", 10, (65001,), 10, "R2", "peer")]
        }
    }
    env = MultiAgentNetworkEnvironment(
        topology=single_path_topology,
        bgp_candidate_routes=candidates,
    )
    state, _ = env.reset()
    action = _zero_action(len(state.link_ids), len(state.bgp_route_targets))
    action["bgp"]["local_preference"][0] = 12
    action["bgp"]["as_path_length"][0] = 3
    action["bgp"]["med"][0] = 14

    env.step(action)

    route = env.env.bgp_candidate_routes["R1"][prefix][0]
    assert route.local_preference == 12
    assert len(route.as_path) == 3
    assert route.med == 14


def test_performance_action_modifies_link_attributes(diamond_topology):
    env = MultiAgentNetworkEnvironment(topology=diamond_topology)
    state, _ = env.reset()
    action = _zero_action(len(state.link_ids), len(state.bgp_route_targets))
    action["performance"]["bandwidth"][0] = 66
    action["performance"]["capacity"][0] = 67
    action["performance"]["queue_length"][0] = 68

    env.step(action)

    link = env.env.topology.links[state.link_ids[0]]
    assert link.bandwidth == 66.0
    assert link.capacity == 67.0
    assert link.queue_length == 68


def test_step_recomputes_metrics_and_traffic_shift(diamond_topology):
    env = MultiAgentNetworkEnvironment(
        topology=diamond_topology,
        traffic_matrix=TrafficMatrix((TrafficDemand("R1", "R4", 100.0),)),
        policies=[ForwardPolicy("must-use-r3", "R1", "R4", "R3")],
    )
    state, _ = env.reset()
    action = _zero_action(len(state.link_ids), len(state.bgp_route_targets))
    action["ospf"]["ospf_weight"][_link_index(state.link_ids, diamond_topology, "R1", "R3")] = 10

    result = env.step(action)

    assert result.info["evaluation"].maximum_link_utilization >= 0.0
    assert result.info["evaluation"].traffic_shift is not None
    assert result.info["evaluation"].policy_consistency.consistency == 0.0


def test_policy_consistency_one_sets_terminated(diamond_topology):
    env = MultiAgentNetworkEnvironment(
        topology=diamond_topology,
        policies=[ForwardPolicy("p1", "R1", "R4", "R2")],
    )
    state, _ = env.reset()

    result = env.step(_zero_action(len(state.link_ids), len(state.bgp_route_targets)))

    assert result.terminated is True


def test_max_steps_sets_truncated(diamond_topology):
    env = MultiAgentNetworkEnvironment(
        topology=diamond_topology,
        config=RLConfig(max_steps=1),
        policies=[ForwardPolicy("unsatisfied", "R1", "R4", "missing")],
    )
    state, _ = env.reset()

    result = env.step(_zero_action(len(state.link_ids), len(state.bgp_route_targets)))

    assert result.truncated is True
    assert result.terminated is False


def test_reset_restores_initial_topology(diamond_topology):
    env = MultiAgentNetworkEnvironment(topology=diamond_topology)
    state, _ = env.reset()
    action = _zero_action(len(state.link_ids), len(state.bgp_route_targets))
    action["ospf"]["ospf_weight"][0] = 7
    env.step(action)

    reset_state, _ = env.reset()

    assert env.env.topology.links[reset_state.link_ids[0]].ospf_weight == 1.0


def test_seed_makes_random_actions_reproducible(diamond_topology):
    env = MultiAgentNetworkEnvironment(topology=diamond_topology)
    env.reset(seed=11)
    first = env.sample_random_action()

    env.reset(seed=11)
    second = env.sample_random_action()

    assert first == second


def _zero_action(link_count: int, route_count: int) -> dict[str, dict[str, list[int]]]:
    return {
        "ospf": {"ospf_weight": [0] * link_count},
        "bgp": {
            "local_preference": [0] * route_count,
            "as_path_length": [0] * route_count,
            "med": [0] * route_count,
        },
        "performance": {
            "bandwidth": [0] * link_count,
            "capacity": [0] * link_count,
            "queue_length": [0] * link_count,
        },
    }


def _link_index(link_ids, topology, source: str, target: str) -> int:
    for index, link_id in enumerate(link_ids):
        link = topology.links[link_id]
        if {link.source, link.target} == {source, target}:
            return index
    raise AssertionError(f"missing link {source}-{target}")
