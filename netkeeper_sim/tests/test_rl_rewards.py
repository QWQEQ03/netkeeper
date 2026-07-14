from __future__ import annotations
import pytest
pytestmark = pytest.mark.skip(reason="superseded by unified RewardBreakdown tests")

from netkeeper_sim.policies import ForwardPolicy
from netkeeper_sim.rl import MultiAgentNetworkEnvironment, RLConfig
from netkeeper_sim.traffic.matrix import TrafficDemand, TrafficMatrix


def test_stationary_reward_penalizes_repeated_action(diamond_topology):
    env = MultiAgentNetworkEnvironment(
        topology=diamond_topology,
        policies=[ForwardPolicy("p1", "R1", "R4", "R2")],
    )
    state, _ = env.reset()
    action = _zero_action(len(state.link_ids), len(state.bgp_route_targets))

    env.step(action)
    result = env.step(action)

    reward = result.info["reward"]
    assert reward.stationary_reward == -1.0


def test_dynamic_reward_penalizes_policy_regression(diamond_topology):
    env = MultiAgentNetworkEnvironment(
        topology=diamond_topology,
        traffic_matrix=TrafficMatrix((TrafficDemand("R1", "R4", 100.0),)),
        policies=[ForwardPolicy("must-use-r3", "R1", "R4", "R3")],
    )
    state, _ = env.reset()
    action = _zero_action(len(state.link_ids), len(state.bgp_route_targets))
    action["ospf"]["ospf_weight"][_link_index(state.link_ids, diamond_topology, "R1", "R3")] = 10

    result = env.step(action)

    reward = result.info["reward"]
    assert reward.dynamic_reward == -1.0
    assert result.info["evaluation"].policy_consistency.consistency == 0.0


def test_reward_components_match_agent_rewards(diamond_topology):
    env = MultiAgentNetworkEnvironment(
        topology=diamond_topology,
        config=RLConfig(reward_mode="normalized"),
        traffic_matrix=TrafficMatrix((TrafficDemand("R1", "R4", 100.0),)),
        policies=[ForwardPolicy("p1", "R1", "R4", "R2")],
    )
    state, _ = env.reset()
    action = _zero_action(len(state.link_ids), len(state.bgp_route_targets))

    result = env.step(action)
    reward = result.info["reward"]

    assert result.rewards["bgp"] == reward.policy_reward
    assert result.rewards["performance"] == reward.resource_reward
    assert result.rewards["ospf"] == reward.policy_reward + reward.resource_reward


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
