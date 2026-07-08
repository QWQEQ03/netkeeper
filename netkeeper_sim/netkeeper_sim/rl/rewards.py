from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from netkeeper_sim.metrics.evaluation import EvaluationResult
from netkeeper_sim.rl import _tensor as T
from netkeeper_sim.rl.config import RLConfig


@dataclass(frozen=True)
class RewardResult:
    rewards: dict[str, float]
    policy_reward: float
    resource_reward: float
    stationary_reward: float
    dynamic_reward: float
    normalized_load: float
    traffic_shift: float


def compute_multi_agent_rewards(
    previous_policy_observation: Any,
    current_policy_observation: Any,
    evaluation: EvaluationResult,
    previous_action: tuple[Any, ...] | None,
    current_action: tuple[Any, ...],
    config: RLConfig,
) -> RewardResult:
    stationary_reward = -1.0 if previous_action == current_action else 0.0
    dynamic_reward = _dynamic_reward(
        T.to_numpy(previous_policy_observation).reshape(-1).tolist(),
        T.to_numpy(current_policy_observation).reshape(-1).tolist(),
    )
    normalized_load = _normalized_load(evaluation.maximum_link_utilization, config)
    traffic_shift = 0.0 if evaluation.traffic_shift is None else evaluation.traffic_shift.shift_ratio
    if config.reward_mode == "normalized":
        traffic_shift = max(0.0, min(1.0, traffic_shift))

    policy_reward = (
        config.reward_k * evaluation.policy_consistency.consistency
        + stationary_reward
        + dynamic_reward
    )
    resource_reward = config.reward_k * ((1.0 - normalized_load) + (1.0 - traffic_shift))
    return RewardResult(
        rewards={
            "ospf": policy_reward + resource_reward,
            "bgp": policy_reward,
            "performance": resource_reward,
        },
        policy_reward=policy_reward,
        resource_reward=resource_reward,
        stationary_reward=stationary_reward,
        dynamic_reward=dynamic_reward,
        normalized_load=normalized_load,
        traffic_shift=traffic_shift,
    )


def _dynamic_reward(previous: list[float], current: list[float]) -> float:
    reward = 0.0
    for old, new in zip(previous, current):
        if old < 0.5 <= new:
            reward += 1.0
        elif old >= 0.5 > new:
            reward -= 1.0
    return reward


def _normalized_load(maximum_link_utilization: float, config: RLConfig) -> float:
    value = float(maximum_link_utilization)
    if config.reward_mode == "paper":
        return value
    denominator = config.maximum_link_utilization_normalizer
    if denominator <= 0:
        denominator = 1.0
    return max(0.0, min(1.0, value / denominator))
