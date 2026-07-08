from __future__ import annotations

import copy
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from netkeeper_sim.metrics.evaluation import EvaluationResult
from netkeeper_sim.policies.model import Policy
from netkeeper_sim.rl import _tensor as T
from netkeeper_sim.rl.action_space import (
    ActionMasks,
    action_values,
    build_action_masks,
    canonical_joint_action,
)
from netkeeper_sim.rl.config import ParameterRange, RLConfig
from netkeeper_sim.rl.graph_state import NetworkGraphState, build_network_graph_state_with_policies
from netkeeper_sim.rl.observations import (
    AgentObservations,
    build_agent_observations,
    link_utilization_vector,
    policy_satisfaction_vector,
)
from netkeeper_sim.rl.rewards import RewardResult, compute_multi_agent_rewards
from netkeeper_sim.routing.bgp import BGPRoute
from netkeeper_sim.simulator.environment import NetworkSimulationEnvironment
from netkeeper_sim.topology.loader import load_topology
from netkeeper_sim.topology.model import Topology
from netkeeper_sim.traffic.matrix import TrafficMatrix


@dataclass(frozen=True)
class StepResult:
    state: NetworkGraphState
    observations: AgentObservations
    rewards: dict[str, float]
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class MultiAgentNetworkEnvironment:
    """Multi-agent RL adapter around the deterministic simulation environment."""

    def __init__(
        self,
        topology: Topology | str | Path | None = None,
        traffic_matrix: TrafficMatrix | None = None,
        policies: tuple[Policy, ...] | list[Policy] = (),
        bgp_candidate_routes: dict[str, dict[str, list[BGPRoute]]] | None = None,
        config: RLConfig | None = None,
    ) -> None:
        self.config = config or RLConfig()
        self._initial_topology = self._load_or_copy_topology(topology) if topology is not None else None
        self._initial_bgp_candidate_routes = copy.deepcopy(bgp_candidate_routes)
        self.traffic_matrix = traffic_matrix
        self.policies = tuple(policies)
        self.rng = random.Random()
        self.env = NetworkSimulationEnvironment()
        self.step_count = 0
        self.current_state: NetworkGraphState | None = None
        self.current_observations: AgentObservations | None = None
        self.current_evaluation: EvaluationResult | None = None
        self.current_action_masks: ActionMasks | None = None
        self._previous_action: tuple[Any, ...] | None = None
        self._last_policy_observation: T.Tensor | None = None

    def set_traffic_matrix(self, matrix: TrafficMatrix) -> None:
        self.traffic_matrix = matrix
        self.env.set_traffic_matrix(matrix)

    def set_policies(self, policies: tuple[Policy, ...] | list[Policy]) -> None:
        self.policies = tuple(policies)
        self.env.set_policies(self.policies)

    def set_bgp_candidate_routes(
        self,
        candidate_routes: dict[str, dict[str, list[BGPRoute]]] | None,
    ) -> None:
        self._initial_bgp_candidate_routes = copy.deepcopy(candidate_routes)
        self.env.bgp_candidate_routes = copy.deepcopy(candidate_routes)

    def reset(
        self,
        topology: Topology | str | Path | None = None,
        seed: int | None = None,
    ) -> tuple[NetworkGraphState, AgentObservations]:
        if seed is not None:
            self.rng.seed(seed)
        if topology is not None:
            self._initial_topology = self._load_or_copy_topology(topology)
        if self._initial_topology is None:
            raise RuntimeError("reset requires a topology on the first call")

        self.env = NetworkSimulationEnvironment()
        self.env.topology = copy.deepcopy(self._initial_topology)
        if self.traffic_matrix is not None:
            self.env.set_traffic_matrix(self.traffic_matrix)
        self.env.set_policies(self.policies)
        self.env.bgp_candidate_routes = copy.deepcopy(self._initial_bgp_candidate_routes)
        self.step_count = 0
        self._previous_action = None

        evaluation = self._recompute()
        state, observations, masks = self._build_state_and_observations(evaluation)
        self.current_state = state
        self.current_observations = observations
        self.current_evaluation = evaluation
        self.current_action_masks = masks
        self._last_policy_observation = observations.policy_satisfaction
        return state, observations

    def step(self, joint_action: Mapping[str, Mapping[str, Any]]) -> StepResult:
        if self.env.topology is None:
            raise RuntimeError("reset must be called before step")
        previous_snapshot = self.env.capture_forwarding_snapshot()
        previous_policy_observation = (
            self._last_policy_observation
            if self._last_policy_observation is not None
            else T.tensor([], "float32")
        )
        masks = build_action_masks(self.env.topology, self.env.bgp_candidate_routes)
        canonical_action = canonical_joint_action(joint_action)

        self._apply_joint_action(joint_action, masks)
        evaluation = self._recompute(previous_snapshot=previous_snapshot)
        state, observations, next_masks = self._build_state_and_observations(evaluation)
        reward_result = compute_multi_agent_rewards(
            previous_policy_observation,
            observations.policy_satisfaction,
            evaluation,
            self._previous_action,
            canonical_action,
            self.config,
        )

        self.step_count += 1
        terminated = evaluation.policy_consistency.consistency == 1.0
        truncated = self.step_count >= self.config.max_steps
        self.current_state = state
        self.current_observations = observations
        self.current_evaluation = evaluation
        self.current_action_masks = next_masks
        self._previous_action = canonical_action
        self._last_policy_observation = observations.policy_satisfaction

        return StepResult(
            state=state,
            observations=observations,
            rewards=reward_result.rewards,
            terminated=terminated,
            truncated=truncated,
            info={
                "evaluation": evaluation,
                "reward": reward_result,
                "action_masks": next_masks,
                "step_count": self.step_count,
            },
        )

    def sample_random_action(self) -> dict[str, dict[str, list[int]]]:
        if self.env.topology is None:
            raise RuntimeError("reset must be called before sampling actions")
        masks = build_action_masks(self.env.topology, self.env.bgp_candidate_routes)
        return {
            "ospf": {
                "ospf_weight": self._sample_values(masks.ospf_weight_mask, self.config.ospf_weight),
            },
            "bgp": {
                "local_preference": self._sample_values(
                    masks.local_preference_mask,
                    self.config.local_preference,
                ),
                "as_path_length": self._sample_values(
                    masks.as_path_length_mask,
                    self.config.as_path_length,
                ),
                "med": self._sample_values(masks.med_mask, self.config.med),
            },
            "performance": {
                "bandwidth": self._sample_values(masks.bandwidth_mask, self.config.bandwidth),
                "capacity": self._sample_values(masks.capacity_mask, self.config.capacity),
                "queue_length": self._sample_values(masks.queue_length_mask, self.config.queue_length),
            },
        }

    def _recompute(self, previous_snapshot: object | None = None) -> EvaluationResult:
        self.env.compute_ospf_routes()
        if self.env.bgp_candidate_routes is not None:
            self.env.compute_bgp_routes()
        if self.env.traffic_matrix is not None:
            self.env.propagate_traffic()
        return self.env.evaluate_metrics(previous_snapshot=previous_snapshot)

    def _build_state_and_observations(
        self,
        evaluation: EvaluationResult,
    ) -> tuple[NetworkGraphState, AgentObservations, ActionMasks]:
        topology = self._require_topology()
        masks = build_action_masks(topology, self.env.bgp_candidate_routes)
        policy_observation = policy_satisfaction_vector(evaluation.policy_consistency)
        utilization_observation = link_utilization_vector(evaluation, masks.link_ids)
        observations = build_agent_observations(policy_observation, utilization_observation)
        state = build_network_graph_state_with_policies(
            topology,
            self.config,
            self.policies,
            policy_observation,
            utilization_observation,
            candidate_routes=self.env.bgp_candidate_routes,
            link_utilizations=evaluation.link_utilizations,
            action_masks=masks,
        )
        return state, observations, masks

    def _apply_joint_action(
        self,
        joint_action: Mapping[str, Mapping[str, Any]],
        masks: ActionMasks,
    ) -> None:
        self._apply_ospf_actions(joint_action.get("ospf", {}), masks)
        self._apply_bgp_actions(joint_action.get("bgp", {}), masks)
        self._apply_performance_actions(joint_action.get("performance", {}), masks)
        self.env.forwarding_table = None
        self.env.bgp_routes = None
        self.env.propagation_result = None
        self.env.link_metrics = None

    def _apply_ospf_actions(
        self,
        action: Mapping[str, Any],
        masks: ActionMasks,
    ) -> None:
        topology = self._require_topology()
        values = action_values(action.get("ospf_weight"), len(masks.link_ids))
        valid = T.to_numpy(masks.ospf_weight_mask).astype(bool)
        for index, link_id in enumerate(masks.link_ids):
            value = int(values[index])
            if not valid[index] or not self.config.ospf_weight.contains(value):
                continue
            link = topology.links[link_id]
            self.env.update_ospf_weight(link.source, link.target, float(value))

    def _apply_bgp_actions(
        self,
        action: Mapping[str, Any],
        masks: ActionMasks,
    ) -> None:
        if self.env.bgp_candidate_routes is None:
            return
        routes = self.env.bgp_candidate_routes
        self._apply_bgp_parameter(
            routes,
            masks,
            action.get("local_preference"),
            "local_preference",
            self.config.local_preference,
        )
        self._apply_bgp_parameter(
            routes,
            masks,
            action.get("as_path_length"),
            "as_path_length",
            self.config.as_path_length,
        )
        self._apply_bgp_parameter(
            routes,
            masks,
            action.get("med"),
            "med",
            self.config.med,
        )

    def _apply_bgp_parameter(
        self,
        routes: dict[str, dict[str, list[BGPRoute]]],
        masks: ActionMasks,
        raw_values: Any,
        parameter: str,
        value_range: ParameterRange,
    ) -> None:
        values = action_values(raw_values, len(masks.bgp_route_targets))
        valid = T.to_numpy(getattr(masks, f"{parameter}_mask" if parameter != "as_path_length" else "as_path_length_mask")).astype(bool)
        for index, target in enumerate(masks.bgp_route_targets):
            value = int(values[index])
            if not valid[index] or not value_range.contains(value):
                continue
            route = routes[target.router][target.prefix][target.route_index]
            if parameter == "local_preference":
                updated = replace(route, local_preference=value)
            elif parameter == "med":
                updated = replace(route, med=value)
            else:
                updated = replace(route, as_path=tuple(range(65000, 65000 + value)))
            routes[target.router][target.prefix][target.route_index] = updated

    def _apply_performance_actions(
        self,
        action: Mapping[str, Any],
        masks: ActionMasks,
    ) -> None:
        self._apply_link_parameter(
            action.get("bandwidth"),
            masks,
            "bandwidth",
            self.config.bandwidth,
            as_int=False,
        )
        self._apply_link_parameter(
            action.get("capacity"),
            masks,
            "capacity",
            self.config.capacity,
            as_int=False,
        )
        self._apply_link_parameter(
            action.get("queue_length"),
            masks,
            "queue_length",
            self.config.queue_length,
            as_int=True,
        )

    def _apply_link_parameter(
        self,
        raw_values: Any,
        masks: ActionMasks,
        parameter: str,
        value_range: ParameterRange,
        as_int: bool,
    ) -> None:
        topology = self._require_topology()
        values = action_values(raw_values, len(masks.link_ids))
        valid = T.to_numpy(getattr(masks, f"{parameter}_mask")).astype(bool)
        for index, link_id in enumerate(masks.link_ids):
            value = int(values[index])
            if not valid[index] or not value_range.contains(value):
                continue
            link = topology.links[link_id]
            new_value: int | float = value if as_int else float(value)
            setattr(link, parameter, new_value)
            topology.graph[link.source][link.target][link.link_id][parameter] = new_value

    def _sample_values(self, mask: T.Tensor, value_range: ParameterRange) -> list[int]:
        values: list[int] = []
        for is_valid in T.to_numpy(mask).astype(bool).reshape(-1):
            if not is_valid or self.rng.random() < 0.5:
                values.append(0)
            else:
                values.append(self.rng.randint(value_range.minimum, value_range.maximum))
        return values

    def _require_topology(self) -> Topology:
        if self.env.topology is None:
            raise RuntimeError("No topology has been loaded")
        return self.env.topology

    @staticmethod
    def _load_or_copy_topology(topology: Topology | str | Path | None) -> Topology:
        if topology is None:
            raise RuntimeError("topology is required")
        if isinstance(topology, Topology):
            return copy.deepcopy(topology)
        return load_topology(topology)
