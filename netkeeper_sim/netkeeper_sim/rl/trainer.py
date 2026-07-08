from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from netkeeper_sim.rl.algorithms.losses import actor_loss, critic_loss, td_target
from netkeeper_sim.rl.algorithms.target_update import hard_update, soft_update
from netkeeper_sim.rl.config import COMATrainingConfig
from netkeeper_sim.rl.multi_agent_env import MultiAgentNetworkEnvironment
from netkeeper_sim.rl.networks import MultiAgentActor
from netkeeper_sim.rl.networks.centralized_critic import CentralizedCritic
from netkeeper_sim.rl.networks.target_network import clone_target_network
from netkeeper_sim.rl.replay_buffer import ReplayBuffer, Transition


PARAMETER_SPECS = {
    "ospf": (("ospf_weight", 1),),
    "bgp": (("local_preference", 1), ("as_path_length", 1), ("med", 1)),
    "performance": (("bandwidth", 65), ("capacity", 65), ("queue_length", 65)),
}


@dataclass(frozen=True)
class TrainingLogEntry:
    episode: int
    step: int
    epsilon: float
    policy_consistency: float
    maximum_link_utilization: float
    traffic_shift: float
    ospf_reward: float
    bgp_reward: float
    performance_reward: float
    actor_loss: float | None
    critic_loss: float | None
    gradient_norm: float | None
    terminated: bool
    truncated: bool


class EpsilonGreedyExplorer:
    def __init__(
        self,
        start: float = 1.0,
        end: float = 0.01,
        decay: float = 0.99,
        seed: int | None = None,
    ) -> None:
        self.epsilon = start
        self.end = end
        self.decay = decay
        self.rng = random.Random(seed)

    def decay_once(self) -> float:
        self.epsilon = max(self.end, self.epsilon * self.decay)
        return self.epsilon

    def select(
        self,
        logits: torch.Tensor,
        entity_mask: Any,
        deterministic: bool = False,
    ) -> torch.Tensor:
        mask = torch.as_tensor(entity_mask, device=logits.device, dtype=torch.bool).reshape(-1)
        if logits.size(0) == 0:
            return torch.empty((0,), dtype=torch.long, device=logits.device)
        actions = torch.full((logits.size(0),), -1, dtype=torch.long, device=logits.device)
        valid_indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
        if valid_indices.numel() == 0:
            return actions
        if deterministic:
            actions[valid_indices] = torch.argmax(logits[valid_indices], dim=-1)
            return actions
        for index in valid_indices.tolist():
            if self.rng.random() < self.epsilon:
                actions[index] = self.rng.randrange(logits.size(-1))
            else:
                distribution = torch.distributions.Categorical(logits=logits[index])
                actions[index] = distribution.sample()
        return actions


class COMATrainer:
    def __init__(
        self,
        env: MultiAgentNetworkEnvironment,
        training_config: COMATrainingConfig | None = None,
    ) -> None:
        self.env = env
        self.config = training_config or COMATrainingConfig.debug()
        self.device = self._resolve_device()
        _seed_everything(self.config.deterministic_seed)
        state, _observations = self.env.reset(seed=self.config.deterministic_seed)
        graph_config = self.config.graph_config(state.node_features.shape[1], state.edge_features.shape[1])
        self.actor = MultiAgentActor(graph_config).to(self.device)
        self.critic = CentralizedCritic(
            graph_config,
            hidden_layers=self.config.decoder_linear_layers,
        ).to(self.device)
        self.target_critic = clone_target_network(self.critic).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
        self.buffer = ReplayBuffer(self.config.replay_buffer_size, seed=self.config.deterministic_seed)
        self.explorer = EpsilonGreedyExplorer(
            self.config.epsilon_start,
            self.config.epsilon_end,
            self.config.epsilon_decay,
            seed=self.config.deterministic_seed,
        )
        self.global_step = 0

    def train(self) -> list[TrainingLogEntry]:
        logs: list[TrainingLogEntry] = []
        for episode in range(self.config.episodes):
            state, observations = self.env.reset(seed=self.config.deterministic_seed + episode)
            previous_action: dict[str, dict[str, list[int]]] | None = None
            previous_action_indices: dict[str, dict[str, Any]] | None = None
            for step in range(self.config.max_steps):
                action, action_indices = self.select_action(state, deterministic=False)
                result = self.env.step(action)
                transition = Transition(
                    state=state,
                    observations=observations,
                    action=action,
                    action_indices=action_indices,
                    previous_action=previous_action,
                    previous_action_indices=previous_action_indices,
                    rewards=result.rewards,
                    next_state=result.state,
                    next_observations=result.observations,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    action_masks=state.parameter_masks,
                    next_action_masks=result.state.parameter_masks,
                    topology_identifier="episode-topology",
                )
                self.buffer.add(transition)
                actor_loss_value: float | None = None
                critic_loss_value: float | None = None
                gradient_norm: float | None = None
                if len(self.buffer) >= self.config.batch_size:
                    batch = self.buffer.sample(self.config.batch_size)
                    actor_loss_tensor, critic_loss_tensor, gradient_norm = self.update(batch)
                    actor_loss_value = float(actor_loss_tensor.detach().cpu())
                    critic_loss_value = float(critic_loss_tensor.detach().cpu())
                if self.global_step % self.config.target_update_interval == 0:
                    hard_update(self.target_critic, self.critic)
                if self.config.target_update_tau is not None:
                    soft_update(self.target_critic, self.critic, self.config.target_update_tau)
                evaluation = result.info["evaluation"]
                logs.append(
                    TrainingLogEntry(
                        episode=episode,
                        step=step,
                        epsilon=self.explorer.epsilon,
                        policy_consistency=evaluation.policy_consistency.consistency,
                        maximum_link_utilization=evaluation.maximum_link_utilization,
                        traffic_shift=0.0
                        if evaluation.traffic_shift is None
                        else evaluation.traffic_shift.shift_ratio,
                        ospf_reward=result.rewards["ospf"],
                        bgp_reward=result.rewards["bgp"],
                        performance_reward=result.rewards["performance"],
                        actor_loss=actor_loss_value,
                        critic_loss=critic_loss_value,
                        gradient_norm=gradient_norm,
                        terminated=result.terminated,
                        truncated=result.truncated,
                    )
                )
                self.global_step += 1
                self.explorer.decay_once()
                previous_action = action
                previous_action_indices = action_indices
                state = result.state
                observations = result.observations
                if result.terminated or result.truncated:
                    break
        return logs

    def select_action(
        self,
        state: Any,
        deterministic: bool = False,
    ) -> tuple[dict[str, dict[str, list[int]]], dict[str, dict[str, torch.Tensor]]]:
        with torch.no_grad():
            output = self.actor.forward_state(state)
        logits_by_agent = {
            "ospf": output.ospf.logits,
            "bgp": output.bgp.logits,
            "performance": output.performance.logits,
        }
        action: dict[str, dict[str, list[int]]] = {"ospf": {}, "bgp": {}, "performance": {}}
        action_indices: dict[str, dict[str, torch.Tensor]] = {"ospf": {}, "bgp": {}, "performance": {}}
        for agent, specs in PARAMETER_SPECS.items():
            for parameter, minimum in specs:
                logits = logits_by_agent[agent][parameter]
                mask = state.parameter_masks[f"{parameter}_mask"]
                selected = self.explorer.select(logits, mask, deterministic=deterministic)
                action_indices[agent][parameter] = selected.detach().cpu()
                values = []
                for index in selected.detach().cpu().tolist():
                    values.append(0 if index < 0 else int(index) + minimum)
                action[agent][parameter] = values
        return action, action_indices

    def update(self, batch: list[Transition]) -> tuple[torch.Tensor, torch.Tensor, float]:
        self.actor.train()
        self.critic.train()
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_losses: list[torch.Tensor] = []
        critic_losses: list[torch.Tensor] = []
        for transition in batch:
            actor_output = self.actor.forward_state(transition.state)
            logits_by_agent = {
                "ospf": actor_output.ospf.logits,
                "bgp": actor_output.bgp.logits,
                "performance": actor_output.performance.logits,
            }
            for agent, specs in PARAMETER_SPECS.items():
                for parameter, _minimum in specs:
                    selected = _action_tensor(transition.action_indices[agent][parameter], self.device)
                    current = selected
                    previous = (
                        None
                        if transition.previous_action_indices is None
                        else _action_tensor(transition.previous_action_indices[agent][parameter], self.device)
                    )
                    mask = _mask_tensor(transition.state.parameter_masks[f"{parameter}_mask"], self.device)
                    next_mask = _mask_tensor(
                        transition.next_state.parameter_masks[f"{parameter}_mask"],
                        self.device,
                    )
                    if mask.numel() == 0:
                        continue
                    q_values = self.critic.forward_state(
                        transition.state,
                        agent,
                        parameter,
                        current_actions=current,
                        previous_actions=previous,
                    ).logits[parameter]
                    with torch.no_grad():
                        next_q = self.target_critic.forward_state(
                            transition.next_state,
                            agent,
                            parameter,
                            current_actions=None,
                            previous_actions=current,
                        ).logits[parameter]
                        reward = torch.full(
                            (next_q.size(0),),
                            float(transition.rewards[agent]),
                            dtype=torch.float32,
                            device=self.device,
                        )
                        done = torch.full(
                            (next_q.size(0),),
                            float(transition.done),
                            dtype=torch.float32,
                            device=self.device,
                        )
                        target = td_target(reward, done, next_q, next_mask, self.config.gamma)
                    critic_losses.append(
                        critic_loss(q_values, selected, target, mask, loss=self.config.loss)
                    )
                    if logits_by_agent[agent][parameter].numel() > 0:
                        actor_losses.append(
                            actor_loss(
                                logits_by_agent[agent][parameter],
                                q_values.detach(),
                                selected,
                                mask,
                            )
                        )
        if not actor_losses or not critic_losses:
            zero = next(self.actor.parameters()).new_zeros(())
            return zero, zero, 0.0
        actor_loss_total = torch.stack(actor_losses).mean()
        critic_loss_total = torch.stack(critic_losses).mean()
        if not torch.isfinite(actor_loss_total) or not torch.isfinite(critic_loss_total):
            raise FloatingPointError("NaN or Inf detected in training loss")
        critic_loss_total.backward()
        actor_loss_total.backward()
        critic_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradient_clip_norm)
        actor_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradient_clip_norm)
        self.critic_optimizer.step()
        self.actor_optimizer.step()
        gradient_norm = float(max(float(critic_norm), float(actor_norm)))
        return actor_loss_total, critic_loss_total, gradient_norm

    def evaluate_deterministic(self) -> tuple[dict[str, dict[str, list[int]]], Any]:
        state = self.env.current_state
        if state is None:
            state, _ = self.env.reset(seed=self.config.deterministic_seed)
        action, _indices = self.select_action(state, deterministic=True)
        return action, self.actor.forward_state(state)

    def _resolve_device(self) -> torch.device:
        if self.config.device == "cpu":
            return torch.device("cpu")
        if self.config.device == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _action_tensor(value: Any, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.long, device=device).reshape(-1)


def _mask_tensor(value: Any, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.bool, device=device).reshape(-1)
