from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from netkeeper_sim.rl.algorithms.coma import (
    coma_counterfactual_baseline,
    selected_log_probabilities,
)


def actor_loss(
    actor_logits: torch.Tensor,
    critic_q_values: torch.Tensor,
    selected_actions: torch.Tensor,
    entity_mask: torch.Tensor,
) -> torch.Tensor:
    if actor_logits.numel() == 0:
        return actor_logits.new_zeros(())
    mask = torch.as_tensor(entity_mask, device=actor_logits.device, dtype=torch.bool).reshape(-1)
    if not torch.any(mask):
        return actor_logits.new_zeros(())
    _baseline, _selected_q, advantage = coma_counterfactual_baseline(
        actor_logits,
        critic_q_values,
        selected_actions,
        mask,
        detach_advantage=True,
    )
    log_probs = selected_log_probabilities(actor_logits, selected_actions)
    return -((log_probs * advantage) * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def td_target(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_q_values: torch.Tensor,
    next_entity_mask: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    rewards = rewards.to(device=next_q_values.device, dtype=torch.float32).reshape(-1)
    dones = dones.to(device=next_q_values.device, dtype=torch.float32).reshape(-1)
    mask = torch.as_tensor(next_entity_mask, device=next_q_values.device, dtype=torch.bool).reshape(-1)
    if next_q_values.numel() == 0:
        return rewards
    if mask.numel() != next_q_values.size(0):
        raise ValueError("next entity mask shape does not match next q-values")
    bootstrap = torch.where(
        mask,
        next_q_values.max(dim=-1).values,
        torch.zeros(next_q_values.size(0), dtype=torch.float32, device=next_q_values.device),
    )
    return rewards + gamma * (1.0 - dones) * bootstrap


def critic_loss(
    critic_q_values: torch.Tensor,
    selected_actions: torch.Tensor,
    target: torch.Tensor,
    entity_mask: torch.Tensor,
    loss: Literal["huber", "mse"] = "huber",
) -> torch.Tensor:
    if critic_q_values.numel() == 0:
        return critic_q_values.new_zeros(())
    mask = torch.as_tensor(entity_mask, device=critic_q_values.device, dtype=torch.bool).reshape(-1)
    if not torch.any(mask):
        return critic_q_values.new_zeros(())
    selected = torch.as_tensor(selected_actions, device=critic_q_values.device, dtype=torch.long).reshape(-1)
    if selected.numel() != critic_q_values.size(0):
        raise ValueError("selected action count must match entity count")
    if torch.any((selected[mask] < 0) | (selected[mask] >= critic_q_values.size(-1))):
        raise ValueError("selected actions for valid entities must be in range")
    prediction = critic_q_values.gather(1, selected.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    target = target.to(device=critic_q_values.device, dtype=torch.float32).reshape(-1)
    if loss == "huber":
        per_entity = F.smooth_l1_loss(prediction, target, reduction="none")
    elif loss == "mse":
        per_entity = F.mse_loss(prediction, target, reduction="none")
    else:
        raise ValueError("loss must be 'huber' or 'mse'")
    return (per_entity * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
