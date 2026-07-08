from __future__ import annotations

import torch


def legal_action_probabilities(
    actor_logits: torch.Tensor,
    entity_mask: torch.Tensor,
) -> torch.Tensor:
    if actor_logits.numel() == 0:
        return actor_logits
    mask = torch.as_tensor(entity_mask, device=actor_logits.device, dtype=torch.bool).reshape(-1)
    if mask.numel() != actor_logits.size(0):
        raise ValueError("entity mask shape does not match actor logits")
    probabilities = torch.softmax(actor_logits, dim=-1)
    return probabilities * mask.unsqueeze(-1)


def coma_counterfactual_baseline(
    actor_logits: torch.Tensor,
    critic_q_values: torch.Tensor,
    selected_actions: torch.Tensor,
    entity_mask: torch.Tensor,
    detach_advantage: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if actor_logits.shape != critic_q_values.shape:
        raise ValueError("actor logits and critic q-values must have the same shape")
    if actor_logits.numel() == 0:
        empty = actor_logits.new_zeros((0,))
        return empty, empty, empty
    mask = torch.as_tensor(entity_mask, device=actor_logits.device, dtype=torch.bool).reshape(-1)
    selected = torch.as_tensor(selected_actions, device=actor_logits.device, dtype=torch.long).reshape(-1)
    if selected.numel() != actor_logits.size(0):
        raise ValueError("selected action count must match entity count")
    if torch.any((selected[mask] < 0) | (selected[mask] >= actor_logits.size(-1))):
        raise ValueError("selected actions for valid entities must be in range")

    probabilities = legal_action_probabilities(actor_logits, mask)
    baseline = (probabilities * critic_q_values).sum(dim=-1)
    selected_q = critic_q_values.gather(1, selected.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    advantage = (selected_q - baseline) * mask.float()
    if detach_advantage:
        advantage = advantage.detach()
    return baseline, selected_q, advantage


def selected_log_probabilities(
    actor_logits: torch.Tensor,
    selected_actions: torch.Tensor,
) -> torch.Tensor:
    if actor_logits.numel() == 0:
        return actor_logits.new_zeros((0,))
    selected = torch.as_tensor(selected_actions, device=actor_logits.device, dtype=torch.long).reshape(-1)
    if selected.numel() != actor_logits.size(0):
        raise ValueError("selected action count must match entity count")
    valid_selected = selected >= 0
    if torch.any(selected[valid_selected] >= actor_logits.size(-1)):
        raise ValueError("selected actions must be in range")
    log_probs = torch.log_softmax(actor_logits, dim=-1)
    return log_probs.gather(1, selected.clamp_min(0).unsqueeze(-1)).squeeze(-1)
