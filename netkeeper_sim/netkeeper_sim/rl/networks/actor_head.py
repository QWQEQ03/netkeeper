from __future__ import annotations

import torch
from torch import nn


NEGATIVE_LOGIT = -1.0e9


class FactorizedParameterHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_count: int = 64,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        hidden = hidden_dim or input_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_count),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def apply_entity_mask(
    logits: torch.Tensor,
    entity_mask: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if logits.size(0) == 0:
        return logits
    mask = torch.as_tensor(entity_mask, device=logits.device, dtype=torch.bool).reshape(-1)
    if mask.numel() != logits.size(0):
        raise ValueError(f"{name} mask length does not match logits rows")
    if not torch.any(mask):
        raise ValueError(f"{name} has no valid action entity")
    masked = logits.clone()
    masked[~mask] = NEGATIVE_LOGIT
    return masked


def masked_action_probabilities(
    logits: torch.Tensor,
    entity_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.size(0) == 0:
        return logits
    mask = torch.as_tensor(entity_mask, device=logits.device, dtype=torch.bool).reshape(-1)
    probabilities = torch.softmax(logits, dim=-1)
    probabilities = probabilities * mask.unsqueeze(-1)
    return probabilities


def sample_masked_actions(
    logits: torch.Tensor,
    entity_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.size(0) == 0:
        return torch.empty((0,), dtype=torch.long, device=logits.device)
    mask = torch.as_tensor(entity_mask, device=logits.device, dtype=torch.bool).reshape(-1)
    actions = torch.full((logits.size(0),), -1, dtype=torch.long, device=logits.device)
    valid_indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
    if valid_indices.numel() == 0:
        raise ValueError("Cannot sample because no valid action entity exists")
    distribution = torch.distributions.Categorical(logits=logits[valid_indices])
    actions[valid_indices] = distribution.sample()
    return actions


def argmax_masked_actions(
    logits: torch.Tensor,
    entity_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.size(0) == 0:
        return torch.empty((0,), dtype=torch.long, device=logits.device)
    mask = torch.as_tensor(entity_mask, device=logits.device, dtype=torch.bool).reshape(-1)
    actions = torch.full((logits.size(0),), -1, dtype=torch.long, device=logits.device)
    valid_indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
    if valid_indices.numel() == 0:
        raise ValueError("Cannot argmax because no valid action entity exists")
    actions[valid_indices] = torch.argmax(logits[valid_indices], dim=-1)
    return actions
