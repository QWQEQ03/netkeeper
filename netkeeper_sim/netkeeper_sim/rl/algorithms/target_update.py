from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())
    for parameter in target.parameters():
        parameter.requires_grad_(False)


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be between 0 and 1")
    for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
        target_parameter.data.mul_(1.0 - tau).add_(source_parameter.data, alpha=tau)
        target_parameter.requires_grad_(False)
