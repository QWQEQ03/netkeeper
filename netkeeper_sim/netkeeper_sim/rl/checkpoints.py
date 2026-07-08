from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    target_critic: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer | None = None,
    critic_optimizer: torch.optim.Optimizer | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    checkpoint = {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "target_critic": target_critic.state_dict(),
        "metadata": metadata or {},
    }
    if actor_optimizer is not None:
        checkpoint["actor_optimizer"] = actor_optimizer.state_dict()
    if critic_optimizer is not None:
        checkpoint["critic_optimizer"] = critic_optimizer.state_dict()
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, target_path)


def load_checkpoint(
    path: str | Path,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    target_critic: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer | None = None,
    critic_optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    actor.load_state_dict(checkpoint["actor"])
    critic.load_state_dict(checkpoint["critic"])
    target_critic.load_state_dict(checkpoint["target_critic"])
    if actor_optimizer is not None and "actor_optimizer" in checkpoint:
        actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
    if critic_optimizer is not None and "critic_optimizer" in checkpoint:
        critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
    return dict(checkpoint.get("metadata", {}))
