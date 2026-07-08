from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Transition:
    state: Any
    observations: Any
    action: dict[str, dict[str, list[int]]]
    action_indices: dict[str, dict[str, Any]]
    previous_action: dict[str, dict[str, list[int]]] | None
    previous_action_indices: dict[str, dict[str, Any]] | None
    rewards: dict[str, float]
    next_state: Any
    next_observations: Any
    terminated: bool
    truncated: bool
    action_masks: dict[str, Any]
    next_action_masks: dict[str, Any]
    topology_identifier: str

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class ReplayBuffer:
    def __init__(self, capacity: int = 100, seed: int | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items: deque[Transition] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def add(self, transition: Transition) -> None:
        self._items.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._items):
            raise ValueError("batch_size exceeds replay buffer size")
        return self._rng.sample(list(self._items), batch_size)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
