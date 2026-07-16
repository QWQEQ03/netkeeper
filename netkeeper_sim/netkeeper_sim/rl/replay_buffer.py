from __future__ import annotations
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

@dataclass(frozen=True)
class Transition:
    graph: Any
    masks: dict[str, Any]
    actions: dict[str, int]
    reward: float | Mapping[str, float]
    next_graph: Any
    next_masks: dict[str, Any]
    terminated: bool
    truncated: bool
    snapshot_id: str
    next_snapshot_id: str
    advantages: Mapping[str, float] | None = None
    actor_weight: float = 1.0

class ReplayBuffer:
    def __init__(self, capacity: int = 1000, seed: int | None = None) -> None:
        self._items = deque(maxlen=capacity); self._rng = random.Random(seed)
    def add(self, item: Transition) -> None:
        if item.graph.snapshot_id != item.snapshot_id or item.next_graph.snapshot_id != item.next_snapshot_id: raise ValueError("transition snapshot mismatch")
        self._items.append(item)
    def sample(self, batch_size: int) -> list[Transition]: return self._rng.sample(list(self._items), batch_size)
    def __len__(self) -> int: return len(self._items)
    def state_dict(self) -> dict: return {"items": list(self._items), "rng": self._rng.getstate()}
    def load_state_dict(self, state: dict) -> None: self._items.extend(state["items"]); self._rng.setstate(state["rng"])
