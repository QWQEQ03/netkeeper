"""Unified-schema RL environment façade.

`MultiAgentNetworkEnvironment` no longer owns a mutable simulator or a second
StepResult type.  Its state is always a `NetworkSnapshot`; tensor state is a
derived `SnapshotGraph` for the current snapshot only.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from netkeeper_sim.dataset.scenarios import scenario_from_record
from netkeeper_sim.schemas import JointAction, NetworkScenario, StepResult
from netkeeper_sim.simulator import UnifiedNetworkEnvironment
from netkeeper_sim.simulator import RewardConfig
from netkeeper_sim.rl.schema_adapter import SnapshotGraph, action_masks, candidate_to_joint_action, snapshot_to_graph


@dataclass(frozen=True)
class RLStep:
    result: StepResult
    graph: SnapshotGraph
    masks: dict


class BalancedScenarioSampler:
    """Deterministic, balanced train sampler; validation is a fixed shuffle."""
    def __init__(self, dataset_root: str | Path, split: str, seed: int = 42) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("RL sampling permits only train or validation splits")
        self.root, self.split, self.seed = Path(dataset_root), split, seed
        rows = [json.loads(line) for line in (self.root / "scenarios" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line]
        if any(row.get("split") != split for row in rows): raise ValueError("scenario split metadata mismatch")
        self.rows = tuple(rows)
        self._by_key: dict[tuple[str, str, str, str], list[dict]] = {}
        for row in self.rows:
            self._by_key.setdefault((row["topology_id"], row["traffic"]["pattern"], row["traffic"]["load_level"], row["difficulty"]), []).append(row)
        self._keys = tuple(sorted(self._by_key))
        self._by_topology: dict[str, list[dict]] = {}
        for row in self.rows: self._by_topology.setdefault(row["topology_id"], []).append(row)
        self._topologies = tuple(sorted(self._by_topology))
        self._rng = random.Random(seed)
        self._positions = {key: 0 for key in self._keys}
        for values in self._by_key.values(): self._rng.shuffle(values)
        for values in self._by_topology.values(): self._rng.shuffle(values)
        self._topology_positions = {key: 0 for key in self._topologies}
        self._validation_order = list(self.rows); random.Random(seed).shuffle(self._validation_order)
        self._validation_position = 0

    def next_record(self) -> dict:
        if self.split == "validation":
            row = self._validation_order[self._validation_position % len(self._validation_order)]; self._validation_position += 1; return row
        key = self._topologies[sum(self._topology_positions.values()) % len(self._topologies)]
        values = self._by_topology[key]; position = self._topology_positions[key]; self._topology_positions[key] += 1
        return values[position % len(values)]

    def next_scenario(self) -> NetworkScenario:
        # The JSONL metadata is indexed once; the heavyweight topology,
        # configuration and traffic objects remain lazy until this selection.
        return scenario_from_record(self.root, self.next_record())

    def state_dict(self) -> dict:
        return {"positions": dict(self._positions), "topology_positions":dict(self._topology_positions), "rng": self._rng.getstate(), "validation_position": self._validation_position}

    def load_state_dict(self, state: dict) -> None:
        if set(state["positions"]) != set(self._positions): raise ValueError("sampler topology bucket mismatch")
        self._positions = dict(state["positions"]); self._topology_positions=dict(state["topology_positions"]); self._rng.setstate(state["rng"]); self._validation_position = int(state["validation_position"])


class MultiAgentNetworkEnvironment:
    """Schema-only environment used by future training code.

    Pass either an explicit `scenario` or `dataset_root` plus `split`.  Legacy
    topology/traffic constructor arguments were removed; callers should use
    `NetworkScenario` and are warned to migrate rather than receiving a second
    mutable environment.
    """
    def __init__(self, *, scenario: NetworkScenario | None = None, dataset_root: str | Path | None = None, split: str = "train", seed: int = 42, reward_config: RewardConfig | None = None, **legacy: object) -> None:
        if legacy:
            raise TypeError("legacy topology/traffic arguments are removed; construct a NetworkScenario")
        if scenario is None and dataset_root is None: raise ValueError("scenario or dataset_root is required")
        if scenario is not None and dataset_root is not None: raise ValueError("choose explicit scenario or dataset sampler")
        self._scenario = scenario; self.sampler = None if scenario else BalancedScenarioSampler(dataset_root, split, seed)
        self.environment = UnifiedNetworkEnvironment(reward_config); self.seed = seed; self.snapshot = None; self.observation = None; self.graph = None

    def reset(self, *, seed: int | None = None, scenario: NetworkScenario | None = None) -> tuple[SnapshotGraph, object, dict]:
        if seed is not None: self.seed = seed
        selected = scenario or self._scenario or self.sampler.next_scenario()  # type: ignore[union-attr]
        self.snapshot, self.observation = self.environment.reset(selected, seed=self.seed)
        self.graph = snapshot_to_graph(self.snapshot)
        return self.graph, self.observation, action_masks(self.snapshot, self.graph)

    def step(self, action: JointAction | Mapping[str, int]) -> RLStep:
        if self.snapshot is None or self.graph is None: raise RuntimeError("reset must be called before step")
        joint = action if isinstance(action, JointAction) else candidate_to_joint_action(self.snapshot, self.graph, action)
        result = self.environment.step(self.snapshot, joint)
        self.snapshot, self.observation = result.next_snapshot, result.observations
        self.graph = snapshot_to_graph(self.snapshot)
        return RLStep(result, self.graph, action_masks(self.snapshot, self.graph))


# New name used by training code.  The historical class name remains only as a
# compatibility import; it no longer wraps NetworkSimulationEnvironment.
UnifiedRLEnvironment = MultiAgentNetworkEnvironment
