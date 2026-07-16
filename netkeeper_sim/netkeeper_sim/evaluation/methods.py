"""Stable method protocol; implementations can only return schema actions."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Mapping, Protocol

from netkeeper_sim.api.models import OptimizationRequest
from netkeeper_sim.schemas import AgentObservation, JointAction, NetworkScenario, NetworkSnapshot


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode()).hexdigest()


@dataclass(frozen=True)
class MethodMetadata:
    name: str
    version: str
    config_hash: str
    checkpoint_hash: str | None
    deterministic: bool
    allowed_parameter_types: tuple[str, ...]
    uses_lookahead: bool
    checkpoint_status: str = "none"

    @property
    def identity(self) -> str:
        return canonical_hash(self.__dict__)


@dataclass(frozen=True)
class EvaluationContext:
    run_id: str
    scenario: NetworkScenario
    scenario_id: str
    seed: int
    initial_snapshot: NetworkSnapshot
    max_steps: int
    sequence_id: str | None = None
    recovery_budget_steps: int | None = None
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MethodDecision:
    action: JointAction
    decision_time_ns: int = 0
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    status: str = "ok"
    failure_code: str | None = None


class EvaluationMethod(Protocol):
    metadata: MethodMetadata
    def reset(self, context: EvaluationContext) -> None: ...
    def act(self, snapshot: NetworkSnapshot, observation: AgentObservation, context: EvaluationContext) -> MethodDecision | JointAction: ...


class UnavailableMethod:
    """A stable non-random failure path for absent/incompatible checkpoints."""
    def __init__(self, metadata: MethodMetadata, code: str, detail: str) -> None:
        self.metadata, self.code, self.detail = metadata, code, detail
    def reset(self, context: EvaluationContext) -> None: pass
    def act(self, snapshot, observation, context) -> MethodDecision:
        return MethodDecision(JointAction((), requested_by=self.metadata.name, snapshot_id=snapshot.snapshot_id), status="unavailable", failure_code=self.code, diagnostics={"detail": self.detail})


class DispatcherMethodAdapter:
    """Greedy Block-5 dispatcher adapter.  It never substitutes random weights."""
    def __init__(self, checkpoint: str | Path, *, config: Mapping[str, Any] | None = None, checkpoint_status: str = "debug_unconverged") -> None:
        path = Path(checkpoint)
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        self._load_error: tuple[str, str] | None = None
        try:
            if not path.is_file():
                raise FileNotFoundError(str(path))
            from netkeeper_sim.rl.dispatcher import TrainedPolicyDispatcher
            self.dispatcher = TrainedPolicyDispatcher(str(path), device=(config or {}).get("device", "cpu"))
        except FileNotFoundError as exc:
            self.dispatcher = None; self._load_error = ("checkpoint_unavailable", str(exc))
        except Exception as exc:  # dispatcher has the authoritative compatibility check
            self.dispatcher = None; self._load_error = ("checkpoint_incompatible", str(exc))
        revised=getattr(self.dispatcher,"training_semantics_version",None)=="coma-counterfactual-v4"
        inference_config={"inference":"greedy_eval_no_grad","objectives":["policy_consistency","mlu"],"action_adapter":"block5-candidate-v2" if revised else "block5-candidate-v1"}
        self.metadata = MethodMetadata("coma_dispatcher", "rl-coma-v3-adapter.1" if revised else "legacy-inference-adapter", canonical_hash(inference_config), digest, True, ("ospf_weight", "local_preference", "as_path_length", "med", "bandwidth_bps", "capacity_bps", "queue_packets"), False, checkpoint_status)
    def reset(self, context: EvaluationContext) -> None: pass
    def act(self, snapshot, observation, context) -> MethodDecision:
        started = perf_counter_ns()
        if self._load_error:
            code, detail = self._load_error
            return MethodDecision(JointAction((), requested_by="coma_dispatcher", snapshot_id=snapshot.snapshot_id), perf_counter_ns()-started, {"detail": detail}, "unavailable", code)
        try:
            action = self.dispatcher.dispatch(snapshot, OptimizationRequest(snapshot.snapshot_id, ("policy_consistency", "mlu")))
            return MethodDecision(action, perf_counter_ns()-started)
        except Exception as exc:
            return MethodDecision(JointAction((), requested_by="coma_dispatcher", snapshot_id=snapshot.snapshot_id), perf_counter_ns()-started, {"detail": str(exc)}, "failed", "dispatcher_error")
