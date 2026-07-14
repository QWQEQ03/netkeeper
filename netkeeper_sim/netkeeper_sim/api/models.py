"""Immutable wire models for the versioned NetKeeper operation API."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


def _freeze(value: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


@dataclass(frozen=True)
class ApiCall:
    api: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", _freeze(self.arguments))

    def to_dict(self) -> dict[str, Any]: return {"api": self.api, "arguments": _plain(self.arguments)}
    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ApiCall": return cls(str(value.get("api", "")), value.get("arguments", {}))


@dataclass(frozen=True)
class ApiRequest:
    api_version: str
    request_id: str
    calls: tuple[ApiCall, ...]
    expected_snapshot_id: str | None = None
    need_optimization: bool = False
    dry_run: bool = False

    def __post_init__(self) -> None: object.__setattr__(self, "calls", tuple(self.calls))
    def to_dict(self) -> dict[str, Any]:
        value = {"api_version": self.api_version, "request_id": self.request_id,
                 "calls": [call.to_dict() for call in self.calls], "need_optimization": self.need_optimization, "dry_run": self.dry_run}
        if self.expected_snapshot_id is not None: value["expected_snapshot_id"] = self.expected_snapshot_id
        return value
    def to_json(self) -> str: return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, allow_nan=False)
    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ApiRequest":
        return cls(str(value.get("api_version", "")), str(value.get("request_id", "")), tuple(ApiCall.from_dict(v) for v in value.get("calls", ())),
                   value.get("expected_snapshot_id"), value.get("need_optimization", False), value.get("dry_run", False))
    @classmethod
    def from_json(cls, value: str) -> "ApiRequest": return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class ApiError:
    error_type: str
    code: str
    message: str
    call_index: int | None = None
    api: str | None = None
    details: Mapping[str, Any] = field(default_factory=_freeze)

    def __post_init__(self) -> None: object.__setattr__(self, "details", _freeze(self.details))
    def to_dict(self) -> dict[str, Any]:
        return {"error_type": self.error_type, "code": self.code, "message": self.message, "call_index": self.call_index, "api": self.api, "details": _plain(self.details)}
    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ApiError": return cls(**dict(value))


@dataclass(frozen=True)
class ExecutionPlan:
    base_snapshot_id: str
    ordered_calls: tuple[int, ...]
    events: tuple[Mapping[str, Any], ...] = ()
    actions: tuple[Mapping[str, Any], ...] = ()
    read_calls: tuple[int, ...] = ()
    optimization_requested: bool = False
    predicted_diff: Mapping[str, Any] = field(default_factory=_freeze)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ordered_calls", tuple(self.ordered_calls)); object.__setattr__(self, "events", tuple(_freeze(v) for v in self.events))
        object.__setattr__(self, "actions", tuple(_freeze(v) for v in self.actions)); object.__setattr__(self, "read_calls", tuple(self.read_calls)); object.__setattr__(self, "predicted_diff", _freeze(self.predicted_diff))
    def to_dict(self) -> dict[str, Any]: return _plain(self.__dict__)
    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionPlan": return cls(**dict(value))


@dataclass(frozen=True)
class OptimizationRequest:
    snapshot_id: str
    objectives: tuple[str, ...]
    max_steps: int | None = None
    request_id: str = ""

    def __post_init__(self) -> None: object.__setattr__(self, "objectives", tuple(self.objectives))
    def to_dict(self) -> dict[str, Any]: return _plain(self.__dict__)


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: tuple[ApiError, ...] = ()
    plan: ExecutionPlan | None = None

    def __post_init__(self) -> None: object.__setattr__(self, "errors", tuple(self.errors))
    def to_dict(self) -> dict[str, Any]: return {"valid": self.valid, "errors": [v.to_dict() for v in self.errors], "plan": self.plan.to_dict() if self.plan else None}


@dataclass(frozen=True)
class ApiResponse:
    success: bool
    request_id: str
    applied_calls: tuple[int, ...] = ()
    before_snapshot_id: str | None = None
    after_snapshot_id: str | None = None
    configuration_diff: Mapping[str, Any] = field(default_factory=_freeze)
    metrics: Mapping[str, Any] = field(default_factory=_freeze)
    optimization_status: str = "not_requested"
    event_diff: Mapping[str, Any] = field(default_factory=_freeze)
    state: Mapping[str, Any] = field(default_factory=_freeze)
    errors: tuple[ApiError, ...] = ()
    plan: ExecutionPlan | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "applied_calls", tuple(self.applied_calls)); object.__setattr__(self, "configuration_diff", _freeze(self.configuration_diff)); object.__setattr__(self, "metrics", _freeze(self.metrics)); object.__setattr__(self, "event_diff", _freeze(self.event_diff)); object.__setattr__(self, "state", _freeze(self.state)); object.__setattr__(self, "errors", tuple(self.errors))
    def to_dict(self) -> dict[str, Any]:
        return {"success": self.success, "request_id": self.request_id, "applied_calls": list(self.applied_calls), "before_snapshot_id": self.before_snapshot_id, "after_snapshot_id": self.after_snapshot_id, "configuration_diff": _plain(self.configuration_diff), "metrics": _plain(self.metrics), "optimization_status": self.optimization_status, "event_diff": _plain(self.event_diff), "state": _plain(self.state), "errors": [v.to_dict() for v in self.errors], "plan": self.plan.to_dict() if self.plan else None}
    def to_json(self) -> str: return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, allow_nan=False)
    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ApiResponse":
        raw = dict(value); raw["errors"] = tuple(ApiError.from_dict(v) for v in raw.get("errors", ())); raw["plan"] = ExecutionPlan.from_dict(raw["plan"]) if raw.get("plan") else None; return cls(**raw)
    @classmethod
    def from_json(cls, value: str) -> "ApiResponse": return cls.from_dict(json.loads(value))
