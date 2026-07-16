"""Result schema, field-level configuration accounting and atomic JSONL store."""
from __future__ import annotations

import csv
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from netkeeper_sim.evaluation.methods import canonical_hash
from netkeeper_sim.schemas import NetworkConfiguration

EVALUATOR_VERSION = "netkeeper-evaluation-v3"

def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if math.isfinite(value): return value
        return {"non_finite": "nan" if math.isnan(value) else ("positive_infinity" if value > 0 else "negative_infinity")}
    if hasattr(value, "to_dict"): return json_safe(value.to_dict())
    if isinstance(value, Mapping): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    return value

def configuration_fields(config: NetworkConfiguration) -> dict[str, Any]:
    """The action-addressable scalar universe, keyed independently for parallel links/routes."""
    fields: dict[str, Any] = {}
    for link_id in sorted(config.ospf_weights): fields[f"ospf_weight:{link_id}"] = config.ospf_weights[link_id]
    for link_id, attr in sorted(config.performance.items()):
        for name in ("bandwidth_bps", "capacity_bps", "queue_packets", "loss_rate"):
            fields[f"performance:{link_id}:{name}"] = getattr(attr, name)
    for link_id in sorted(config.link_states): fields[f"link_state:{link_id}"] = config.link_states[link_id]
    for node_id in sorted(config.node_states): fields[f"node_state:{node_id}"] = config.node_states[node_id]
    for route in config.bgp.routes:
        key = f"{route.router_id}|{route.prefix}|{route.next_hop}"
        fields[f"bgp:{key}:local_preference"] = route.local_preference
        fields[f"bgp:{key}:as_path_length"] = len(route.as_path)
        fields[f"bgp:{key}:med"] = route.med
    return fields

def configuration_change(initial: NetworkConfiguration, final: NetworkConfiguration) -> dict[str, Any]:
    before, after = configuration_fields(initial), configuration_fields(final)
    universe = sorted(set(before) | set(after))
    changed = [key for key in universe if before.get(key) != after.get(key)]
    return {"count": len(changed), "denominator": len(universe), "ratio": len(changed) / len(universe) if universe else 0.0, "changed_fields": changed}

def run_key(method: Mapping[str, Any], scenario_id: str, sequence_id: str | None, seed: int, evaluator_config_hash: str) -> str:
    return canonical_hash({"method": dict(method), "scenario_id": scenario_id, "sequence_id": sequence_id, "seed": seed, "evaluator_config_hash": evaluator_config_hash})

class ResultStore:
    """Portable JSONL persistence; append is atomically replaced and idempotent by key."""
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
    @contextmanager
    def _locked(self):
        """Serialize read/modify/replace operations across evaluator processes."""
        lock_path = self.root / ".results.lock"
        with lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    def append_unique(self, relative: str, record: Mapping[str, Any], *, key: str) -> bool:
        with self._locked():
            path = self.root / relative; path.parent.mkdir(parents=True, exist_ok=True)
            rows = self.read(relative)
            if any(row.get("run_key") == key for row in rows): return False
            rows.append(json_safe(dict(record)))
            self._atomic_jsonl(path, rows); return True
    def commit_run(
        self,
        *,
        run_key: str,
        summary: Mapping[str, Any],
        steps: Iterable[Mapping[str, Any]],
        events: Iterable[Mapping[str, Any]],
        terminal_file: str,
    ) -> bool:
        """Commit one complete run with the terminal record written last.

        A killed process may leave pre-terminal step/event rows.  The next
        resume replaces those rows for the run instead of mixing trajectories.
        The process lock also prevents two resume workers from losing updates.
        """
        if terminal_file not in {"episodes.jsonl", "failures.jsonl"}:
            raise ValueError(f"invalid terminal file: {terminal_file}")
        step_rows = [json_safe(dict(row)) for row in steps]
        event_rows = [json_safe(dict(row)) for row in events]
        terminal = json_safe(dict(summary))
        with self._locked():
            existing_terminals = self.read("episodes.jsonl") + self.read("failures.jsonl")
            if any(row.get("run_key") == run_key for row in existing_terminals):
                return False

            saved_steps = [
                row for row in self.read("steps.jsonl")
                if str(row.get("run_key", "")).rsplit(":", 1)[0] != run_key
            ]
            saved_events = [
                row for row in self.read("event_recovery.jsonl")
                if row.get("run_key") != run_key
            ]
            self._atomic_jsonl(self.root / "steps.jsonl", saved_steps + step_rows)
            self._atomic_jsonl(self.root / "event_recovery.jsonl", saved_events + event_rows)

            terminal_path = self.root / terminal_file
            terminal_rows = self.read(terminal_file)
            terminal_rows.append(terminal)
            # Terminal is deliberately last: its presence certifies that all
            # trajectory rows for this run were durably replaced first.
            self._atomic_jsonl(terminal_path, terminal_rows)
            return True
    def read(self, relative: str) -> list[dict[str, Any]]:
        path = self.root / relative
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []
    def write_json(self, relative: str, value: Mapping[str, Any]) -> None:
        path = self.root / relative; path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_bytes(path, (json.dumps(json_safe(value), sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)+"\n").encode())
    def write_csv(self, relative: str, rows: Iterable[Mapping[str, Any]]) -> None:
        rows = list(rows); path = self.root / relative; path.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({key for row in rows for key in row})
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=path.parent) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows([{k: json.dumps(json_safe(row.get(k))) if isinstance(row.get(k), (dict, list)) else row.get(k) for k in fields} for row in rows]); temporary = Path(handle.name)
        os.replace(temporary, path)
    def _atomic_jsonl(self, path: Path, rows: list[Mapping[str, Any]]) -> None:
        self._atomic_bytes(path, "".join(json.dumps(json_safe(row), sort_keys=True, ensure_ascii=False, allow_nan=False)+"\n" for row in rows).encode())
    @staticmethod
    def _atomic_bytes(path: Path, value: bytes) -> None:
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=path.parent) as handle:
            handle.write(value); handle.flush(); os.fsync(handle.fileno()); temporary = Path(handle.name)
        os.replace(temporary, path)
