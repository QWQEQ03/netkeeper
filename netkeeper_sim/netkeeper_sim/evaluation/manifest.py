"""Deterministically freeze evaluation inputs without evaluating a method."""
from __future__ import annotations
import hashlib, json, platform, subprocess
from pathlib import Path
from typing import Any, Mapping
from netkeeper_sim.evaluation.methods import canonical_hash
from netkeeper_sim.schemas.ids import SCHEMA_VERSION

EVALUATION_MANIFEST_VERSION = "netkeeper-evaluation-manifest-v2"

def _rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
def _sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def _commit(root: Path) -> str | None:
    try: return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return None

def _load_group(record: Mapping[str, Any]) -> str:
    traffic=record["traffic"]; pattern=traffic["pattern"]; level=traffic["load_level"]
    if level == "Normal" and pattern in {"gravity", "diurnal"}: return "Normal"
    if level == "Low" and pattern == "hotspot": return "Hotspot"
    if level == "Low" and pattern == "burst": return "Burst"
    if level == "High": return "High-load"
    raise ValueError(f"unassigned formal load group:{record['scenario_id']}:{level}:{pattern}")

def generate_evaluation_manifest(dataset_root: str | Path, *, seeds: tuple[int, ...] = (20260714, 20260715, 20260716), deterministic_seed: int | None = None, max_steps: int | None = None, hold_steps: int = 3) -> dict[str, Any]:
    root=Path(dataset_root); scenario_meta=json.loads((root/"metadata/scenario_manifest.json").read_text(encoding="utf-8")); dynamic_meta=json.loads((root/"metadata/dynamic_sequences_manifest.json").read_text(encoding="utf-8"))
    static_rows=_rows(root/scenario_meta["splits"]["test"]["file"]); dynamic_rows=_rows(root/dynamic_meta["file"])
    deterministic=int(deterministic_seed if deterministic_seed is not None else seeds[0])
    static=[{"scenario_id":x["scenario_id"],"topology_id":x["topology_id"],"difficulty":x["difficulty"],"traffic_pattern":x["traffic"]["pattern"],"load_level":x["traffic"]["load_level"],"load_multiplier":x["traffic"]["load_multiplier"],"load_group":_load_group(x)} for x in static_rows]
    groups={name:[x["scenario_id"] for x in static if x["load_group"]==name] for name in ("Normal","Hotspot","Burst","High-load")}
    value={"manifest_version":EVALUATION_MANIFEST_VERSION,"schema_version":SCHEMA_VERSION,"dataset_manifest_sha256":_sha(root/"metadata/manifest.json"),"dataset_manifest_file":"metadata/manifest.json","source_files":{"static":{"file":scenario_meta["splits"]["test"]["file"],"sha256":_sha(root/scenario_meta["splits"]["test"]["file"])},"dynamic":{"file":dynamic_meta["file"],"sha256":_sha(root/dynamic_meta["file"])}},"static":static,"dynamic":[{"sequence_id":x["sequence_id"],"scenario_id":x["initial_scenario_id"],"topology_id":x["topology_id"],"event_types":[e["kind"] for e in x["logical_events"]],"recovery_budget_steps":x["recovery_budget_steps"],"max_steps":x["dynamic_max_steps"]} for x in dynamic_rows],"analysis_groups":{"load":{"mutually_exclusive":True,"complete":True,"rules":{"Normal":"load_level == Normal and traffic_pattern in {gravity,diurnal}","Hotspot":"load_level == Low and traffic_pattern == hotspot","Burst":"load_level == Low and traffic_pattern == burst","High-load":"load_level == High (load_multiplier == 3.0)"},"scenario_ids":groups}},"method_seeds":{"deterministic":[deterministic],"random":list(seeds)},"thresholds":{"hold_steps":hold_steps,"static_policy_consistency":1.0,"dynamic_policy_relative_to_pre_event":True,"recovery_budget_steps":30},"static_max_steps":max_steps or int(scenario_meta["max_steps"]),"generation":{"python":platform.python_version(),"git_commit":_commit(root)}}
    value["manifest_hash"] = canonical_hash(value); return value
