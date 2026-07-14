"""Deterministic release metadata, statistics, and full dataset validation."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping

import yaml

from netkeeper_sim.dataset.dynamic_sequences import validate_dynamic_sequences
from netkeeper_sim.dataset.scenarios import DIFFICULTY_COUNTS, ScenarioDataset, scenario_from_record, validate_scenarios
from netkeeper_sim.dataset.traffic import LOAD_LEVELS, PATTERNS
from netkeeper_sim.schemas import JointAction, Topology
from netkeeper_sim.schemas.ids import SCHEMA_VERSION
from netkeeper_sim.simulator import UnifiedNetworkEnvironment
from netkeeper_sim.simulator.deterministic import simulate_deterministic


PUBLICATION_VERSION = "netkeeper-lite.release.v1"


def generate_release_metadata(dataset_root: str | Path, *, root_seed: int = 20260713) -> dict[str, Any]:
    root = Path(dataset_root)
    topology_split = _load(root / "metadata" / "topology_split.json")
    traffic_manifest = _load(root / "metadata" / "traffic_manifest.json")
    scenario_manifest = _load(root / "metadata" / "scenario_manifest.json")
    dynamic_manifest = _load(root / "metadata" / "dynamic_sequences_manifest.json")
    config = {
        "dataset_version": PUBLICATION_VERSION,
        "schema_version": SCHEMA_VERSION,
        "generator_versions": {"topology": topology_split["generator_version"], "traffic": traffic_manifest["generator_version"], "scenarios": scenario_manifest["generator_version"], "dynamic": dynamic_manifest["generator_version"]},
        "root_seed": root_seed,
        "seed_derivation": {"algorithm": "SHA-256", "formula": "int.from_bytes(sha256(f'{root_seed}:{namespace}:{part1}:...').digest()[:8], 'big')", "python_hash_used": False, "namespaces": ["topology", "scenario", "traffic", "policy", "event"]},
        "topology": {"selection_seed": topology_split["selection_seed"], "selection_policy": topology_split["selection_policy"], "defaults": topology_split["defaults"], "counts": {split: len(records) for split, records in topology_split["splits"].items()}},
        "traffic": {"patterns": {"gravity": "node capacity mass product with OD jitter", "diurnal": "gravity plus stored phase/time-index sinusoid", "hotspot": "deterministic hotspot ingress/egress multiplier", "burst": "deterministic OD burst multiplier"}, "parameters": traffic_manifest["generation_config"], "quotas": {"train": {"gravity": 900, "diurnal": 600, "hotspot": 750, "burst": 750}, "validation": {"gravity": 120, "diurnal": 80, "hotspot": 100, "burst": 100}, "test": {"gravity": 150, "diurnal": 100, "hotspot": 125, "burst": 125}}, "load_levels": dict(LOAD_LEVELS)},
        "policies": {"difficulty_counts_per_kind": DIFFICULTY_COUNTS, "difficulty_ratios": scenario_manifest["difficulty_ratios"], "static_filter": "0 < initial consistency < 1; no invalid/conflict/infeasible policy"},
        "scenarios": {"counts": scenario_manifest["generated_scenario_counts"], "max_steps": scenario_manifest["max_steps"], "traffic_ratios": scenario_manifest["traffic_ratios"], "load_ratios": scenario_manifest["load_ratios"]},
        "dynamic": {"count": dynamic_manifest["count"], "logical_event_plan": dynamic_manifest["logical_event_plan"], "recovery_budget_steps": dynamic_manifest["recovery_budget_steps"], "expected_valid": dynamic_manifest["expected_valid"]},
    }
    _write_yaml(root / "metadata" / "generation_config.yaml", config)
    random_seeds = {"schema_version": SCHEMA_VERSION, "root_seed": root_seed, "algorithm": config["seed_derivation"], "recorded_seed_fields": {"topology_split": "metadata/topology_split.json:selection_seed", "traffic": "metadata/traffic_manifest.json:root_seed and matrices[].seed", "scenarios": "scenarios/*.jsonl:seed and derived_seeds", "dynamic": "dynamic_sequences/test.jsonl:seed and derived_seeds"}}
    _write_json(root / "metadata" / "random_seeds.json", random_seeds)
    statistics = dataset_statistics(root, topology_split, traffic_manifest, scenario_manifest, dynamic_manifest)
    _write_json(root / "metadata" / "dataset_statistics.json", statistics)
    manifest = create_manifest(root)
    _write_json(root / "metadata" / "manifest.json", manifest)
    return {"generation_config": config, "statistics": statistics, "manifest": manifest}


def dataset_statistics(root: Path, topology_split: Mapping[str, Any], traffic_manifest: Mapping[str, Any], scenario_manifest: Mapping[str, Any], dynamic_manifest: Mapping[str, Any]) -> dict[str, Any]:
    topology_stats = {}
    for split, records in topology_split["splits"].items():
        nodes, edges, densities = [item["node_count"] for item in records], [item["edge_count"] for item in records], [item["density"] for item in records]
        topology_stats[split] = {"count": len(records), "nodes": _summary(nodes), "edges": _summary(edges), "density": _summary(densities), "multigraph_count": sum(item["is_multigraph"] for item in records)}
    scenario_rows = {split: list(_jsonl(root / info["file"])) for split, info in scenario_manifest["splits"].items()}
    traffic_counts: dict[str, Counter[str]] = {}; load_counts: dict[str, Counter[str]] = {}; difficulty_counts: dict[str, Counter[str]] = {}; policy_counts: dict[str, Counter[str]] = {}
    demand_counts: dict[str, list[int]] = defaultdict(list); totals: dict[str, list[float]] = defaultdict(list); consistencies: dict[str, list[float]] = defaultdict(list); matrix_mlu: dict[str, float] = {}; matrix_values: dict[str, tuple[int, float]] = {}
    for split, rows in scenario_rows.items():
        traffic_counts[split] = Counter(row["traffic"]["pattern"] for row in rows); load_counts[split] = Counter(row["traffic"]["load_level"] for row in rows); difficulty_counts[split] = Counter(row["difficulty"] for row in rows); policy_counts[split] = Counter(policy["kind"] for row in rows for policy in row["policies"])
        for row in rows:
            traffic = row["traffic"]; matrix_id = traffic["matrix_id"]
            if matrix_id not in matrix_mlu:
                scenario = scenario_from_record(root, row)
                matrix_mlu[matrix_id] = simulate_deterministic(scenario.topology, scenario.configuration, scenario.traffic).metrics.maximum_link_utilization
                matrix_values[matrix_id] = (len(scenario.traffic.demands), sum(item.traffic_rate_bps for item in scenario.traffic.demands) * scenario.traffic.load_multiplier)
            demands, total = matrix_values[matrix_id]
            demand_counts[split].append(demands); totals[split].append(total)
            consistencies[split].append(float(row["initial_policy"]["initial_policy_consistency"]))
    dynamic_rows = list(_jsonl(root / dynamic_manifest["file"]))
    dynamic_events = Counter(event["kind"] for row in dynamic_rows for event in row["events"])
    return {"schema_version": SCHEMA_VERSION, "topologies": topology_stats, "scenarios": {split: {"count": len(rows), "traffic_patterns": _count_ratio(traffic_counts[split]), "load_levels": _count_ratio(load_counts[split]), "policy_difficulties": _count_ratio(difficulty_counts[split]), "policy_kinds": dict(policy_counts[split]), "traffic_demand_count": _summary(demand_counts[split]), "total_input_bps": _summary(totals[split]), "initial_mlu": _summary([matrix_mlu[row["traffic"]["matrix_id"]] for row in rows]), "initial_policy_consistency": _summary(consistencies[split]), "policy_resample_attempts": _summary([row["initial_policy"]["attempt"] for row in rows])} for split, rows in scenario_rows.items()}, "dynamic": {"sequence_count": len(dynamic_rows), "event_type_counts": dict(dynamic_events), "logical_event_counts": dict(Counter(logical["kind"] for row in dynamic_rows for logical in row["logical_events"])), "expected_valid_count": sum(row["expected_valid"] for row in dynamic_rows)}, "rejections": {"policy_sampling_exhausted": 0, "dynamic_partition_or_invalid": 0}}


def create_manifest(root: Path) -> dict[str, Any]:
    files = []
    excluded = {"metadata/manifest.json", "metadata/scenario_generation_config.json", "metadata/dynamic_smoke_manifest.json"}
    for path in sorted((item for item in root.rglob("*") if item.is_file() and item.relative_to(root).as_posix() not in excluded and not item.relative_to(root).as_posix().startswith(("scenarios/smoke/", "dynamic_sequences/smoke/"))), key=lambda item: item.relative_to(root).as_posix()):
        files.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path.read_bytes()), "schema_version": SCHEMA_VERSION})
    return {"dataset_version": PUBLICATION_VERSION, "schema_version": SCHEMA_VERSION, "self_hash_rule": "metadata/manifest.json is excluded from files to avoid recursive hashing", "file_count": len(files), "files": files}


def validate_release(dataset_root: str | Path, *, sample_environment: bool = True) -> dict[str, Any]:
    root = Path(dataset_root); errors: list[dict[str, Any]] = []
    topology = _load(root / "metadata" / "topology_split.json"); scenarios = _load(root / "metadata" / "scenario_manifest.json"); dynamic = _load(root / "metadata" / "dynamic_sequences_manifest.json"); manifest = _load(root / "metadata" / "manifest.json")
    expected_topology = {"train": 12, "validation": 3, "test": 4}; expected_scenarios = {"train": 3000, "validation": 400, "test": 500}
    if {split: len(rows) for split, rows in topology["splits"].items()} != expected_topology: errors.append({"code": "topology_count_mismatch"})
    static = validate_scenarios(root)
    if not static["valid"] or static["scenario_counts"] != expected_scenarios: errors.append({"code": "static_scenario_validation", "detail": static})
    dynamic_result = validate_dynamic_sequences(root)
    if not dynamic_result["valid"] or dynamic_result["count"] != 100: errors.append({"code": "dynamic_validation", "detail": dynamic_result})
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"] or _sha256(path.read_bytes()) != entry["sha256"]: errors.append({"code": "manifest_hash_mismatch", "path": entry["path"]})
    sampled = 0
    if sample_environment:
        needed: dict[str, set[str]] = {"topology": set(), "pattern": set(PATTERNS), "load": set(LOAD_LEVELS), "difficulty": set(DIFFICULTY_COUNTS)}
        selected = []
        for split, info in scenarios["splits"].items():
            for row in _jsonl(root / info["file"]):
                keys = {"topology": row["topology_id"], "pattern": row["traffic"]["pattern"], "load": row["traffic"]["load_level"], "difficulty": row["difficulty"]}
                if keys["topology"] not in needed["topology"] or keys["pattern"] in needed["pattern"] or keys["load"] in needed["load"] or keys["difficulty"] in needed["difficulty"]:
                    selected.append(row); needed["topology"].add(keys["topology"]); needed["pattern"].discard(keys["pattern"]); needed["load"].discard(keys["load"]); needed["difficulty"].discard(keys["difficulty"])
                if len(needed["topology"]) == 19 and not needed["pattern"] and not needed["load"] and not needed["difficulty"]: break
        for row in selected:
            scenario = scenario_from_record(root, row); environment = UnifiedNetworkEnvironment(); snapshot, _ = environment.reset(scenario, seed=int(row["seed"])); result = environment.step(snapshot, JointAction((), snapshot_id=snapshot.snapshot_id)); sampled += 1
            if result.errors: errors.append({"code": "sample_environment_error", "scenario_id": row["scenario_id"]})
    return {"valid": not errors, "static": {"scenario_counts": static["scenario_counts"], "unique_scenario_ids": static["unique_scenario_ids"]}, "dynamic": dynamic_result, "sampled_environment_scenarios": sampled, "errors": errors}


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    items = list(values)
    return {"count": len(items), "min": min(items) if items else None, "max": max(items) if items else None, "mean": fmean(items) if items else None}


def _count_ratio(counter: Counter[str]) -> dict[str, dict[str, float | int]]:
    total = sum(counter.values())
    return {key: {"count": value, "ratio": value / total if total else 0.0} for key, value in sorted(counter.items())}


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip(): yield json.loads(line)


def _load(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding="utf-8"))
def _sha256(value: bytes) -> str: return hashlib.sha256(value).hexdigest()
def _write_json(path: Path, value: Any) -> None: path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
def _write_yaml(path: Path, value: Any) -> None: path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=True), encoding="utf-8")
