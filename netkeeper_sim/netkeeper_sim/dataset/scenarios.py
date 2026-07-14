"""Static, lazy-loadable MADRL smoke scenarios using only unified schemas."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import networkx as nx
import numpy as np

from netkeeper_sim.dataset.traffic import derive_seed, load_traffic_record
from netkeeper_sim.policies.schema_evaluator import evaluate_schema_policies
from netkeeper_sim.schemas import NetworkConfiguration, NetworkScenario, Policy, Topology, TrafficMatrix
from netkeeper_sim.schemas.ids import SCHEMA_VERSION
from netkeeper_sim.simulator.deterministic import simulate_deterministic


SCENARIO_GENERATOR_VERSION = "1.0.0"
SCENARIO_DATASET_VERSION = "netkeeper-lite.static-scenarios.v1"
DIFFICULTY_COUNTS = {"Easy": 2, "Medium": 4, "Hard": 8}
TARGET_SCENARIO_COUNTS = {"train": 3000, "validation": 400, "test": 500}
SMOKE_SCENARIO_COUNTS = {"train": 12, "validation": 12, "test": 12}
DIFFICULTY_RATIOS = {"train": {"Easy": 0.45, "Medium": 0.45, "Hard": 0.10}, "validation": {"Easy": 0.40, "Medium": 0.40, "Hard": 0.20}, "test": {"Easy": 0.35, "Medium": 0.40, "Hard": 0.25}}
TRAFFIC_RATIOS = {"gravity": 0.30, "diurnal": 0.20, "hotspot": 0.25, "burst": 0.25}
LOAD_RATIOS = {"Low": 1 / 3, "Normal": 1 / 3, "High": 1 / 3}


@dataclass(frozen=True)
class ScenarioGenerationError(Exception):
    code: str
    message: str
    details: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": dict(self.details)}


def largest_remainder_quota(total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    exact = {key: total * value for key, value in ratios.items()}
    result = {key: int(value) for key, value in exact.items()}
    for key, _ in sorted(exact.items(), key=lambda item: (-(item[1] - int(item[1])), item[0]))[:total - sum(result.values())]:
        result[key] += 1
    return result


def sample_policies(topology: Topology, configuration: NetworkConfiguration, difficulty: str, *, seed: int, max_attempts: int = 128, routing_result: Any | None = None) -> tuple[tuple[Policy, ...], dict[str, Any]]:
    if difficulty not in DIFFICULTY_COUNTS:
        raise ValueError(f"unknown difficulty: {difficulty}")
    count = DIFFICULTY_COUNTS[difficulty]
    graph = nx.Graph()
    graph.add_nodes_from(node.node_id for node in topology.nodes)
    graph.add_edges_from((link.source, link.target) for link in topology.links)
    if not nx.is_connected(graph):
        raise ScenarioGenerationError("topology_not_connected", "policy sampling requires a connected topology", {"topology_id": topology.topology_id})
    routing = routing_result or simulate_deterministic(topology, configuration, TrafficMatrix("TM:empty", tuple(node.node_id for node in topology.nodes), ()))
    for attempt in range(max_attempts):
        rng = np.random.default_rng(derive_seed(seed, "policy", topology.topology_id, difficulty, attempt))
        policies = _sample_once(topology, graph, count, rng, attempt)
        if detect_policy_conflicts(policies):
            continue
        report = evaluate_schema_policies(policies, routing.routing_table, configuration, topology, routing.selected_bgp_routes)
        statuses = [item.status for item in report.evaluations]
        # Deterministic degradation filter: every policy must be structurally
        # evaluable, and a static task must retain both a satisfied and an
        # unsatisfied constraint.  No trained model is consulted.
        if set(statuses) <= {"satisfied", "unsatisfied"} and 0.0 < report.overall_consistency < 1.0:
            return policies, {"attempt": attempt, "initial_policy_consistency": report.overall_consistency, "initial_status_counts": dict(sorted(Counter(statuses).items())), "filter": "require 0 < consistency < 1 and no invalid/conflict/infeasible policy"}
    raise ScenarioGenerationError("policy_sampling_exhausted", "unable to sample a non-degenerate policy set", {"topology_id": topology.topology_id, "difficulty": difficulty, "seed": seed, "max_attempts": max_attempts})


def _sample_once(topology: Topology, graph: nx.Graph, count: int, rng: np.random.Generator, attempt: int) -> tuple[Policy, ...]:
    nodes = [node.node_id for node in topology.nodes]
    used_reachable: set[tuple[str, str]] = set()
    used_forward: set[tuple[str, str, str]] = set()
    used_isolation: set[tuple[str, str, str, str]] = set()
    policies: list[Policy] = []
    for index in range(count):
        for _ in range(256):
            source, destination = _pair(rng, nodes)
            if (source, destination) not in used_reachable:
                used_reachable.add((source, destination))
                policies.append(Policy(f"P:reachable:{attempt}:{index}", "reachable", {"source": source, "destination": destination}))
                break
        else: raise ScenarioGenerationError("reachable_sampling_exhausted", "no unique reachable pair", {"count": count})
    for index in range(count):
        for _ in range(256):
            source, destination = _pair(rng, nodes)
            waypoint = str(rng.choice([node for node in nodes if node not in {source, destination}]))
            key = (source, destination, waypoint)
            if key not in used_forward and _has_simple_waypoint_path(graph, source, waypoint, destination):
                used_forward.add(key)
                policies.append(Policy(f"P:forward:{attempt}:{index}", "forward_pass", {"source": source, "destination": destination, "waypoint": waypoint, "path_mode": "all_path"}))
                break
        else: raise ScenarioGenerationError("forward_sampling_exhausted", "no unique forward triple", {"count": count})
    for index in range(count):
        for _ in range(256):
            endpoints = [str(item) for item in rng.choice(nodes, size=4, replace=False)]
            first_source, first_destination, second_source, second_destination = endpoints
            key = tuple(endpoints)
            if key not in used_isolation:
                used_isolation.add(key)
                policies.append(Policy(f"P:isolation:{attempt}:{index}", "isolation", {"first_source": first_source, "first_destination": first_destination, "second_source": second_source, "second_destination": second_destination, "resource": "node"}))
                break
        else: raise ScenarioGenerationError("isolation_sampling_exhausted", "no unique isolation endpoint set", {"count": count})
    return tuple(policies)


def _pair(rng: np.random.Generator, nodes: list[str]) -> tuple[str, str]:
    pair = rng.choice(nodes, size=2, replace=False)
    return str(pair[0]), str(pair[1])


def _has_simple_waypoint_path(graph: nx.Graph, source: str, waypoint: str, destination: str) -> bool:
    """Require a simple source→waypoint→destination path, not two walks."""
    sink = "__netkeeper_waypoint_sink__"
    while sink in graph:
        sink += "_"
    augmented = graph.copy()
    augmented.add_edge(sink, source)
    augmented.add_edge(sink, destination)
    return nx.node_connectivity(augmented, waypoint, sink) >= 2


def detect_policy_conflicts(policies: Iterable[Policy]) -> tuple[str, ...]:
    """Return explicit duplicate/forward-pass-vs-avoid conflicts before eval."""
    conflicts: set[str] = set(); signatures: dict[tuple[str, tuple[tuple[str, Any], ...]], str] = {}
    passes: dict[tuple[str, str, str], str] = {}; avoids: dict[tuple[str, str, str], str] = {}
    for policy in policies:
        signature = (policy.kind, tuple(sorted(policy.fields.items())))
        if signature in signatures: conflicts.update((policy.policy_id, signatures[signature]))
        signatures[signature] = policy.policy_id
        if policy.kind not in {"forward_pass", "forward_avoid"}: continue
        source, destination = policy.fields.get("source"), policy.fields.get("destination")
        node = policy.fields.get("waypoint" if policy.kind == "forward_pass" else "forbidden_node")
        if all(isinstance(value, str) for value in (source, destination, node)):
            key = (source, destination, node)
            other = avoids.get(key) if policy.kind == "forward_pass" else passes.get(key)
            if other: conflicts.update((policy.policy_id, other))
            (passes if policy.kind == "forward_pass" else avoids)[key] = policy.policy_id
    return tuple(sorted(conflicts))


def generate_static_scenarios(dataset_root: str | Path, *, root_seed: int = 20260713, counts: Mapping[str, int] = TARGET_SCENARIO_COUNTS, output_directory: str = "scenarios", manifest_filename: str = "scenario_manifest.json", id_prefix: str = "S") -> dict[str, Any]:
    root = Path(dataset_root)
    topologies = json.loads((root / "metadata" / "topology_split.json").read_text(encoding="utf-8"))["splits"]
    traffic_manifest = json.loads((root / "metadata" / "traffic_manifest.json").read_text(encoding="utf-8"))
    config_by_topology = {item["topology_id"]: item for item in traffic_manifest["configurations"]}
    traffic_by_key = {(item["topology_id"], item["pattern"], item["load_level"]): item for item in traffic_manifest["matrices"]}
    outputs: dict[str, dict[str, Any]] = {}
    routing_cache: dict[str, Any] = {}
    for split, total in counts.items():
        difficulty_schedule = _schedule(total, DIFFICULTY_RATIOS[split], root_seed, split, "difficulty")
        traffic_schedule = _schedule(total, TRAFFIC_RATIOS, root_seed, split, "traffic")
        load_schedule = _schedule(total, LOAD_RATIOS, root_seed, split, "load")
        topology_records = sorted(topologies[split], key=lambda item: item["topology_id"])
        records: list[dict[str, Any]] = []
        for index in range(total):
            topology_info = topology_records[index % len(topology_records)]
            topology = Topology.from_dict(json.loads((root / topology_info["normalized_file"]).read_text(encoding="utf-8")))
            configuration_info = config_by_topology[topology.topology_id]
            configuration = NetworkConfiguration.from_dict(json.loads((root / configuration_info["file"]).read_text(encoding="utf-8")))
            routing_cache.setdefault(topology.topology_id, simulate_deterministic(topology, configuration, TrafficMatrix("TM:empty", tuple(node.node_id for node in topology.nodes), ())))
            difficulty, pattern, load = difficulty_schedule[index], traffic_schedule[index], load_schedule[index]
            scenario_seed = derive_seed(root_seed, "scenario", split, index)
            policy_seed = derive_seed(root_seed, "policy", split, index)
            policies, policy_metadata = sample_policies(topology, configuration, difficulty, seed=policy_seed, routing_result=routing_cache[topology.topology_id])
            traffic_info = traffic_by_key[topology.topology_id, pattern, load]
            scenario_id = f"{id_prefix}:{split}:{index:05d}"
            record = {
                "schema_version": SCHEMA_VERSION,
                "generator_version": SCENARIO_GENERATOR_VERSION,
                "scenario_id": scenario_id,
                "split": split,
                "topology_id": topology.topology_id,
                "topology": {"file": topology_info["normalized_file"], "sha256": topology_info["content_sha256"]},
                "seed": scenario_seed,
                "derived_seeds": {"scenario": scenario_seed, "policy": policy_seed, "traffic": traffic_info["seed"], "topology": configuration_info["bgp"]["seed"]},
                "initial_config": {"file": configuration_info["file"], "sha256": configuration_info["sha256"], "synthetic_bgp": configuration_info["bgp"]["synthetic"]},
                "difficulty": difficulty,
                "policies": [policy.to_dict() for policy in policies],
                "initial_policy": policy_metadata,
                "traffic": {key: traffic_info[key] for key in ("topology_id", "matrix_id", "pattern", "load_level", "load_multiplier", "seed", "base_matrix_file", "sha256", "node_order", "dtype", "shape", "unit", "parameters")},
                "events": [], "failures": [], "max_steps": 50,
            }
            record["content_sha256"] = _content_hash(record)
            records.append(record)
        path = Path(output_directory) / f"{split}.jsonl"
        _write_jsonl(root / path, records)
        outputs[split] = {"file": path.as_posix(), "count": len(records), "difficulty_quota": largest_remainder_quota(total, DIFFICULTY_RATIOS[split]), "traffic_quota": largest_remainder_quota(total, TRAFFIC_RATIOS), "load_quota": largest_remainder_quota(total, LOAD_RATIOS)}
    manifest = {"dataset_version": SCENARIO_DATASET_VERSION, "generator_version": SCENARIO_GENERATOR_VERSION, "root_seed": root_seed, "target_scenario_counts": TARGET_SCENARIO_COUNTS, "generated_scenario_counts": dict(counts), "difficulty_ratios": DIFFICULTY_RATIOS, "traffic_ratios": TRAFFIC_RATIOS, "load_ratios": LOAD_RATIOS, "policy_counts_per_kind": DIFFICULTY_COUNTS, "max_steps": 50, "splits": outputs}
    _write_text(root / "metadata" / manifest_filename, _canonical_json(manifest))
    return manifest


def generate_smoke_scenarios(dataset_root: str | Path, *, root_seed: int = 20260713, counts: Mapping[str, int] = SMOKE_SCENARIO_COUNTS) -> dict[str, Any]:
    return generate_static_scenarios(dataset_root, root_seed=root_seed, counts=counts, output_directory="scenarios/smoke", manifest_filename="scenario_generation_config.json", id_prefix="S:smoke")


def _schedule(total: int, ratios: Mapping[str, float], root_seed: int, split: str, name: str) -> list[str]:
    values = [key for key, count in largest_remainder_quota(total, ratios).items() for _ in range(count)]
    values.sort(key=lambda value: hashlib.sha256(f"{root_seed}:{split}:{name}:{value}".encode()).hexdigest())
    # Stable interleaving avoids long runs while retaining exact quotas.
    return [values[index % len(values)] for index in range(total)]


class ScenarioDataset:
    """Lazy JSONL iterator that materializes a unified NetworkScenario per row."""
    def __init__(self, dataset_root: str | Path, jsonl_file: str | Path) -> None:
        self.root = Path(dataset_root)
        self.path = self.root / jsonl_file

    def __iter__(self) -> Iterator[NetworkScenario]:
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield scenario_from_record(self.root, json.loads(line))


def scenario_from_record(root: str | Path, record: Mapping[str, Any]) -> NetworkScenario:
    root = Path(root)
    _verify_record_hash(record)
    topology_file, config_file = root / record["topology"]["file"], root / record["initial_config"]["file"]
    if _sha256_file(topology_file) != record["topology"]["sha256"] or _sha256_file(config_file) != record["initial_config"]["sha256"]:
        raise ValueError("topology_or_configuration_checksum_mismatch")
    topology = Topology.from_dict(json.loads(topology_file.read_text(encoding="utf-8")))
    configuration = NetworkConfiguration.from_dict(json.loads(config_file.read_text(encoding="utf-8")))
    traffic = load_traffic_record(root, record["traffic"])
    return NetworkScenario(str(record["scenario_id"]), topology, traffic, tuple(Policy.from_dict(item) for item in record["policies"]), configuration=configuration, events=(), max_steps=int(record["max_steps"]))


def validate_scenarios(dataset_root: str | Path, manifest: Mapping[str, Any] | None = None, *, check_environment: bool = False) -> dict[str, Any]:
    root = Path(dataset_root)
    default_manifest = root / "metadata" / "scenario_manifest.json"
    config = dict(manifest or json.loads((default_manifest if default_manifest.is_file() else root / "metadata" / "scenario_generation_config.json").read_text(encoding="utf-8")))
    topology_splits = json.loads((root / "metadata" / "topology_split.json").read_text(encoding="utf-8"))["splits"]
    owner = {item["topology_id"]: split for split, items in topology_splits.items() for item in items}
    seen: set[str] = set(); errors: list[dict[str, Any]] = []; totals: Counter[str] = Counter(); consistency: list[float] = []
    for split, info in config["splits"].items():
        path = root / info["file"]
        if not path.is_file(): errors.append({"code": "missing_scenario_file", "file": info["file"]}); continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                record = json.loads(line); _verify_record_hash(record)
                if record["scenario_id"] in seen: raise ValueError("duplicate_scenario_id")
                seen.add(record["scenario_id"])
                if record["split"] != split or owner.get(record["topology_id"]) != split: raise ValueError("topology_split_leakage")
                if Path(record["topology"]["file"]).is_absolute() or Path(record["initial_config"]["file"]).is_absolute() or Path(record["traffic"]["base_matrix_file"]).is_absolute(): raise ValueError("absolute_dataset_path")
                scenario = scenario_from_record(root, record)
                expected = DIFFICULTY_COUNTS[record["difficulty"]]
                kinds = Counter(policy.kind for policy in scenario.policies)
                if len(scenario.policies) != 3 * expected or any(kinds[kind] != expected for kind in ("reachable", "forward_pass", "isolation")): raise ValueError("policy_count_mismatch")
                _validate_policy_entities(scenario)
                if scenario.events or record["failures"] or scenario.max_steps != 50: raise ValueError("static_scenario_event_or_step_violation")
                if check_environment:
                    from netkeeper_sim.simulator import UnifiedNetworkEnvironment
                    UnifiedNetworkEnvironment().reset(scenario, seed=int(record["seed"]))
                totals[split] += 1; consistency.append(float(record["initial_policy"]["initial_policy_consistency"]))
            except Exception as exc: errors.append({"code": "invalid_scenario", "split": split, "line": line_number, "error": str(exc)})
    return {"valid": not errors, "scenario_counts": dict(totals), "unique_scenario_ids": len(seen), "initial_policy_consistency": {"min": min(consistency) if consistency else None, "max": max(consistency) if consistency else None, "mean": sum(consistency) / len(consistency) if consistency else None}, "errors": errors}


def _validate_policy_entities(scenario: NetworkScenario) -> None:
    nodes = {node.node_id for node in scenario.topology.nodes}
    for policy in scenario.policies:
        fields = policy.fields
        if policy.kind == "reachable":
            if fields.get("source") not in nodes or fields.get("destination") not in nodes or fields["source"] == fields["destination"]: raise ValueError("invalid_reachable_entity")
        elif policy.kind == "forward_pass":
            values = (fields.get("source"), fields.get("destination"), fields.get("waypoint"))
            if not set(values) <= nodes or len(set(values)) != 3: raise ValueError("invalid_forward_entity")
        elif policy.kind == "isolation":
            values = (fields.get("first_source"), fields.get("first_destination"), fields.get("second_source"), fields.get("second_destination"))
            if not set(values) <= nodes or len(set(values)) != 4: raise ValueError("invalid_isolation_entity")


def _content_hash(record: Mapping[str, Any]) -> str:
    value = {key: item for key, item in record.items() if key != "content_sha256"}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _verify_record_hash(record: Mapping[str, Any]) -> None:
    if record.get("content_sha256") != _content_hash(record): raise ValueError("scenario_content_hash_mismatch")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(value, encoding="utf-8")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
