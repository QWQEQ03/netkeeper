"""Test-only dynamic Event sequences compatible with UnifiedNetworkEnvironment."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import networkx as nx
import numpy as np

from netkeeper_sim.dataset.scenarios import scenario_from_record
from netkeeper_sim.dataset.traffic import derive_seed
from netkeeper_sim.schemas import Event, NetworkScenario, Policy, TrafficDemand
from netkeeper_sim.schemas.ids import SCHEMA_VERSION


DYNAMIC_DATASET_VERSION = "netkeeper-lite.dynamic-sequences.v1"
DYNAMIC_GENERATOR_VERSION = "1.0.0"
DEFAULT_SEQUENCE_COUNT = 100
SMOKE_SEQUENCE_COUNT = 4
RECOVERY_BUDGET_STEPS = 30


def generate_dynamic_sequences(dataset_root: str | Path, *, count: int = DEFAULT_SEQUENCE_COUNT, root_seed: int = 20260713, output_file: str = "dynamic_sequences/test.jsonl") -> dict[str, Any]:
    root = Path(dataset_root)
    eligible = []
    # A deterministic bounded candidate pool avoids decoding every formal test
    # matrix merely to build 100 cyclic sequences.  Twenty-four candidates
    # cover the four test topologies repeatedly while preserving test-only use.
    for record in _test_scenario_records(root):
        if _fault_candidates(scenario_from_record(root, record)) is not None:
            eligible.append(record)
        if len(eligible) == 24:
            break
    if not eligible:
        raise ValueError("no test scenario has non-partitioning link/node fault candidates")
    records: list[dict[str, Any]] = []
    for index in range(count):
        sequence_seed = derive_seed(root_seed, "event", "dynamic", index)
        # Stable cycling distributes sequences over only test split scenario IDs.
        base = eligible[index % len(eligible)]
        scenario = scenario_from_record(root, base)
        record = _sequence_record(scenario, base, index, sequence_seed, root_seed)
        records.append(record)
    path = root / output_file
    _write_jsonl(path, records)
    manifest = {
        "dataset_version": DYNAMIC_DATASET_VERSION,
        "generator_version": DYNAMIC_GENERATOR_VERSION,
        "root_seed": root_seed,
        "split": "test",
        "count": count,
        "file": output_file,
        "recovery_budget_steps": RECOVERY_BUDGET_STEPS,
        "logical_event_plan": ["policy_add", "traffic_scale", "link_failure_recovery", "policy_remove", "hotspot_change", "node_failure_recovery"],
        "expected_valid": True,
    }
    _write_text(root / "metadata" / ("dynamic_smoke_manifest.json" if count != DEFAULT_SEQUENCE_COUNT else "dynamic_sequences_manifest.json"), _canonical_json(manifest))
    return manifest


def _test_scenario_records(root: Path) -> Iterator[dict[str, Any]]:
    manifest = root / "metadata" / "scenario_manifest.json"
    config = json.loads((manifest if manifest.is_file() else root / "metadata" / "scenario_generation_config.json").read_text(encoding="utf-8"))
    path = root / config["splits"]["test"]["file"]
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sequence_record(scenario: NetworkScenario, base: Mapping[str, Any], index: int, sequence_seed: int, root_seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(sequence_seed)
    candidates = _fault_candidates(scenario)
    if candidates is None:
        raise ValueError("base scenario does not have safe dynamic fault candidates")
    link_id, node_id = candidates
    nodes = [node.node_id for node in scenario.topology.nodes]
    existing_pairs = {(policy.fields.get("source"), policy.fields.get("destination")) for policy in scenario.policies if policy.kind in {"reachable", "forward_pass", "forward_avoid"}}
    source, destination = _new_pair(rng, nodes, existing_pairs)
    added_policy = Policy(f"P:dynamic:{index}:add", "reachable", {"source": source, "destination": destination})
    removed_policy = sorted(scenario.policies, key=lambda item: item.policy_id)[0]
    demand = scenario.traffic.demands[int(rng.integers(len(scenario.traffic.demands)))]
    replacement = TrafficDemand(demand.source, demand.traffic_rate_bps * 2.0, destination=demand.destination, prefix=demand.prefix, traffic_class=demand.traffic_class)
    times = (0, 30, 60, 90, 120, 150, 180, 210)
    sequence_id = f"DS:test:{index:04d}"
    events = (
        Event(f"E:{sequence_id}:policy-add", times[0], "policy_add", {"policy": added_policy.to_dict()}),
        Event(f"E:{sequence_id}:traffic-scale", times[1], "traffic_scale", {"factor": 1.5}),
        Event(f"E:{sequence_id}:link-down", times[2], "link_down", target_id=link_id),
        Event(f"E:{sequence_id}:link-up", times[3], "link_up", target_id=link_id),
        Event(f"E:{sequence_id}:policy-remove", times[4], "policy_remove", target_id=removed_policy.policy_id),
        Event(f"E:{sequence_id}:hotspot", times[5], "hotspot_change", {"demand": replacement.to_dict()}),
        Event(f"E:{sequence_id}:node-down", times[6], "node_down", target_id=node_id),
        Event(f"E:{sequence_id}:node-up", times[7], "node_up", target_id=node_id),
    )
    record = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": DYNAMIC_GENERATOR_VERSION,
        "sequence_id": sequence_id,
        "split": "test",
        "topology_id": scenario.topology.topology_id,
        "initial_scenario_id": scenario.scenario_id,
        "initial_scenario_file": "scenarios/test.jsonl",
        "seed": sequence_seed,
        "derived_seeds": {"event": sequence_seed, "policy": derive_seed(root_seed, "policy", sequence_id), "traffic": derive_seed(root_seed, "traffic", sequence_id), "topology": derive_seed(root_seed, "topology", scenario.topology.topology_id)},
        "events": [event.to_dict() for event in events],
        "logical_events": [
            {"category": "Forwarding", "kind": "policy_add", "event_ids": [events[0].event_id]},
            {"category": "Network traffic", "kind": "traffic_scale", "event_ids": [events[1].event_id]},
            {"category": "Physical", "kind": "link_failure_recovery", "event_ids": [events[2].event_id, events[3].event_id], "expected_partition": False},
            {"category": "Forwarding", "kind": "policy_remove", "event_ids": [events[4].event_id]},
            {"category": "Network traffic", "kind": "hotspot_change", "event_ids": [events[5].event_id]},
            {"category": "Physical", "kind": "node_failure_recovery", "event_ids": [events[6].event_id, events[7].event_id], "expected_partition": False},
        ],
        "recovery_budget_steps": RECOVERY_BUDGET_STEPS,
        "dynamic_max_steps": times[-1] + RECOVERY_BUDGET_STEPS,
        "expected_valid": True,
    }
    record["content_sha256"] = _content_hash(record)
    return record


def _fault_candidates(scenario: NetworkScenario) -> tuple[str, str] | None:
    graph = nx.MultiGraph()
    graph.add_nodes_from(node.node_id for node in scenario.topology.nodes)
    for link in scenario.topology.links:
        graph.add_edge(link.source, link.target, key=link.link_id)
    safe_links = []
    for link in scenario.topology.links:
        candidate = graph.copy(); candidate.remove_edge(link.source, link.target, key=link.link_id)
        if nx.is_connected(nx.Graph(candidate)):
            safe_links.append(link.link_id)
    critical = _policy_critical_nodes(scenario)
    safe_nodes = []
    for node in scenario.topology.nodes:
        if node.node_id in critical: continue
        candidate = graph.copy(); candidate.remove_node(node.node_id)
        if candidate.number_of_nodes() and nx.is_connected(nx.Graph(candidate)):
            safe_nodes.append(node.node_id)
    if not safe_links or not safe_nodes:
        return None
    return sorted(safe_links)[0], sorted(safe_nodes)[0]


def _policy_critical_nodes(scenario: NetworkScenario) -> set[str]:
    keys = ("source", "destination", "waypoint", "first_source", "first_destination", "second_source", "second_destination")
    return {str(policy.fields[key]) for policy in scenario.policies for key in keys if isinstance(policy.fields.get(key), str)}


def _new_pair(rng: np.random.Generator, nodes: list[str], existing: set[tuple[Any, Any]]) -> tuple[str, str]:
    pairs = [(source, destination) for source in nodes for destination in nodes if source != destination and (source, destination) not in existing]
    if not pairs: raise ValueError("no unused policy pair for dynamic addition")
    return pairs[int(rng.integers(len(pairs)))]


def dynamic_scenario(root: str | Path, sequence: Mapping[str, Any]) -> NetworkScenario:
    root = Path(root)
    base = next(record for record in _test_scenario_records(root) if record["scenario_id"] == sequence["initial_scenario_id"])
    scenario = scenario_from_record(root, base)
    return replace(scenario, events=tuple(Event.from_dict(item) for item in sequence["events"]), max_steps=int(sequence["dynamic_max_steps"]))


def validate_dynamic_sequences(dataset_root: str | Path, manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    root = Path(dataset_root)
    config = dict(manifest or json.loads((root / "metadata" / "dynamic_sequences_manifest.json").read_text(encoding="utf-8")))
    test_records = {record["scenario_id"]: record for record in _test_scenario_records(root)}
    test_topology_ids = {record["topology_id"] for record in test_records.values()}
    ids: set[str] = set(); errors: list[dict[str, Any]] = []; coverage: dict[str, int] = {}
    path = root / config["file"]
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(line); _verify_hash(record)
            if record["sequence_id"] in ids: raise ValueError("duplicate_sequence_id")
            ids.add(record["sequence_id"])
            if record["split"] != "test" or record["topology_id"] not in test_topology_ids or record["initial_scenario_id"] not in test_records: raise ValueError("non_test_reference")
            scenario = dynamic_scenario(root, record)
            _validate_events(scenario, record["events"])
            for logical in record["logical_events"]: coverage[logical["kind"]] = coverage.get(logical["kind"], 0) + 1
        except Exception as exc: errors.append({"line": line_number, "code": "invalid_dynamic_sequence", "error": str(exc)})
    return {"valid": not errors, "count": len(ids), "coverage": coverage, "errors": errors}


def _validate_events(scenario: NetworkScenario, raw_events: Iterable[Mapping[str, Any]]) -> None:
    events = tuple(Event.from_dict(item) for item in raw_events)
    if len(events) != 8 or tuple(event.step for event in events) != tuple(sorted(event.step for event in events)) or len({event.step for event in events}) != len(events): raise ValueError("events_not_strictly_ordered")
    links = {link.link_id for link in scenario.topology.links}; nodes = {node.node_id for node in scenario.topology.nodes}
    link_state = {link: "up" for link in links}; node_state = {node: "up" for node in nodes}; policies = {policy.policy_id for policy in scenario.policies}
    for event in events:
        if event.kind == "link_down":
            if event.target_id not in links or link_state[event.target_id] != "up": raise ValueError("invalid_link_down")
            link_state[event.target_id] = "down"
        elif event.kind == "link_up":
            if event.target_id not in links or link_state[event.target_id] != "down": raise ValueError("invalid_link_up")
            link_state[event.target_id] = "up"
        elif event.kind == "node_down":
            if event.target_id not in nodes or node_state[event.target_id] != "up": raise ValueError("invalid_node_down")
            node_state[event.target_id] = "down"
        elif event.kind == "node_up":
            if event.target_id not in nodes or node_state[event.target_id] != "down": raise ValueError("invalid_node_up")
            node_state[event.target_id] = "up"
        elif event.kind == "policy_add":
            policy = Policy.from_dict(event.payload["policy"])
            if policy.policy_id in policies: raise ValueError("duplicate_dynamic_policy")
            policies.add(policy.policy_id)
        elif event.kind == "policy_remove":
            if event.target_id not in policies: raise ValueError("unknown_dynamic_policy")
            policies.remove(event.target_id)
        elif event.kind == "traffic_scale":
            if not isinstance(event.payload.get("factor"), (int, float)) or event.payload["factor"] <= 0: raise ValueError("invalid_traffic_scale")
        elif event.kind == "hotspot_change":
            demand = TrafficDemand.from_dict(event.payload["demand"])
            if demand.source not in nodes or demand.destination not in nodes or demand.source == demand.destination: raise ValueError("invalid_hotspot_demand")
    if "down" in link_state.values() or "down" in node_state.values(): raise ValueError("unpaired_physical_fault")


def _content_hash(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps({key: value for key, value in record.items() if key != "content_sha256"}, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _verify_hash(record: Mapping[str, Any]) -> None:
    if record.get("content_sha256") != _content_hash(record): raise ValueError("sequence_content_hash_mismatch")


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records), encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(value, encoding="utf-8")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
