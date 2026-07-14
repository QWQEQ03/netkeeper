"""Deterministic, offline natural-language intent/API dataset generation.

This module intentionally has no HTTP or model-client dependency.  Gold calls
are derived from real Block-2 snapshots and accepted only after Block-3's
validator and dry-run executor have accepted them.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from netkeeper_sim.api import API_REGISTRY, ApiCall, ApiRequest, execute, validate_request
from netkeeper_sim.dataset.scenarios import scenario_from_record
from netkeeper_sim.schemas import NetworkSnapshot
from netkeeper_sim.simulator import UnifiedNetworkEnvironment


INTENT_SCHEMA_VERSION = "netkeeper.intent.v1"
INTENT_GENERATOR_VERSION = "1.0.0"
DEFAULT_SEED = 20260714
SPLIT_COUNTS = {"train": 1200, "validation": 300, "test": 500}
LEVEL_RATIOS = {"1": 0.40, "2": 0.40, "3": 0.20}
ERROR_SPECS = (
    ("UNKNOWN_NODE", "UNKNOWN_NODE", "unknown_node"),
    ("UNKNOWN_LINK", "UNKNOWN_LINK", "unknown_link"),
    ("POLICY_NOT_FOUND", "POLICY_NOT_FOUND", "unknown_policy"),
    ("UNKNOWN_BGP_ROUTE", "UNKNOWN_BGP_ROUTE", "unknown_bgp"),
    ("INVALID_VALUE", "OUT_OF_RANGE", "out_of_range"),
    ("INVALID_ENDPOINTS", "INVALID_ENDPOINTS", "same_endpoint"),
    ("INVALID_WAYPOINT", "INVALID_WAYPOINT", "bad_waypoint"),
    ("OVERLAPPING_ISOLATION", "OVERLAPPING_ISOLATION", "overlap_isolation"),
    ("CONFLICTING_INTENT", "POLICY_CONFLICT", "policy_conflict"),
    ("CONFLICTING_INTENT", "TRAFFIC_OPERATION_CONFLICT", "traffic_conflict"),
    ("MISSING_ARGUMENT", "MISSING_ARGUMENT", "missing_argument"),
    ("AMBIGUOUS_REFERENCE", None, "ambiguous_reference"),
    ("UNSUPPORTED_OPERATION", "UNSUPPORTED_API", "unsupported_operation"),
    ("ORDER_CONFLICT", "ORDER_VIOLATION", "order_conflict"),
)


@dataclass(frozen=True)
class Template:
    template_id: str
    family_id: str
    splits: tuple[str, ...]
    level: str
    category: str
    api_sequence: tuple[str, ...]
    slots: tuple[str, ...]
    text: str


def _templates() -> tuple[Template, ...]:
    rows: list[Template] = []
    # Split-specific families are intentional evaluation isolation, not wording
    # variants.  Their syntactic constructions differ materially.
    wording = {
        "train": {"1": "DSL：执行 {verb}。", "2": "请{verb}。"},
        "validation": {"1": "指令[{verb}]。", "2": "网络管理员请求：{verb}。"},
        "test": {"1": "操作规范：{verb}。", "2": "请在当前网络中完成：{verb}。"},
    }
    api_words = {
        "add_reachable_policy": "添加从 {src} 到 {dst} 的可达策略",
        "add_forward_policy": "要求 {src} 到 {dst} 经由 {pass_node}",
        "add_avoid_policy": "要求 {src} 到 {dst} 避开 {avoid_node}",
        "add_isolation_policy": "隔离流 {first_src}-{first_dst} 与流 {second_src}-{second_dst}",
        "remove_policy": "删除策略 {policy_id}",
        "set_traffic_demand": "将 {src} 到 {dst} 的流量设为 {demand_bps} bps",
        "scale_traffic_demand": "将 {src} 到 {dst} 的流量乘以 {factor}",
        "set_traffic_hotspot": "把 {src} 到 {dst} 设为 {demand_bps} bps 热点",
        "set_link_state": "将链路 {link_id} 置为 {state}",
        "set_node_state": "将节点 {node_id} 置为 {state}",
        "set_ospf_weight": "将链路 {link_id} 的 OSPF 权重设为 {weight}",
        "set_bgp_local_pref": "将 BGP 路由 {router_id}/{prefix}/{next_hop} 的 LocalPref 设为 {value}",
        "set_bgp_as_path_length": "将 BGP 路由 {router_id}/{prefix}/{next_hop} 的 AS Path 长度设为 {length}",
        "set_bgp_med": "将 BGP 路由 {router_id}/{prefix}/{next_hop} 的 MED 设为 {value}",
        "set_link_bandwidth": "将链路 {link_id} 的有效带宽设为 {bandwidth_bps} bps",
        "set_link_capacity": "将链路 {link_id} 的容量设为 {capacity_bps} bps",
        "set_queue_length": "将链路 {link_id} 的队列设为 {queue_packets} packets",
        "get_network_state": "查询网络状态摘要",
        "optimize_network": "请求以 {objectives} 为目标的优化",
    }
    for split in SPLIT_COUNTS:
        for level in ("1", "2"):
            for api, verb in api_words.items():
                text = wording[split][level].format(verb=verb)
                rows.append(Template(f"{split}.{level}.{api}", f"{split}.single.{api}", (split,), level, API_REGISTRY[api].category, (api,), tuple(), text))
        for kind, text, sequence in (
            ("policy_opt", "先添加 {src} 到 {dst} 的可达策略，再请求优化以降低最大链路利用率。", ("add_reachable_policy", "optimize_network")),
            ("failure_reach_opt", "链路 {link_id} 发生故障；仍要求 {src} 到 {dst} 可达，并请求优化。", ("set_link_state", "add_reachable_policy", "optimize_network")),
            ("traffic_opt", "将 {src} 到 {dst} 的流量乘以 {factor}，随后请求优化。", ("scale_traffic_demand", "optimize_network")),
            ("policy_config", "添加 {src} 到 {dst} 的可达策略，并将 {config_link} 的 OSPF 权重设为 {weight}；需要后续优化。", ("add_reachable_policy", "set_ospf_weight")),
            ("recovery", "先使链路 {link_id} 故障，再恢复该链路，最后请求优化。", ("set_link_state", "set_link_state", "optimize_network")),
            ("read_policy_opt", "查询状态后，要求 {src} 到 {dst} 避开 {avoid_node}，并请求优化。", ("get_network_state", "add_avoid_policy", "optimize_network")),
        ):
            rows.append(Template(f"{split}.3.{kind}", f"{split}.multi.{kind}", (split,), "3", "multi_operation", sequence, tuple(), text))
    return tuple(rows)


TEMPLATES = _templates()


def _category(api: str) -> str:
    return API_REGISTRY[api].category


def largest_remainder(total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    exact = {key: total * value for key, value in ratios.items()}
    result = {key: int(value) for key, value in exact.items()}
    for key in sorted(exact, key=lambda k: (-(exact[k] - int(exact[k])), k))[: total - sum(result.values())]:
        result[key] += 1
    return result


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read_records(root: Path, split: str) -> list[dict[str, Any]]:
    path = root / "scenarios" / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _snapshot(root: Path, record: Mapping[str, Any]) -> NetworkSnapshot:
    scenario = scenario_from_record(root, record)
    return UnifiedNetworkEnvironment().reset(scenario, seed=int(record["seed"]))[0]


def _context(snapshot: NetworkSnapshot, serial: int) -> dict[str, Any]:
    nodes = [node.node_id for node in snapshot.topology.nodes]
    links = list(snapshot.topology.links)
    demands = [d for d in snapshot.traffic.demands if d.destination is not None]
    if not demands or not snapshot.configuration.bgp.routes:
        raise ValueError("scenario lacks OD demand or BGP route")
    isolation_pairs = {pair for p in snapshot.policies if p.kind == "isolation" for pair in ((p.fields.get("first_source"), p.fields.get("first_destination")), (p.fields.get("second_source"), p.fields.get("second_destination")))}
    forward_pass = {(p.fields.get("source"), p.fields.get("destination"), p.fields.get("waypoint")) for p in snapshot.policies if p.kind == "forward_pass"}
    candidates = [(d.source, d.destination) for d in demands if (d.source, d.destination) not in isolation_pairs]
    if not candidates: raise ValueError("scenario has no policy-safe OD")
    src, dst = candidates[serial % len(candidates)]
    assert dst is not None
    waypoint = next(node for node in nodes if node not in {src, dst} and (src, dst, node) not in forward_pass)
    other = [node for node in nodes if node not in {src, dst, waypoint}]
    link = links[serial % len(links)]
    route = snapshot.configuration.bgp.routes[serial % len(snapshot.configuration.bgp.routes)]
    attrs = snapshot.configuration.performance[link.link_id]
    return {"src": src, "dst": dst, "pass_node": waypoint, "avoid_node": waypoint,
            "first_src": src, "first_dst": dst, "second_src": other[0], "second_dst": other[1],
            "policy_id": snapshot.policies[serial % len(snapshot.policies)].policy_id,
            "link_id": link.link_id, "config_link": link.link_id, "node_id": nodes[serial % len(nodes)],
            "router_id": route.router_id, "prefix": route.prefix, "next_hop": route.next_hop,
            "demand_bps": int(max(1, demands[serial % len(demands)].traffic_rate_bps * 1.1)), "factor": 1.5,
            "state": "down", "weight": 20, "value": 32, "length": 4, "queue_packets": min(5000, max(0, attrs.queue_packets + 1)),
            "bandwidth_bps": max(attrs.capacity_bps, min(attrs.physical_bandwidth_bps, attrs.bandwidth_bps)),
            "capacity_bps": max(1, min(attrs.bandwidth_bps, attrs.capacity_bps)),
            "objectives": ["policy_consistency", "mlu"], "include": ["summary", "metrics"]}


def _call(api: str, c: Mapping[str, Any]) -> dict[str, Any]:
    keys = {
        "add_reachable_policy": ("src", "dst"), "add_forward_policy": ("src", "dst", "pass_node"), "add_avoid_policy": ("src", "dst", "avoid_node"),
        "add_isolation_policy": ("first_src", "first_dst", "second_src", "second_dst"), "remove_policy": ("policy_id",),
        "set_traffic_demand": ("src", "dst", "demand_bps"), "scale_traffic_demand": ("src", "dst", "factor"), "set_traffic_hotspot": ("src", "dst", "demand_bps"),
        "set_link_state": ("link_id", "state"), "set_node_state": ("node_id", "state"), "set_ospf_weight": ("link_id", "weight"),
        "set_bgp_local_pref": ("router_id", "prefix", "next_hop", "value"), "set_bgp_as_path_length": ("router_id", "prefix", "next_hop", "length"), "set_bgp_med": ("router_id", "prefix", "next_hop", "value"),
        "set_link_bandwidth": ("link_id", "bandwidth_bps"), "set_link_capacity": ("link_id", "capacity_bps"), "set_queue_length": ("link_id", "queue_packets"),
        "get_network_state": ("include",), "optimize_network": ("objectives",),
    }[api]
    return {"api": api, "arguments": {key: c[key] for key in keys}}


def _valid_calls(template: Template, c: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    if template.level != "3":
        return [_call(template.api_sequence[0], c)], False
    name = template.template_id.rsplit(".", 1)[-1]
    if name == "failure_reach_opt":
        return [_call("set_link_state", c), _call("add_reachable_policy", c), _call("optimize_network", c)], False
    if name == "recovery":
        down = dict(c); down["state"] = "down"; up = dict(c); up["state"] = "up"
        return [_call("set_link_state", down), _call("set_link_state", up), _call("optimize_network", c)], False
    if name == "policy_config":
        return [_call("add_reachable_policy", c), _call("set_ospf_weight", {**c, "link_id": c["config_link"]})], True
    if name == "read_policy_opt":
        return [_call("get_network_state", c), _call("add_avoid_policy", c), _call("optimize_network", c)], False
    return [_call(api, c) for api in template.api_sequence], False


def _invalid_condition(kind: str, c: dict[str, Any]) -> tuple[str, str | None, dict[str, Any] | None, str]:
    base = {"api_version": "v1", "request_id": "invalid", "calls": []}
    if kind == "unknown_node": base["calls"] = [{"api": "add_reachable_policy", "arguments": {"src": "R999", "dst": c["dst"]}}]
    elif kind == "unknown_link": base["calls"] = [{"api": "set_ospf_weight", "arguments": {"link_id": "L:missing", "weight": 20}}]
    elif kind == "unknown_policy": base["calls"] = [{"api": "remove_policy", "arguments": {"policy_id": "P:missing"}}]
    elif kind == "unknown_bgp": base["calls"] = [{"api": "set_bgp_med", "arguments": {"router_id": c["router_id"], "prefix": "P:missing", "next_hop": c["next_hop"], "value": 10}}]
    elif kind == "out_of_range": base["calls"] = [{"api": "set_ospf_weight", "arguments": {"link_id": c["link_id"], "weight": 999}}, {"api": "set_queue_length", "arguments": {"link_id": c["link_id"], "queue_packets": -1}}]
    elif kind == "same_endpoint": base["calls"] = [{"api": "set_traffic_demand", "arguments": {"src": c["src"], "dst": c["src"], "demand_bps": 1}}]
    elif kind == "bad_waypoint": base["calls"] = [{"api": "add_forward_policy", "arguments": {"src": c["src"], "dst": c["dst"], "pass_node": c["src"]}}]
    elif kind == "overlap_isolation": base["calls"] = [{"api": "add_isolation_policy", "arguments": {"first_src": c["src"], "first_dst": c["dst"], "second_src": c["src"], "second_dst": c["second_dst"]}}]
    elif kind == "policy_conflict": base["calls"] = [_call("add_forward_policy", c), _call("add_avoid_policy", c)]
    elif kind == "traffic_conflict": base["calls"] = [_call("set_traffic_demand", c), _call("scale_traffic_demand", c)]
    elif kind == "missing_argument": base["calls"] = [{"api": "set_ospf_weight", "arguments": {"link_id": c["link_id"]}}]
    elif kind == "ambiguous_reference": return "AMBIGUOUS_REFERENCE", None, None, "请把那条链路的权重调高。"
    elif kind == "unsupported_operation": base["calls"] = [{"api": "delete_router", "arguments": {"node_id": c["src"]}}]
    else: base["calls"] = [_call("set_ospf_weight", c), _call("add_reachable_policy", c)]
    code = next(spec[1] for spec in ERROR_SPECS if spec[2] == kind)
    translation = next(spec[0] for spec in ERROR_SPECS if spec[2] == kind)
    text = "拒绝测试：" + kind.replace("_", " ")
    return translation, code, base, text


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def _content_record(record: dict[str, Any]) -> dict[str, Any]:
    bare = {key: value for key, value in record.items() if key != "content_sha256"}
    record["content_sha256"] = _hash(bare)
    return record


def generate_intent_dataset(dataset_root: str | Path, *, output_directory: str = "intents", seed: int = DEFAULT_SEED, counts: Mapping[str, int] = SPLIT_COUNTS, smoke: bool = False) -> dict[str, Any]:
    root = Path(dataset_root); output = root / output_directory
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for split, total in counts.items():
        records = _read_records(root, split)
        templates = [t for t in TEMPLATES if split in t.splits]
        levels = largest_remainder(total, LEVEL_RATIOS)
        valid_total = total - (60 if split == "test" and total == 500 else 0)
        rows: list[dict[str, Any]] = []
        serial = 0
        snapshots: dict[str, NetworkSnapshot] = {}
        rewrite_indices = set(sorted(range(total), key=lambda index: _hash({"seed": seed, "split": split, "index": index}))[:round(total * 0.20)])
        for level, amount in levels.items():
            for offset in range(amount):
                is_invalid = split == "test" and total == 500 and len(rows) >= total - 60
                # Keep invalids in L3 tail: preserves exact global level quotas.
                if is_invalid and level != "3":
                    continue
                # A bounded, split-local pool keeps full validation fast while
                # still cycling all topology IDs (static generation itself is
                # topology round-robin).
                record = records[serial % min(len(records), 24)]; serial += 1
                if record["scenario_id"] not in snapshots:
                    snapshots[record["scenario_id"]] = _snapshot(root, record)
                snapshot = snapshots[record["scenario_id"]]; c = _context(snapshot, serial)
                if is_invalid:
                    spec = ERROR_SPECS[(len(rows) - (total - 60)) % len(ERROR_SPECS)]
                    trans, validator_code, raw, text = _invalid_condition(spec[2], c)
                    expected = {"status": "rejected", "calls": [], "need_optimization": False, "error": {"code": f"TRANSLATION_{trans}", "reason": spec[2]}}
                    category, template = "invalid", next(t for t in templates if t.level == "3")
                    calls, need_opt = [], False
                else:
                    candidates = [t for t in templates if t.level == level]
                    template = candidates[offset % len(candidates)]
                    calls, need_opt = _valid_calls(template, c)
                    text = template.text.format(**c)
                    category = template.category
                    expected = {"status": "accepted", "calls": calls, "need_optimization": need_opt}
                    raw, validator_code = None, None
                intent_id = f"I:{split}:{len(rows):05d}"
                row = {"schema_version": INTENT_SCHEMA_VERSION, "generator_version": INTENT_GENERATOR_VERSION, "intent_id": intent_id, "split": split,
                       "topology_id": snapshot.topology.topology_id, "scenario": {"scenario_id": record["scenario_id"], "file": f"scenarios/{split}.jsonl", "seed": record["seed"], "snapshot_id": snapshot.snapshot_id},
                       "category": category, "level": int(level), "template_id": template.template_id, "family_id": template.family_id,
                       "text_source": "template", "rewrite_selected": len(rows) in rewrite_indices,
                       "original_text": text, "natural_language": text, "expected_translation": expected, "expected_calls": calls,
                       "need_optimization": need_opt, "is_valid": not is_invalid, "expected_error": None if not is_invalid else {"translation_code": expected["error"]["code"], "validator_code": validator_code, "invalid_request": raw}, "seed": seed}
                rows.append(_content_record(row))
        # Invalids replace the final sixty Level-3 valid rows; add enough if
        # the level loop did not reach them (only relevant to formal test).
        if split == "test" and total == 500:
            valid_rows = rows[:440]
            invalid_rows: list[dict[str, Any]] = []
            for i in range(60):
                record = records[(440 + i) % min(len(records), 24)]
                if record["scenario_id"] not in snapshots:
                    snapshots[record["scenario_id"]] = _snapshot(root, record)
                snapshot = snapshots[record["scenario_id"]]; c = _context(snapshot, 440 + i)
                spec = ERROR_SPECS[i % len(ERROR_SPECS)]; trans, validator_code, raw, text = _invalid_condition(spec[2], c)
                template = next(t for t in templates if t.level == "3")
                expected = {"status": "rejected", "calls": [], "need_optimization": False, "error": {"code": f"TRANSLATION_{trans}", "reason": spec[2]}}
                row = {"schema_version": INTENT_SCHEMA_VERSION, "generator_version": INTENT_GENERATOR_VERSION, "intent_id": f"I:test:{440+i:05d}", "split": split, "topology_id": snapshot.topology.topology_id, "scenario": {"scenario_id": record["scenario_id"], "file": "scenarios/test.jsonl", "seed": record["seed"], "snapshot_id": snapshot.snapshot_id}, "category": "invalid", "level": 3, "template_id": template.template_id, "family_id": template.family_id, "text_source": "template", "rewrite_selected": (440+i) in rewrite_indices, "original_text": text, "natural_language": text, "expected_translation": expected, "expected_calls": [], "need_optimization": False, "is_valid": False, "expected_error": {"translation_code": expected["error"]["code"], "validator_code": validator_code, "invalid_request": raw}, "seed": seed}
                invalid_rows.append(_content_record(row))
            rows = valid_rows + invalid_rows
        if len(rows) != total: raise AssertionError((split, len(rows), total))
        all_rows[split] = rows; _write_jsonl(output / f"{split}.jsonl", rows)
    if "train" in all_rows:
        _write_jsonl(output / "few_shot_candidates.jsonl", _few_shot(all_rows["train"]))
    config = {"schema_version": INTENT_SCHEMA_VERSION, "generator_version": INTENT_GENERATOR_VERSION, "seed": seed, "counts": dict(counts), "level_ratios": LEVEL_RATIOS, "rewrite_ratio": 0.20, "offline_only": True, "paths_relative_to_dataset_root": True}
    (output / "generation_config.yaml").write_text(yaml.safe_dump(config, sort_keys=True, allow_unicode=True), encoding="utf-8")
    stats = dataset_statistics(all_rows)
    (output / "dataset_statistics.json").write_text(json.dumps(stats, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    template_stats = {"templates": [{"template_id": t.template_id, "family_id": t.family_id, "splits": list(t.splits), "level": t.level, "category": t.category, "api_sequence": list(t.api_sequence), "slots": list(t.slots)} for t in TEMPLATES]}
    (output / "template_statistics.json").write_text(json.dumps(template_stats, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output / "random_seeds.json").write_text(json.dumps({"intent_root_seed": seed}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    manifest = {"schema_version": INTENT_SCHEMA_VERSION, "files": []}
    for path in sorted(p for p in output.iterdir() if p.is_file() and p.name != "manifest.json"):
        manifest["files"].append({"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return stats


def _few_shot(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted: set[tuple[int, str]] = {(1, api) for api in API_REGISTRY} | {(2, api) for api in API_REGISTRY} | {(3, "multi_operation")}
    selected = []
    for row in rows:
        api = row["expected_calls"][0]["api"] if row["expected_calls"] else row["category"]
        key = (row["level"], api)
        if key in wanted:
            selected.append({"intent_id": row["intent_id"], "split": "train", "natural_language": row["natural_language"], "expected_translation": row["expected_translation"], "fixed_candidate": True})
            wanted.remove(key)
    return selected


def dataset_statistics(rows_by_split: Mapping[str, Iterable[Mapping[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {"splits": {}}
    for split, iterable in rows_by_split.items():
        rows = list(iterable); api = Counter(call["api"] for row in rows for call in row["expected_calls"])
        result["splits"][split] = {"count": len(rows), "levels": dict(Counter(str(row["level"]) for row in rows)), "apis": dict(sorted(api.items())), "categories": dict(Counter(row["category"] for row in rows)), "validity": dict(Counter(str(row["is_valid"]).lower() for row in rows)), "operations": dict(Counter("multi" if len(row["expected_calls"]) > 1 else "single" for row in rows)), "errors": dict(Counter((row["expected_error"] or {}).get("translation_code", "") for row in rows if not row["is_valid"])), "families": dict(Counter(row["family_id"] for row in rows)), "rewrite_selected": sum(bool(row["rewrite_selected"]) for row in rows)}
    return result


def validate_intent_dataset(dataset_root: str | Path, *, directory: str = "intents", dry_run: bool = True) -> dict[str, Any]:
    root = Path(dataset_root); base = root / directory; errors: list[dict[str, Any]] = []; rows_by_split: dict[str, list[dict[str, Any]]] = {}
    topology_splits = json.loads((root / "metadata" / "topology_split.json").read_text(encoding="utf-8"))["splits"]
    owner = {item["topology_id"]: split for split, items in topology_splits.items() for item in items}
    snapshot_cache: dict[tuple[str, str], tuple[Mapping[str, Any], NetworkSnapshot]] = {}
    environment_cache: dict[tuple[str, str], UnifiedNetworkEnvironment] = {}
    for split, expected_count in SPLIT_COUNTS.items():
        rows = [json.loads(line) for line in (base / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]; rows_by_split[split] = rows
        if len(rows) != expected_count: errors.append({"code": "count", "split": split})
        for row in rows:
            if row.get("content_sha256") != _hash({k: v for k, v in row.items() if k != "content_sha256"}): errors.append({"code": "hash", "id": row.get("intent_id")}); continue
            if owner.get(row["topology_id"]) != split or row["scenario"]["file"] != f"scenarios/{split}.jsonl": errors.append({"code": "split_leak", "id": row["intent_id"]}); continue
            key = (split, row["scenario"]["scenario_id"])
            if key not in snapshot_cache:
                record = next(r for r in _read_records(root, split) if r["scenario_id"] == key[1])
                snapshot_cache[key] = (record, _snapshot(root, record))
            record, snapshot = snapshot_cache[key]
            if snapshot.snapshot_id != row["scenario"]["snapshot_id"]: errors.append({"code": "snapshot", "id": row["intent_id"]}); continue
            if row["is_valid"]:
                req = ApiRequest("v1", row["intent_id"], tuple(ApiCall.from_dict(call) for call in row["expected_calls"]), snapshot.snapshot_id, row["need_optimization"], dry_run)
                checked = validate_request(req, snapshot)
                if not checked.valid: errors.append({"code": "valid_rejected", "id": row["intent_id"], "errors": [e.code for e in checked.errors]}); continue
                if dry_run:
                    if key not in environment_cache:
                        env = UnifiedNetworkEnvironment()
                        env.reset(scenario_from_record(root, record), seed=int(record["seed"]))
                        environment_cache[key] = env
                    env = environment_cache[key]
                    if not execute(env, snapshot, req).success: errors.append({"code": "dry_run", "id": row["intent_id"]})
            else:
                err = row["expected_error"]
                if row["expected_calls"] or row["expected_translation"]["status"] != "rejected": errors.append({"code": "invalid_shape", "id": row["intent_id"]})
                if err["invalid_request"] is not None:
                    checked = validate_request(err["invalid_request"], snapshot)
                    if checked.valid or err["validator_code"] not in [e.code for e in checked.errors]: errors.append({"code": "invalid_condition", "id": row["intent_id"]})
    train_families = {r["family_id"] for r in rows_by_split["train"]}; test_families = {r["family_id"] for r in rows_by_split["test"]}
    if train_families & test_families: errors.append({"code": "family_leak"})
    ids = [r["intent_id"] for rows in rows_by_split.values() for r in rows]
    if len(ids) != len(set(ids)): errors.append({"code": "duplicate_id"})
    if sum(not r["is_valid"] for r in rows_by_split["test"]) != 60: errors.append({"code": "invalid_count"})
    return {"valid": not errors, "errors": errors, **dataset_statistics(rows_by_split)}
