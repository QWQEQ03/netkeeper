"""Reproducible, label-isolated evaluation for Block-4 translation modes."""
from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from netkeeper_sim.api import ApiCall, ApiRequest, execute
from netkeeper_sim.dataset.intents import _read_records, _snapshot
from netkeeper_sim.dataset.scenarios import scenario_from_record
from netkeeper_sim.intent.translation import RecordingDispatcher, TranslationRunResult, Translator
from netkeeper_sim.simulator import UnifiedNetworkEnvironment


MODES = ("prompt_only", "few_shot", "full")

def _canonical(call: Mapping[str, Any]) -> str:
    return json.dumps({"api": call.get("api"), "arguments": call.get("arguments")}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def _pair_calls(predicted: list[Mapping[str, Any]], gold: list[Mapping[str, Any]]) -> tuple[int, int, int]:
    """Stable order-preserving one-to-one exact-call match."""
    used: set[int] = set(); matches = 0
    for call in predicted:
        signature = _canonical(call)
        for index, target in enumerate(gold):
            if index not in used and signature == _canonical(target):
                used.add(index); matches += 1; break
    return matches, len(predicted), len(gold)

def _semantic(snapshot, after, gold_calls: list[Mapping[str, Any]]) -> bool | None:
    """Check only direct Block-3 effects; optimization quality is intentionally N/A."""
    direct = [call for call in gold_calls if call["api"] not in {"get_network_state", "optimize_network"}]
    if not direct: return None
    for call in direct:
        api, a = call["api"], call["arguments"]
        if api == "add_reachable_policy" and not any(p.kind == "reachable" and p.fields.get("source") == a["src"] and p.fields.get("destination") == a["dst"] for p in after.policies): return False
        if api == "add_forward_policy" and not any(p.kind == "forward_pass" and p.fields.get("source") == a["src"] and p.fields.get("destination") == a["dst"] and p.fields.get("waypoint") == a["pass_node"] for p in after.policies): return False
        if api == "add_avoid_policy" and not any(p.kind == "forward_avoid" and p.fields.get("source") == a["src"] and p.fields.get("destination") == a["dst"] and p.fields.get("forbidden_node") == a["avoid_node"] for p in after.policies): return False
        if api == "remove_policy" and any(p.policy_id == a["policy_id"] for p in after.policies): return False
        if api == "set_link_state" and after.configuration.link_states.get(a["link_id"]) != a["state"]: return False
        if api == "set_node_state" and after.configuration.node_states.get(a["node_id"]) != a["state"]: return False
        if api == "set_ospf_weight" and after.configuration.ospf_weights.get(a["link_id"]) != a["weight"]: return False
        if api.startswith("set_bgp_") and not any((r.router_id, r.prefix, r.next_hop) == (a["router_id"], a["prefix"], a["next_hop"]) and ((api == "set_bgp_local_pref" and r.local_preference == a["value"]) or (api == "set_bgp_med" and r.med == a["value"]) or (api == "set_bgp_as_path_length" and len(r.as_path) == a["length"])) for r in after.configuration.bgp.routes): return False
        if api in {"set_link_bandwidth", "set_link_capacity", "set_queue_length"}:
            field = {"set_link_bandwidth": "bandwidth_bps", "set_link_capacity": "capacity_bps", "set_queue_length": "queue_packets"}[api]
            key = {"set_link_bandwidth": "bandwidth_bps", "set_link_capacity": "capacity_bps", "set_queue_length": "queue_packets"}[api]
            if getattr(after.configuration.performance[a["link_id"]], field) != a[key]: return False
        if api == "add_isolation_policy" and not any(p.kind == "isolation" and p.fields.get("first_source") == a["first_src"] and p.fields.get("first_destination") == a["first_dst"] and p.fields.get("second_source") == a["second_src"] and p.fields.get("second_destination") == a["second_dst"] for p in after.policies): return False
        if api in {"set_traffic_demand", "set_traffic_hotspot"}:
            demand = next((d for d in after.traffic.demands if (d.source, d.destination) == (a["src"], a["dst"])), None)
            if demand is None or demand.traffic_rate_bps != float(a["demand_bps"]): return False
        if api == "scale_traffic_demand":
            before = next((d for d in snapshot.traffic.demands if (d.source, d.destination) == (a["src"], a["dst"])), None)
            demand = next((d for d in after.traffic.demands if (d.source, d.destination) == (a["src"], a["dst"])), None)
            if before is None or demand is None or demand.traffic_rate_bps != before.traffic_rate_bps * float(a["factor"]): return False
    return True

def _error_group(result: TranslationRunResult) -> str:
    if result.status != "failed": return "none"
    code = str((result.error or {}).get("code", ""))
    if code in {"INVALID_JSON", "JSON_FALLBACK_REJECTED", "TRANSLATION_SCHEMA_INVALID", "MALFORMED_PROVIDER_RESPONSE", "EMPTY_RESPONSE"}: return "json_schema_failure"
    if code in {"NETWORK_ERROR", "MISSING_API_KEY", "ONLINE_DISABLED"} or code.startswith("HTTP_"): return "model_network_failure"
    return "validation_failure"

def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def evaluate_dataset(translator: Translator, dataset_root: str | Path, *, split: str, mode: str, output_directory: str | Path, online: bool = False, start: int = 0, stop: int | None = None, resume: bool = True) -> dict[str, Any]:
    if mode not in MODES: raise ValueError("invalid evaluation mode")
    root, output = Path(dataset_root), Path(output_directory); output.mkdir(parents=True, exist_ok=True)
    path = output / f"{split}.{mode}.jsonl"; existing = {row["intent_id"] for row in _rows(path)} if resume and path.is_file() else set()
    intents = _rows(root / "intents" / f"{split}.jsonl")[start:stop]
    records = {record["scenario_id"]: record for record in _read_records(root, split)}
    with path.open("a", encoding="utf-8") as handle:
        for intent in intents:
            if intent["intent_id"] in existing: continue
            record = records[intent["scenario"]["scenario_id"]]; snapshot = _snapshot(root, record)
            env = UnifiedNetworkEnvironment(); env.reset(scenario_from_record(root, record), seed=int(record["seed"]))
            dispatcher = RecordingDispatcher()
            # Gold is deliberately used only below this call for scoring.
            run = translator.translate(intent["natural_language"], snapshot, request_id=intent["intent_id"], mode=mode, online=online, scenario_id=record["scenario_id"], env=env, run_executor=True, dispatcher=dispatcher)
            gold = intent["expected_translation"]; predicted = run.translation or {"status": "failed", "calls": []}
            p_calls, g_calls = list(predicted.get("calls", [])), list(gold.get("calls", [])); matched, p_count, g_count = _pair_calls(p_calls, g_calls)
            positional_api = sum(1 for i, call in enumerate(g_calls) if i < len(p_calls) and p_calls[i].get("api") == call.get("api"))
            arg_total = sum(len(call.get("arguments", {})) for call in g_calls)
            arg_correct = sum(sum(p_calls[i].get("arguments", {}).get(k) == v for k, v in call.get("arguments", {}).items()) for i, call in enumerate(g_calls) if i < len(p_calls) and p_calls[i].get("api") == call.get("api"))
            execution_success = bool(run.execution and run.execution.get("success"))
            semantic = _semantic(snapshot, env.current_snapshot, g_calls) if execution_success and intent["is_valid"] else None
            item = {"intent_id": intent["intent_id"], "mode": mode, "prediction": predicted, "gold": gold, "is_valid": intent["is_valid"], "api_name_correct": positional_api, "api_name_total": g_count, "argument_correct": arg_correct, "argument_total": arg_total, "call_matches": matched, "prediction_call_count": p_count, "gold_call_count": g_count, "exact_match": predicted == gold, "accept_reject_correct": predicted.get("status") == gold.get("status"), "execution_success": execution_success, "semantic_success": semantic, "optimization_dispatched": bool(dispatcher.requests), "error_group": _error_group(run), "attempts": run.attempts, "feedback_corrected": run.attempts == 2 and run.status == "accepted", "cache_hit": run.cache_hit, "latency_ms": (run.metadata or {}).get("elapsed_ms", 0.0), "token_usage": (run.metadata or {}).get("token_usage", {}), "run_status": run.status, "run_error": run.error}
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    rows = _rows(path); summary = aggregate(rows); (output / f"{split}.{mode}.json").write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    _write_csv(output / f"{split}.{mode}.csv", rows)
    return summary

def aggregate(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows); n = len(rows)
    api_correct, api_total = sum(r["api_name_correct"] for r in rows), sum(r["api_name_total"] for r in rows)
    arg_correct, arg_total = sum(r["argument_correct"] for r in rows), sum(r["argument_total"] for r in rows)
    match, p_total, g_total = sum(r["call_matches"] for r in rows), sum(r["prediction_call_count"] for r in rows), sum(r["gold_call_count"] for r in rows)
    invalid = [r for r in rows if not r["is_valid"]]; tp=sum(r["prediction"].get("status")=="rejected" for r in invalid); fp=sum(r["prediction"].get("status")=="rejected" for r in rows if r["is_valid"]); fn=len(invalid)-tp
    precision = match/p_total if p_total else 0.0; recall = match/g_total if g_total else 0.0; f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    reject_precision=tp/(tp+fp) if tp+fp else 0.0; reject_recall=tp/(tp+fn) if tp+fn else 0.0
    semantic=[r for r in rows if r["semantic_success"] is not None]
    return {"sample_count": n, "api_name_accuracy": api_correct/api_total if api_total else 0.0, "argument_accuracy": arg_correct/arg_total if arg_total else 0.0, "exact_match": sum(r["exact_match"] for r in rows)/n if n else 0.0, "accept_reject_accuracy": sum(r["accept_reject_correct"] for r in rows)/n if n else 0.0, "invalid_rejection": {"precision": reject_precision, "recall": reject_recall, "f1": 2*reject_precision*reject_recall/(reject_precision+reject_recall) if reject_precision+reject_recall else 0.0}, "execution_success_rate": sum(r["execution_success"] for r in rows)/n if n else 0.0, "semantic_success_rate": sum(r["semantic_success"] is True for r in semantic)/len(semantic) if semantic else None, "semantic_applicable_count": len(semantic), "call_precision": precision, "call_recall": recall, "call_f1": f1, "json_schema_failure_count": sum(r["error_group"]=="json_schema_failure" for r in rows), "validation_failure_count": sum(r["error_group"]=="validation_failure" for r in rows), "model_network_failure_count": sum(r["error_group"]=="model_network_failure" for r in rows), "feedback_correction_success_rate": sum(r["feedback_corrected"] for r in rows)/sum(r["attempts"]==2 for r in rows) if any(r["attempts"]==2 for r in rows) else 0.0, "average_call_count": p_total/n if n else 0.0, "average_latency_ms": sum(float(r["latency_ms"] or 0) for r in rows)/n if n else 0.0, "token_usage": {"total_tokens": sum(int((r["token_usage"] or {}).get("total_tokens", 0)) for r in rows)}, "cache_hit_rate": sum(r["cache_hit"] for r in rows)/n if n else 0.0, "optimization_dispatched_count": sum(r["optimization_dispatched"] for r in rows)}

def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields = ["intent_id", "mode", "is_valid", "exact_match", "accept_reject_correct", "execution_success", "semantic_success", "error_group", "attempts", "cache_hit", "latency_ms"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows([{k: row.get(k) for k in fields} for row in rows])
