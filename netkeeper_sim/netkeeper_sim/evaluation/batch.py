"""Deterministic serial batch execution, validation and presentation helpers."""
from __future__ import annotations
import hashlib, json, shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping

from netkeeper_sim.dataset.dynamic_sequences import dynamic_scenario
from netkeeper_sim.dataset.scenarios import scenario_from_record
from netkeeper_sim.evaluation.aggregate import aggregate_runs
from netkeeper_sim.evaluation.baselines import LocalSearchOSPFMethod, NoUpdateMethod, OSPFDefaultMethod, RandomMethod
from netkeeper_sim.evaluation.manifest import generate_evaluation_manifest
from netkeeper_sim.evaluation.methods import DispatcherMethodAdapter, canonical_hash
from netkeeper_sim.evaluation.results import EVALUATOR_VERSION, ResultStore, run_key
from netkeeper_sim.evaluation.runner import EvaluationRunner

METHOD_NAMES=("no_update","random","ospf_default","local_search_ospf","checkpoint")

def method_factory(name: str, config: Mapping[str, Any]):
    if name=="no_update": return NoUpdateMethod()
    if name=="random": return RandomMethod()
    if name=="ospf_default": return OSPFDefaultMethod()
    if name=="local_search_ospf": return LocalSearchOSPFMethod(candidate_budget=int(config.get("local_search_budget",64)),deltas=tuple(config.get("local_search_deltas",(1,2,4,8))))
    if name=="checkpoint": return DispatcherMethodAdapter(config.get("checkpoint") or "", config=config, checkpoint_status=str(config.get("checkpoint_status","debug_unconverged")))
    raise ValueError(f"unknown_method:{name}")

def _jsonl(path: Path): return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def _verify_formal_checkpoint(root: Path, config: Mapping[str, Any]) -> None:
    checkpoint=Path(str(config.get("checkpoint") or "")); bundle_path=Path(str(config.get("checkpoint_manifest") or checkpoint.parent/"checkpoint_manifest.json"))
    if not checkpoint.is_file() or not bundle_path.is_file(): raise ValueError("formal_checkpoint_or_manifest_missing")
    bundle=json.loads(bundle_path.read_text(encoding="utf-8")); digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if bundle.get("checkpoint_sha256") != digest or bundle.get("checkpoint_status") != "formal_validation_selected": raise ValueError("formal_checkpoint_hash_or_status_mismatch")
    resolved=Path(str(bundle.get("resolved_config_path") or ""))
    if not resolved.is_file() or bundle.get("resolved_config_sha256") != hashlib.sha256(resolved.read_bytes()).hexdigest(): raise ValueError("formal_checkpoint_resolved_config_mismatch")
    if bundle.get("model_version") != "rl-coma-v2" or bundle.get("schema_version") != "netkeeper-sim.schema.v1": raise ValueError("formal_checkpoint_model_schema_mismatch")
    if bundle.get("dataset_manifest_sha256") != hashlib.sha256((root/"metadata/manifest.json").read_bytes()).hexdigest(): raise ValueError("formal_checkpoint_dataset_mismatch")
    provenance=bundle.get("provenance") or {}; formalization=bundle.get("formalization") or {}
    if provenance.get("training_split") != "scenarios/train.jsonl" or provenance.get("validation_split") != "scenarios/validation.jsonl" or provenance.get("selection_metric") != "validation_total_reward" or formalization.get("test_split_accessed") is not False or formalization.get("strict_load") is not True or formalization.get("greedy_dispatch") is not True: raise ValueError("formal_checkpoint_selection_provenance_invalid")
    method=DispatcherMethodAdapter(checkpoint,config=config,checkpoint_status="formal_validation_selected")
    if method._load_error is not None: raise ValueError(f"formal_checkpoint_strict_load_failed:{method._load_error[1]}")

def plan(dataset_root: str | Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root=Path(dataset_root); random_seeds=tuple(int(x) for x in config.get("random_seeds",(20260714,20260715,20260716))); deterministic_seed=int(config.get("deterministic_seed",random_seeds[0])); manifest=generate_evaluation_manifest(root,seeds=random_seeds,deterministic_seed=deterministic_seed,max_steps=config.get("max_steps"),hold_steps=int(config.get("hold_steps",3)))
    if config.get("evaluation_manifest"):
        frozen=json.loads(Path(config["evaluation_manifest"]).read_text(encoding="utf-8"))
        if frozen.get("manifest_hash") != canonical_hash({key:value for key,value in frozen.items() if key!="manifest_hash"}): raise ValueError("evaluation_manifest_hash_mismatch")
        if frozen.get("dataset_manifest_sha256") != hashlib.sha256((root/"metadata/manifest.json").read_bytes()).hexdigest(): raise ValueError("evaluation_manifest_dataset_mismatch")
        for source in frozen.get("source_files",{}).values():
            if hashlib.sha256((root/source["file"]).read_bytes()).hexdigest() != source["sha256"]: raise ValueError("evaluation_manifest_source_mismatch")
        if frozen.get("method_seeds") != {"deterministic":[deterministic_seed],"random":list(random_seeds)}: raise ValueError("evaluation_manifest_seed_mismatch")
        thresholds=frozen.get("thresholds",{})
        if thresholds.get("hold_steps") != int(config.get("hold_steps",3)) or thresholds.get("recovery_budget_steps") != int(config.get("recovery_budget",30)): raise ValueError("evaluation_manifest_threshold_mismatch")
        if len(frozen.get("static",())) != 500 or len(frozen.get("dynamic",())) != 100: raise ValueError("evaluation_manifest_count_mismatch")
        manifest=frozen
    methods=tuple(config["methods"]); unknown=set(methods)-set(METHOD_NAMES)
    if unknown: raise ValueError(f"unknown_methods:{sorted(unknown)}")
    if "checkpoint" in methods and config.get("require_formal_checkpoint"): _verify_formal_checkpoint(root,config)
    static_rows={row["scenario_id"]:row for row in _jsonl(root/"scenarios/validation.jsonl") + _jsonl(root/"scenarios/test.jsonl")}
    dynamic_rows={row["sequence_id"]:row for row in _jsonl(root/"dynamic_sequences/test.jsonl")}
    requested_static=set(config.get("scenario_ids") or ())
    selected_static=([x for x in manifest["static"] if x["scenario_id"] in requested_static] if requested_static else list(manifest["static"]))
    # Validation smoke IDs are deliberately allowed only when explicitly
    # supplied; the frozen formal manifest remains the default test plan.
    for ident in sorted(requested_static-{x["scenario_id"] for x in selected_static}):
        row=static_rows.get(ident)
        if row is None: raise KeyError(f"unknown_scenario_id:{ident}")
        selected_static.append({"scenario_id":ident,"topology_id":row["topology_id"],"difficulty":row["difficulty"],"traffic_pattern":row["traffic"]["pattern"],"load_level":row["traffic"]["load_level"]})
    selected_dynamic=[x for x in manifest["dynamic"] if x["sequence_id"] in set(config.get("sequence_ids") or [x["sequence_id"] for x in manifest["dynamic"]])]
    if config.get("mode","static") == "static": selected_dynamic=[]
    if config.get("mode") == "dynamic": selected_static=[]
    tasks=[]; cfg_hash=canonical_hash(config)
    for method_name in methods:
        metadata=method_factory(method_name,config).metadata.__dict__
        for item in selected_static:
            row=static_rows.get(item["scenario_id"])
            if row is None: continue
            for seed in (random_seeds if method_name=="random" else (deterministic_seed,)):
                tasks.append({"kind":"static","method":method_name,"metadata":metadata,"scenario_id":item["scenario_id"],"seed":seed,"attributes":{"topology_id":row["topology_id"],"difficulty":row["difficulty"],"traffic_pattern":row["traffic"]["pattern"],"load_level":row["traffic"]["load_level"],"load_group":item.get("load_group")}})
        for item in selected_dynamic:
            row=dynamic_rows.get(item["sequence_id"])
            if row is None: continue
            for seed in (random_seeds if method_name=="random" else (deterministic_seed,)):
                tasks.append({"kind":"dynamic","method":method_name,"metadata":metadata,"sequence_id":item["sequence_id"],"scenario_id":item["scenario_id"],"seed":seed,"attributes":{"topology_id":row["topology_id"],"event_type":"|".join(x["kind"] for x in row["logical_events"])}})
    for item in tasks: item["run_key"]=run_key(item["metadata"],item["scenario_id"],item.get("sequence_id"),item["seed"],cfg_hash)
    run_manifest={"evaluator_version":EVALUATOR_VERSION,"evaluation_manifest":manifest,"resolved_config":dict(config),"evaluator_config_hash":cfg_hash,"tasks":tasks,"run_manifest_hash":canonical_hash({"config":config,"dataset_manifest":manifest["manifest_hash"],"tasks":tasks})}
    return run_manifest,tasks

def prepare_output(output: str | Path, run_manifest: Mapping[str,Any], *, resume: bool=False, overwrite: bool=False) -> ResultStore:
    root=Path(output)/f"evaluation-{run_manifest['evaluator_config_hash'][:12]}"
    if root.exists() and not resume and not overwrite: raise FileExistsError(f"output_exists:{root}; use --resume or --overwrite")
    if root.exists() and overwrite: shutil.rmtree(root)
    store=ResultStore(root); store.write_json("run_manifest.json",run_manifest); store.write_json("resolved_evaluation_config.json",run_manifest["resolved_config"]); return store

def execute(dataset_root: str | Path, config: Mapping[str,Any], store: ResultStore, run_manifest: Mapping[str,Any], *, progress: Callable[[str],None]=print) -> dict[str,int]:
    root=Path(dataset_root); cfg_hash=run_manifest["evaluator_config_hash"]; terminal={row["run_key"] for filename in ("episodes.jsonl","failures.jsonl") for row in store.read(filename)}; counts={"planned":len(run_manifest["tasks"]),"skipped":0,"completed":0,"failed":0}
    scenario_records={row["scenario_id"]:row for row in _jsonl(root/"scenarios/validation.jsonl")+_jsonl(root/"scenarios/test.jsonl")}; sequence_records={row["sequence_id"]:row for row in _jsonl(root/"dynamic_sequences/test.jsonl")}
    for number,task in enumerate(run_manifest["tasks"],1):
        if task["run_key"] in terminal: counts["skipped"]+=1; continue
        method=method_factory(task["method"],config); runner=EvaluationRunner(hold_steps=int(config.get("hold_steps",3)),evaluator_config_hash=cfg_hash)
        try:
            if task["kind"]=="static":
                scenario=scenario_from_record(root,scenario_records[task["scenario_id"]]); outcome=runner.run_static(method,scenario,seed=int(task["seed"]),run_id=run_manifest["run_manifest_hash"],max_steps=config.get("max_steps"))
            else:
                record=sequence_records[task["sequence_id"]]; outcome=runner.run_dynamic_sequence(method,root,record,seed=int(task["seed"]),run_id=run_manifest["run_manifest_hash"])
            summary=dict(outcome.summary); summary.update(task["attributes"]); summary["run_key"]=task["run_key"]; outcome=type(outcome)(summary,outcome.steps,outcome.event_recovery); runner.persist(store,outcome)
            counts["completed" if summary["status"] in {"completed","terminated","truncated"} else "failed"]+=1
            progress(f"[{number}/{counts['planned']}] {task['method']} {task['scenario_id']} seed={task['seed']} {summary['status']} {summary['wall_time_ms']:.1f}ms")
        except Exception as exc:
            failure={"run_key":task["run_key"],"status":"failed","failure_code":"batch_exception","failure_detail":str(exc),"method_name":task["method"],"method_version":task["metadata"]["version"],"scenario_id":task["scenario_id"],"sequence_id":task.get("sequence_id"),"seed":task["seed"],**task["attributes"]}; store.append_unique("failures.jsonl",failure,key=task["run_key"]); counts["failed"]+=1; progress(f"[{number}/{counts['planned']}] FAILED {task['method']} {exc}")
    return counts

def aggregate_output(store: ResultStore, *, group_by: tuple[str,...]=( "method_name",)) -> list[dict[str,Any]]:
    rows=store.read("episodes.jsonl")+store.read("failures.jsonl"); result=aggregate_runs(rows,group_by=group_by); store.write_json("aggregate.json",{"evaluator_version":EVALUATOR_VERSION,"group_by":list(group_by),"groups":result}); store.write_csv("aggregate.csv",[{**{key:value for key,value in row.items() if not isinstance(value,dict)},"success_rate":row["succeeded"]/row["attempted"] if row["attempted"] else None,"policy_mean_plus_std":_pm(row["policy_consistency_final"]),"mlu_mean_plus_std":_pm(row["mlu_final"])} for row in result]); store.write_csv("mean_plus_std.csv",[{key:value for key,value in row.items() if not isinstance(value,dict)}|{"policy_consistency":_pm(row["policy_consistency_final"]),"mlu":_pm(row["mlu_final"])} for row in result]); return result
def _pm(value): return None if value["mean"] is None else f"{value['mean']} ± {value['std']}" if value["std"] is not None else str(value["mean"])

def _finite(value):
    import math
    if isinstance(value,float): return math.isfinite(value)
    if isinstance(value,dict): return all(_finite(x) for x in value.values())
    if isinstance(value,list): return all(_finite(x) for x in value)
    return True

def validate_output(store: ResultStore) -> dict[str,Any]:
    manifest=store.read("run_manifest.json") if False else json.loads((store.root/"run_manifest.json").read_text(encoding="utf-8")); tasks={x["run_key"]:x for x in manifest["tasks"]}; terminals=store.read("episodes.jsonl")+store.read("failures.jsonl"); errors=[]; seen=set()
    for row in terminals:
        key=row.get("run_key")
        if not _finite(row): errors.append({"code":"non_finite_value","run_key":key})
        if key not in tasks: errors.append({"code":"unknown_run_key","run_key":key})
        elif key in seen: errors.append({"code":"duplicate_terminal","run_key":key})
        else: seen.add(key)
        if key in tasks and (row.get("method_version") != tasks[key]["metadata"]["version"] or row.get("scenario_id") != tasks[key]["scenario_id"] or row.get("seed") != tasks[key]["seed"]): errors.append({"code":"terminal_manifest_mismatch","run_key":key})
    missing=set(tasks)-seen
    for key in sorted(missing): errors.append({"code":"missing_terminal","run_key":key})
    by_run={}
    for row in store.read("steps.jsonl"):
        key=str(row.get("run_key","")).rsplit(":",1)[0]; by_run.setdefault(key,[]).append(row)
    for key,rows in by_run.items():
        for index,row in enumerate(sorted(rows,key=lambda x:x["step"])):
            if row["step"]!=index: errors.append({"code":"non_contiguous_step","run_key":key})
            if index and row["before_snapshot_id"]!=sorted(rows,key=lambda x:x["step"])[index-1]["after_snapshot_id"]: errors.append({"code":"snapshot_chain_mismatch","run_key":key})
            if not _finite(row): errors.append({"code":"non_finite_step","run_key":key})
    initial={}
    for row in terminals:
        if row.get("initial_snapshot_id"):
            group=(row.get("scenario_id"),row.get("sequence_id"),row.get("seed")); initial.setdefault(group,set()).add(row["initial_snapshot_id"])
    for group, hashes in initial.items():
        if len(hashes)>1: errors.append({"code":"initial_snapshot_mismatch","group":group,"hashes":sorted(hashes)})
    # Aggregation must be reproducible from raw terminal rows.
    aggregate_path=store.root/"aggregate.json"
    if aggregate_path.is_file():
        saved=json.loads(aggregate_path.read_text(encoding="utf-8")); recalculated=aggregate_runs(terminals,group_by=tuple(saved.get("group_by",["method_name"])))
        if saved.get("groups") != recalculated: errors.append({"code":"aggregate_recompute_mismatch"})
    return {"valid":not errors,"planned":len(tasks),"terminal":len(terminals),"errors":errors}
