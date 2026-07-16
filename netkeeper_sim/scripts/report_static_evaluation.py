"""Generate Block-7 static tables and figures from validated raw JSONL."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

import matplotlib.pyplot as plt


METHOD_ORDER = ["coma_dispatcher", "no_update", "random", "ospf_default", "local_search_ospf"]
METHOD_LABEL = {
    "coma_dispatcher": "NetKeeper",
    "no_update": "No Update",
    "random": "Random",
    "ospf_default": "OSPF Default",
    "local_search_ospf": "Local Search OSPF",
}
DIFFICULTY_ORDER = ["Overall", "Easy", "Medium", "Hard"]
LOAD_ORDER = ["Normal", "Hotspot", "Burst", "High-load"]
COMPLETED = {"completed", "terminated", "truncated"}


def read_jsonl(path: Path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stat(values):
    values = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    return {
        "count": len(values),
        "mean": mean(values) if values else None,
        "std": stdev(values) if len(values) > 1 else None,
        "median": median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def flatten(prefix, value, output):
    if isinstance(value, dict):
        for key, item in value.items():
            flatten(f"{prefix}_{key}" if prefix else key, item, output)
    else:
        output[prefix] = value


def write_csv(path: Path, rows):
    rows = list(rows)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def grouped_rows(rows, field, values):
    for value in values:
        yield value, rows if value == "Overall" else [row for row in rows if row.get(field) == value]


def aggregate_quality(rows):
    output = []
    for difficulty, subset in grouped_rows(rows, "difficulty", DIFFICULTY_ORDER):
        for method in METHOD_ORDER:
            group = [row for row in subset if row.get("method_name") == method]
            completed = [row for row in group if row.get("status") in COMPLETED]
            record = {
                "difficulty": difficulty,
                "method": method,
                "method_label": METHOD_LABEL[method],
                "scenario_count": len({row.get("scenario_id") for row in group}),
                "run_count": len(group),
                "completed": len(completed),
                "failed": len(group) - len(completed),
                "censored": sum(bool(row.get("censored")) for row in group),
                "success_rate": mean([bool(row.get("success")) for row in group]) if group else None,
            }
            for field in ("policy_consistency_final", "policy_consistency_best", "convergence_step"):
                flatten(field, stat([row.get(field) for row in completed]), record)
            output.append(record)
    return output


def aggregate_load(rows):
    output = []
    for load in LOAD_ORDER:
        subset = [row for row in rows if row.get("load_group") == load]
        for method in METHOD_ORDER:
            group = [row for row in subset if row.get("method_name") == method]
            completed = [row for row in group if row.get("status") in COMPLETED]
            record = {
                "load_group": load,
                "method": method,
                "method_label": METHOD_LABEL[method],
                "scenario_count": len({row.get("scenario_id") for row in group}),
                "run_count": len(group),
                "completed": len(completed),
                "failed": len(group) - len(completed),
                "censored": sum(bool(row.get("censored")) for row in group),
                "congestion_rate_final_mlu_gt_1": mean([row.get("mlu_final", 0) > 1 for row in completed]) if completed else None,
            }
            for field in ("mlu_initial", "mlu_final", "mlu_best", "mlu_worst", "policy_consistency_final", "policy_consistency_best"):
                flatten(field, stat([row.get(field) for row in completed]), record)
            output.append(record)
    return output


def step_shift_statistics(step_path: Path):
    fields = (
        "traffic_shift_step_paper_v1",
        "traffic_shift_total_paper_v1",
        "traffic_shift_step_project_v1",
        "traffic_shift_total_project_v1",
    )
    values = defaultdict(lambda: {field: [] for field in fields})
    last = defaultdict(dict)
    if not step_path.is_file():
        return {}
    with step_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = row["run_key"].rsplit(":", 1)[0]
            metrics = row.get("metrics", {})
            for field in fields:
                value = metrics.get(field)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    values[key][field].append(float(value))
                    last[key][field] = float(value)
    output = {}
    for key, grouped in values.items():
        record = {}
        for version in ("paper", "project"):
            step_field = f"traffic_shift_step_{version}_v1"
            total_field = f"traffic_shift_total_{version}_v1"
            step_values = grouped[step_field]
            record.update({
                f"{version}_step_mean": mean(step_values) if step_values else None,
                f"{version}_step_peak": max(step_values) if step_values else None,
                f"{version}_step_final": last[key].get(step_field),
                f"{version}_trajectory_total_final": last[key].get(total_field),
            })
        output[key] = record
    return output


def aggregate_shift(rows, shifts):
    raw = []
    for row in rows:
        item = {
            "run_key": row.get("run_key"), "scenario_id": row.get("scenario_id"),
            "method": row.get("method_name"), "method_label": METHOD_LABEL.get(row.get("method_name"), row.get("method_name")),
            "seed": row.get("seed"), "status": row.get("status"), "success": row.get("success"),
            "censored": row.get("censored"), "policy_consistency_final": row.get("policy_consistency_final"),
            "configuration_change_ratio": row.get("configuration_change_ratio"),
        }
        item.update(shifts.get(row.get("run_key"), {}))
        raw.append(item)
    aggregate = []
    for method in METHOD_ORDER:
        group = [row for row in raw if row["method"] == method]
        record = {
            "method": method, "method_label": METHOD_LABEL[method], "run_count": len(group),
            "failed": sum(row["status"] not in COMPLETED for row in group),
            "censored": sum(bool(row["censored"]) for row in group),
        }
        for field in (
            "paper_step_mean", "paper_step_peak", "paper_step_final", "paper_trajectory_total_final",
            "project_step_mean", "project_step_peak", "project_step_final", "project_trajectory_total_final",
            "configuration_change_ratio", "policy_consistency_final",
        ):
            flatten(field, stat([row.get(field) for row in group]), record)
        aggregate.append(record)
    return raw, aggregate


def paired_deltas(rows):
    metrics = ("policy_consistency_final", "policy_consistency_best", "mlu_final", "mlu_best", "success")
    by = defaultdict(list)
    for row in rows:
        if row.get("status") in COMPLETED:
            by[(row.get("scenario_id"), row.get("method_name"))].append(row)
    scenario_method = {}
    for key, group in by.items():
        scenario_method[key] = {
            metric: mean([float(row[metric]) for row in group if isinstance(row.get(metric), (int, float, bool))])
            for metric in metrics
            if any(isinstance(row.get(metric), (int, float, bool)) for row in group)
        }
    output = []
    for baseline in METHOD_ORDER[1:]:
        common = sorted({scenario for scenario, method in scenario_method if method == "coma_dispatcher"} & {scenario for scenario, method in scenario_method if method == baseline})
        record = {"baseline": baseline, "baseline_label": METHOD_LABEL[baseline], "paired_scenarios": len(common), "delta_definition": "NetKeeper - baseline"}
        for metric in metrics:
            values = [scenario_method[(scenario, "coma_dispatcher")][metric] - scenario_method[(scenario, baseline)][metric] for scenario in common if metric in scenario_method[(scenario, "coma_dispatcher")] and metric in scenario_method[(scenario, baseline)]]
            flatten(f"delta_{metric}", stat(values), record)
        output.append(record)
    return output


def runtime_cost(rows):
    output=[]
    for method in METHOD_ORDER:
        group=[row for row in rows if row.get("method_name")==method and row.get("status") in COMPLETED]
        record={"method":method,"method_label":METHOD_LABEL[method],"run_count":len(group)}
        for field in ("decision_time_ms","simulator_time_ms","wall_time_ms","lookahead_candidate_evaluations","lookahead_simulator_calls"):
            flatten(field,stat([row.get(field) for row in group]),record)
        output.append(record)
    return output


def bar_chart(path, rows, categories, category_field, value_field, ylabel, title):
    width=0.16; x=list(range(len(categories))); fig,ax=plt.subplots(figsize=(10,5.5))
    for mi,method in enumerate(METHOD_ORDER):
        selected={(row[category_field],row["method"]):row for row in rows}
        records=[selected.get((category,method),{}) for category in categories]
        vals=[record.get(value_field,0) or 0 for record in records]
        bars=ax.bar([value+(mi-2)*width for value in x],vals,width,label=METHOD_LABEL[method])
        for bar,value,record in zip(bars,vals,records):
            n=record.get("scenario_count",record.get("run_count"))
            label=f"{value:.3g}" + (f"\nn={n}" if n is not None else "")
            ax.annotate(label,(bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,3),textcoords="offset points",ha="center",va="bottom",fontsize=6,rotation=90)
    ax.set_xticks(x,categories); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(fontsize=8); ax.grid(axis="y",alpha=.25); fig.tight_layout()
    fig.savefig(path.with_suffix(".png"),dpi=180); fig.savefig(path.with_suffix(".pdf")); plt.close(fig)


def shift_configuration_scatter(path, rows):
    fig,ax=plt.subplots(figsize=(8,5.5))
    for method in METHOD_ORDER:
        group=[row for row in rows if row.get("method")==method and isinstance(row.get("paper_trajectory_total_final"),(int,float)) and isinstance(row.get("configuration_change_ratio"),(int,float))]
        ax.scatter([row["configuration_change_ratio"] for row in group],[row["paper_trajectory_total_final"] for row in group],s=14,alpha=.35,label=f"{METHOD_LABEL[method]} (n={len(group)})")
    ax.set_xlabel("Configuration change ratio")
    ax.set_ylabel("Paper-v1 trajectory total")
    ax.set_title("Traffic Shift versus configuration modification (all statuses with metrics)")
    ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(path.with_suffix(".png"),dpi=180); fig.savefig(path.with_suffix(".pdf")); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--run-root",required=True); args=parser.parse_args()
    root=Path(args.run_root); manifest=json.loads((root/"run_manifest.json").read_text(encoding="utf-8"))
    episodes=read_jsonl(root/"episodes.jsonl"); failures=read_jsonl(root/"failures.jsonl"); rows=episodes+failures
    planned=len(manifest["tasks"])
    if len(rows)!=planned:
        raise RuntimeError(f"refusing formal report: terminal {len(rows)} != planned {planned}")
    out=root/"reports"/"static"; out.mkdir(parents=True,exist_ok=True)
    quality=aggregate_quality(rows); load=aggregate_load(rows); shifts=step_shift_statistics(root/"steps.jsonl"); shift_raw,shift_agg=aggregate_shift(rows,shifts)
    paired=paired_deltas(rows); runtime=runtime_cost(rows)
    failure_rows=[row for row in rows if row.get("status") not in COMPLETED or row.get("censored")]
    composition=[]
    for group in LOAD_ORDER:
        selected=[item for item in manifest["evaluation_manifest"]["static"] if item.get("load_group")==group]
        counts=defaultdict(int)
        for item in selected: counts[(item.get("traffic_pattern"),item.get("load_level"))]+=1
        for (pattern,level),count in sorted(counts.items()): composition.append({"load_group":group,"traffic_pattern":pattern,"load_level":level,"scenario_count":count})
    for filename,data in (("quality.csv",quality),("load.csv",load),("load_composition.csv",composition),("traffic_shift_runs.csv",shift_raw),("traffic_shift_aggregate.csv",shift_agg),("paired_delta.csv",paired),("runtime_cost.csv",runtime),("failure_censored.csv",failure_rows)):
        write_csv(out/filename,data)
    bar_chart(out/"pc_final_by_difficulty",quality,DIFFICULTY_ORDER,"difficulty","policy_consistency_final_mean","Policy Consistency","Final PC by difficulty (n shown in CSV)")
    bar_chart(out/"success_rate_by_difficulty",quality,DIFFICULTY_ORDER,"difficulty","success_rate","Success rate","Static success by difficulty")
    bar_chart(out/"mlu_final_by_load",load,LOAD_ORDER,"load_group","mlu_final_mean","MLU","Final MLU by frozen load group")
    bar_chart(out/"traffic_shift_paper",shift_agg,["Overall"],"_overall","paper_trajectory_total_final_mean","Paper-v1 shift","Traffic Shift (see CSV for n/status)") if False else None
    # Method-only plots use a synthetic category for the common bar helper.
    for row in shift_agg: row["scope"]="Overall"
    for row in runtime: row["scope"]="Overall"
    bar_chart(out/"traffic_shift_paper",shift_agg,["Overall"],"scope","paper_trajectory_total_final_mean","Paper-v1 trajectory total","Traffic Shift by method")
    bar_chart(out/"configuration_change",shift_agg,["Overall"],"scope","configuration_change_ratio_mean","Changed field ratio","Configuration modification by method")
    shift_configuration_scatter(out/"traffic_shift_vs_configuration",shift_raw)
    bar_chart(out/"runtime",runtime,["Overall"],"scope","wall_time_ms_mean","Wall time (ms)","Runtime by method")
    bar_chart(out/"local_search_candidate_cost",runtime,["Overall"],"scope","lookahead_candidate_evaluations_mean","Candidate evaluations/run","Local Search additional compute")
    failure_plot=[]
    for method in METHOD_ORDER:
        group=[row for row in rows if row.get("method_name")==method]
        failure_plot.append({"scope":"Failed","method":method,"value":sum(row.get("status") not in COMPLETED for row in group),"run_count":len(group)})
        failure_plot.append({"scope":"Censored","method":method,"value":sum(bool(row.get("censored")) for row in group),"run_count":len(group)})
    bar_chart(out/"failure_censored",failure_plot,["Failed","Censored"],"scope","value","Run count","Failure and censoring counts")
    summary={"evaluator_version":manifest["evaluator_version"],"evaluator_config_hash":manifest["evaluator_config_hash"],"planned":planned,"terminal":len(rows),"failed":len(failures),"censored":sum(bool(row.get("censored")) for row in rows),"output":str(out)}
    (out/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(summary,sort_keys=True))


if __name__=="__main__":
    main()
