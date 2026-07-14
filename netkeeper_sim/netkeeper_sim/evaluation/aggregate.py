"""Failure-preserving statistical aggregation."""
from __future__ import annotations
from collections import defaultdict
from statistics import mean, median, stdev
from typing import Any, Iterable, Mapping

def summary(values: list[float]) -> dict[str, float | int | None]:
    return {"count": len(values), "mean": mean(values) if values else None, "std": stdev(values) if len(values) > 1 else None, "median": median(values) if values else None, "min": min(values) if values else None, "max": max(values) if values else None}

def aggregate_runs(rows: Iterable[Mapping[str, Any]], *, group_by: tuple[str, ...] = ("method_name",)) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows: groups[tuple(row.get(key) for key in group_by)].append(row)
    output=[]
    for key, group in sorted(groups.items(), key=lambda x: repr(x[0])):
        completed=[row for row in group if row.get("status") in {"completed", "terminated", "truncated"}]
        result={name: value for name, value in zip(group_by, key)}
        result.update({"attempted": len(group), "succeeded": sum(bool(row.get("success")) for row in group), "failed": sum(row.get("status") not in {"completed", "terminated", "truncated"} for row in group), "censored": sum(bool(row.get("censored")) for row in group)})
        for field in (
            "policy_consistency_initial", "policy_consistency_final", "policy_consistency_best",
            "mlu_initial", "mlu_final", "mlu_best", "mlu_worst",
            "traffic_shift_step_project_final", "traffic_shift_total_project_final",
            "traffic_shift_step_paper_final", "traffic_shift_total_paper_final",
            "configuration_change_ratio", "configuration_change_count", "action_count",
            "convergence_step", "decision_time_ms", "simulator_time_ms", "wall_time_ms",
            "lookahead_candidate_evaluations", "lookahead_simulator_calls",
        ):
            result[field] = summary([float(row[field]) for row in completed if isinstance(row.get(field), (int, float))])
        recovery=[item for row in completed for item in row.get("event_recovery", ())]
        result["dynamic_recovery"]={"attempted":len(recovery),"recovered":sum(bool(x.get("recovered")) for x in recovery),"censored":sum(bool(x.get("censored")) for x in recovery),"recovery_steps":summary([float(x["recovery_steps"]) for x in recovery if isinstance(x.get("recovery_steps"),(int,float))])}
        # Repeated Random seeds are paired replications of the same scenario,
        # not additional independent topologies.  Publish both levels.
        by_scenario: dict[tuple[Any, Any], list[Mapping[str, Any]]] = defaultdict(list)
        for row in completed: by_scenario[(row.get("scenario_id"),row.get("sequence_id"))].append(row)
        scenario_means=[mean(float(x["policy_consistency_final"]) for x in values if isinstance(x.get("policy_consistency_final"),(int,float))) for values in by_scenario.values() if any(isinstance(x.get("policy_consistency_final"),(int,float)) for x in values)]
        within=[stdev(float(x["policy_consistency_final"]) for x in values if isinstance(x.get("policy_consistency_final"),(int,float))) for values in by_scenario.values() if sum(isinstance(x.get("policy_consistency_final"),(int,float)) for x in values)>1]
        result["replication"]={"independent_scenarios":len(by_scenario),"runs":len(completed),"policy_scenario_between":summary(scenario_means),"policy_random_seed_within_mean_std":mean(within) if within else None}
        output.append(result)
    return output
