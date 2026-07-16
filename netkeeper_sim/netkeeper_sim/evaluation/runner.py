"""Schema-only static/dynamic episode runners; no routing logic lives here."""
from __future__ import annotations
import random, sys, traceback
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any, Mapping

from tqdm import tqdm

from netkeeper_sim.evaluation.methods import EvaluationContext, EvaluationMethod, MethodDecision
from netkeeper_sim.evaluation.results import EVALUATOR_VERSION, ResultStore, configuration_change, run_key
from netkeeper_sim.schemas import JointAction, NetworkScenario
from netkeeper_sim.simulator import UnifiedNetworkEnvironment
from netkeeper_sim.dataset.scenarios import ScenarioDataset

@dataclass(frozen=True)
class RunOutcome:
    summary: Mapping[str, Any]
    steps: tuple[Mapping[str, Any], ...]
    event_recovery: tuple[Mapping[str, Any], ...] = ()

def _metric(snapshot):
    return snapshot.metrics.to_dict()
def _success(snapshot, target_mlu, hold):
    return snapshot.metrics.policy_consistency >= 1.0 and (target_mlu is None or snapshot.metrics.maximum_link_utilization <= target_mlu)

class EvaluationRunner:
    def __init__(self, *, hold_steps: int = 3, evaluator_config_hash: str = "evaluation-v1") -> None:
        self.hold_steps,self.evaluator_config_hash=hold_steps,evaluator_config_hash
    def run_static(self, method: EvaluationMethod, scenario: NetworkScenario, *, seed: int, run_id: str = "run", max_steps: int | None = None, sequence_id: str | None = None, recovery_budget_steps: int | None = None, run_full_horizon: bool = False, show_step_progress: bool = False) -> RunOutcome:
        environment=UnifiedNetworkEnvironment(); started=perf_counter_ns(); snapshot, observation=environment.reset(scenario, seed=seed); initial=snapshot
        context=EvaluationContext(run_id, scenario, scenario.scenario_id, seed, initial, min(max_steps or scenario.max_steps, scenario.max_steps), sequence_id, recovery_budget_steps, {"hold_steps":self.hold_steps})
        try: method.reset(context)
        except Exception as exc: return self._failed(method, context, initial, (), started, "method_reset_error", exc)
        records=[]; consecutive=0; convergence=None; actions=0; status="completed"; failure=None
        step_pbar=None
        if show_step_progress: step_pbar=tqdm(total=context.max_steps,desc=f"  {method.metadata.name}",unit="step",leave=False,file=sys.stderr,dynamic_ncols=True)
        try:
            for index in range(context.max_steps):
                before=snapshot; wall_before=perf_counter_ns(); decision_before=perf_counter_ns()
                try:
                    raw=method.act(snapshot, observation, context)
                    decision=raw if isinstance(raw, MethodDecision) else MethodDecision(raw, perf_counter_ns()-decision_before)
                    if not isinstance(decision.action, JointAction): raise TypeError("method did not return JointAction")
                    if decision.status != "ok":
                        status=decision.status; failure=decision.failure_code or "method_unavailable"; break
                    if decision.action.snapshot_id not in (None, snapshot.snapshot_id): raise ValueError("stale_action")
                    illegal = [item.parameter_type for item in decision.action.actions if item.mode != "no_update" and item.parameter_type not in method.metadata.allowed_parameter_types]
                    if illegal: raise ValueError("method_permission_violation:" + ",".join(sorted(set(illegal))))
                    actions += sum(a.mode != "no_update" for a in decision.action.actions)
                    simulator_before=perf_counter_ns(); result=environment.step(snapshot, decision.action); simulator=perf_counter_ns()-simulator_before
                except MemoryError as exc: return self._failed(method, context, initial, tuple(records), started, "oom", exc)
                except Exception as exc: return self._failed(method, context, initial, tuple(records), started, "exception", exc)
                snapshot, observation=result.next_snapshot, result.observations
                ok=_success(snapshot, scenario.target_mlu, self.hold_steps); consecutive=consecutive+1 if ok else 0
                if consecutive >= self.hold_steps and convergence is None: convergence=snapshot.step
                cumulative_change=configuration_change(initial.configuration, snapshot.configuration)
                record={"step":index,"before_snapshot_id":before.snapshot_id,"after_snapshot_id":snapshot.snapshot_id,"action":decision.action.to_dict(),"method_diagnostics":dict(decision.diagnostics),"configuration_diff":dict(result.changed_config),"configuration_change_from_initial_count":cumulative_change["count"],"configuration_change_from_initial_ratio":cumulative_change["ratio"],"rewards":result.rewards.to_dict(),"metrics":result.metrics.to_dict(),"terminated":result.terminated,"truncated":result.truncated,"done_reason":result.done_reason,"errors":[e.to_dict() for e in result.errors],"decision_time_ms":decision.decision_time_ns/1e6,"simulator_time_ms":simulator/1e6,"wall_time_ms":(perf_counter_ns()-wall_before)/1e6}
                records.append(record)
                if step_pbar: step_pbar.update(1); step_pbar.set_postfix_str(f"pc={snapshot.metrics.policy_consistency:.2f} mlu={snapshot.metrics.maximum_link_utilization:.2f}")
                if result.errors:
                    status="environment_rejected"; failure=result.errors[0].code; break
                # Environment termination denotes that the instantaneous target is
                # met.  The evaluator still verifies the frozen hold window; a
                # dynamic run must additionally consume its complete event horizon.
                if result.truncated: status="truncated"; break
                if convergence is not None and not run_full_horizon: status="terminated" if result.terminated else "completed"; break
        finally:
            if step_pbar: step_pbar.close()
        final=snapshot; change=configuration_change(initial.configuration, final.configuration)
        success=status in {"completed", "terminated", "truncated"} and (convergence is not None or (_success(final, scenario.target_mlu, self.hold_steps) and self.hold_steps <= 1))
        summary={"evaluator_version":EVALUATOR_VERSION,"run_id":run_id,"method_name":method.metadata.name,"method_version":method.metadata.version,"method_metadata":method.metadata.__dict__,"scenario_id":scenario.scenario_id,"sequence_id":sequence_id,"topology_id":scenario.topology.topology_id,"seed":seed,"status":status,"failure_code":failure,"initial_snapshot_id":initial.snapshot_id,"final_snapshot_id":final.snapshot_id,"policy_consistency_initial":initial.metrics.policy_consistency,"policy_consistency_final":final.metrics.policy_consistency,"policy_consistency_best":max([initial.metrics.policy_consistency]+[r["metrics"]["policy_consistency"] for r in records]),"mlu_initial":initial.metrics.maximum_link_utilization,"mlu_final":final.metrics.maximum_link_utilization,"mlu_best":min([initial.metrics.maximum_link_utilization]+[r["metrics"]["maximum_link_utilization"] for r in records]),"mlu_worst":max([initial.metrics.maximum_link_utilization]+[r["metrics"]["maximum_link_utilization"] for r in records]),"traffic_shift_step_project_final":final.metrics.traffic_shift_step_project_v1,"traffic_shift_total_project_final":final.metrics.traffic_shift_total_project_v1,"traffic_shift_step_paper_final":final.metrics.traffic_shift_step_paper_v1,"traffic_shift_total_paper_final":final.metrics.traffic_shift_total_paper_v1,"configuration_change_count":change["count"],"configuration_change_ratio":change["ratio"],"configuration_change_fields":change["changed_fields"],"action_count":actions,"success":success,"convergence_step":convergence,"censored":not success,"decision_time_ms":sum(r["decision_time_ms"] for r in records),"simulator_time_ms":sum(r["simulator_time_ms"] for r in records),"wall_time_ms":(perf_counter_ns()-started)/1e6}
        summary["run_key"]=run_key(method.metadata.__dict__, scenario.scenario_id, sequence_id, seed, self.evaluator_config_hash)
        summary["lookahead_candidate_evaluations"]=sum(int(r["method_diagnostics"].get("candidate_evaluations",0)) for r in records)
        summary["lookahead_simulator_calls"]=sum(int(r["method_diagnostics"].get("simulator_calls",0)) for r in records)
        return RunOutcome(summary, tuple(records))
    def persist(self, store: ResultStore, outcome: RunOutcome) -> bool:
        """Commit a run transactionally so interrupted resumes cannot mix trajectories."""
        key = str(outcome.summary["run_key"])
        steps=[]
        for index, row in enumerate(outcome.steps):
            item=dict(row); item["run_key"]=f"{key}:{index}"; steps.append(item)
        events=[]
        for event in outcome.event_recovery:
            item=dict(event); item["run_key"]=key; events.append(item)
        target="episodes.jsonl" if outcome.summary.get("status") in {"completed", "terminated", "truncated"} else "failures.jsonl"
        return store.commit_run(run_key=key, summary=outcome.summary, steps=steps, events=events, terminal_file=target)
    def run_static_dataset(self, method: EvaluationMethod, dataset: ScenarioDataset, *, seed: int, scenario_id: str, run_id: str = "run") -> RunOutcome:
        """Materialize exactly the requested lazy dataset record, not the whole split."""
        scenario = next((item for item in dataset if item.scenario_id == scenario_id), None)
        if scenario is None: raise KeyError(f"unknown_scenario_id:{scenario_id}")
        return self.run_static(method, scenario, seed=seed, run_id=run_id)
    def run_dynamic(self, method, scenario, *, seed:int, sequence_id:str, recovery_budget_steps:int=30, run_id:str="run", logical_events: tuple[Mapping[str, Any], ...] | None = None, show_step_progress: bool = False) -> RunOutcome:
        outcome=self.run_static(method, scenario, seed=seed, run_id=run_id, sequence_id=sequence_id, recovery_budget_steps=recovery_budget_steps, run_full_horizon=True, show_step_progress=show_step_progress)
        events=[]; by_id={event.event_id:event for event in scenario.events}
        units = logical_events or tuple({"kind": event.kind, "event_ids": [event.event_id]} for event in scenario.events)
        # Events take effect at the result of the step whose pre-snapshot has the matching step.
        for unit in units:
            members=[by_id[item] for item in unit["event_ids"] if item in by_id]
            if not members: continue
            event_step=min(event.step for event in members)
            recovery_start=max(event.step for event in members)
            affected=[r for r in outcome.steps if r["step"] == event_step]
            if not affected: continue
            before = affected[0]  # metrics before event are reconstructed from the prior record/initial below
            pre_metrics = outcome.summary if event_step == 0 else next((r["metrics"] for r in outcome.steps if r["step"] == event_step-1), outcome.summary)
            recovery_window=[r for r in outcome.steps if recovery_start <= r["step"] < recovery_start+recovery_budget_steps]
            impact_window=[r for r in outcome.steps if event_step <= r["step"] < recovery_start+recovery_budget_steps]
            pcs=[r["metrics"]["policy_consistency"] for r in impact_window]; mlus=[r["metrics"]["maximum_link_utilization"] for r in impact_window]
            pre_pc=float(outcome.summary["policy_consistency_initial"] if event_step == 0 else pre_metrics.get("policy_consistency", outcome.summary["policy_consistency_initial"]))
            pre_mlu=float(outcome.summary["mlu_initial"] if event_step == 0 else pre_metrics.get("maximum_link_utilization", outcome.summary["mlu_initial"]))
            recovered_index=None; consecutive=0
            for offset, record in enumerate(recovery_window):
                qualifies=record["metrics"]["policy_consistency"] >= pre_pc and record["metrics"]["maximum_link_utilization"] <= pre_mlu
                consecutive=consecutive+1 if qualifies else 0
                if consecutive >= self.hold_steps:
                    recovered_index=offset; break
            recovered_record=recovery_window[recovered_index] if recovered_index is not None else None
            shifts={name:[float(r["metrics"].get(name) or 0.0) for r in impact_window] for name in ("traffic_shift_step_project_v1","traffic_shift_total_project_v1","traffic_shift_step_paper_v1","traffic_shift_total_paper_v1")}
            events.append({"event_id":"|".join(unit["event_ids"]),"event_type":unit["kind"],"event_step":event_step,"recovery_start_step":recovery_start,"recovery_budget_steps":recovery_budget_steps,"hold_steps":self.hold_steps,"pre_event_baseline":{"policy_consistency":pre_pc,"mlu":pre_mlu},"worst":{"policy_consistency":min(pcs) if pcs else None,"mlu":max(mlus) if mlus else None},"recovery_value":recovered_record["metrics"] if recovered_record else None,"recovered":recovered_record is not None,"recovery_steps":recovered_index+1 if recovered_index is not None else None,"censored":recovered_record is None,"traffic_shift":{name:{"peak":max(values) if values else None,"mean":sum(values)/len(values) if values else None,"total":sum(values) if values else None} for name,values in shifts.items()},"configuration_change":{"peak_ratio":max((r["configuration_change_from_initial_ratio"] for r in impact_window),default=None),"recovered_ratio":recovered_record["configuration_change_from_initial_ratio"] if recovered_record else None,"action_field_changes":sum(len(r["configuration_diff"]) for r in impact_window)},"runtime_ms":sum(r["wall_time_ms"] for r in impact_window)})
        summary=dict(outcome.summary); summary["dynamic_recovered"]=all(x["recovered"] for x in events) if events else True; summary["dynamic_recovery_count"]=sum(x["recovered"] for x in events); summary["success"]=summary["dynamic_recovered"] and len(outcome.steps)==scenario.max_steps; summary["censored"]=not summary["success"]
        summary["event_recovery"]=events
        return RunOutcome(summary, outcome.steps, tuple(events))
    def run_dynamic_sequence(self, method, dataset_root, sequence: Mapping[str, Any], *, seed: int, run_id: str = "run", show_step_progress: bool = False) -> RunOutcome:
        from netkeeper_sim.dataset.dynamic_sequences import dynamic_scenario
        scenario=dynamic_scenario(dataset_root, sequence)
        return self.run_dynamic(method, scenario, seed=seed, sequence_id=str(sequence["sequence_id"]), recovery_budget_steps=int(sequence["recovery_budget_steps"]), run_id=run_id, logical_events=tuple(sequence["logical_events"]), show_step_progress=show_step_progress)
    def _failed(self, method, context, initial, records, started, code, exc):
        summary={"evaluator_version":EVALUATOR_VERSION,"run_id":context.run_id,"method_name":method.metadata.name,"method_version":method.metadata.version,"method_metadata":method.metadata.__dict__,"scenario_id":context.scenario_id,"sequence_id":context.sequence_id,"seed":context.seed,"status":"failed","failure_code":code,"failure_detail":str(exc),"initial_snapshot_id":initial.snapshot_id,"final_snapshot_id":initial.snapshot_id,"policy_consistency_initial":initial.metrics.policy_consistency,"policy_consistency_final":initial.metrics.policy_consistency,"mlu_initial":initial.metrics.maximum_link_utilization,"mlu_final":initial.metrics.maximum_link_utilization,"configuration_change_count":0,"configuration_change_ratio":0.0,"action_count":0,"success":False,"convergence_step":None,"censored":True,"decision_time_ms":0.0,"simulator_time_ms":0.0,"wall_time_ms":(perf_counter_ns()-started)/1e6}
        summary["run_key"]=run_key(method.metadata.__dict__, context.scenario_id, context.sequence_id, context.seed, self.evaluator_config_hash)
        return RunOutcome(summary, records)
