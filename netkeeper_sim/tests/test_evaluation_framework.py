from __future__ import annotations
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest
from netkeeper_sim.evaluation import EvaluationRunner, LocalSearchOSPFMethod, MethodDecision, MethodMetadata, NoUpdateMethod, OSPFDefaultMethod, RandomMethod, ResultStore, aggregate_runs, configuration_change, generate_evaluation_manifest
from netkeeper_sim.evaluation.results import EVALUATOR_VERSION, run_key
from netkeeper_sim.evaluation.methods import DispatcherMethodAdapter, canonical_hash
from netkeeper_sim.evaluation.batch import plan
from netkeeper_sim.schemas import AtomicAction, BGPConfiguration, BGPRoute, JointAction, Link, LinkAttributes, NetworkConfiguration, NetworkScenario, Node, Policy, Topology, TrafficDemand, TrafficMatrix

class NoUpdate:
    metadata=MethodMetadata("no_update", "1", "cfg", None, True, (), False)
    def reset(self, context): self.context=context
    def act(self, snapshot, observation, context): return JointAction((), snapshot_id=snapshot.snapshot_id)
class OneOspf:
    metadata=MethodMetadata("one_ospf", "1", "cfg", None, True, ("ospf_weight",), False)
    def reset(self, context): self.used=False
    def act(self, snapshot, observation, context):
        if self.used:return JointAction((), snapshot_id=snapshot.snapshot_id)
        self.used=True; return MethodDecision(JointAction((AtomicAction("ospf","ospf_weight",{"link_id":"L0"},"set",2),), snapshot_id=snapshot.snapshot_id))
class BadMethod:
    metadata=MethodMetadata("bad", "1", "cfg", None, True, (), False)
    def reset(self, context): pass
    def act(self, snapshot, observation, context): return JointAction((AtomicAction("ospf","ospf_weight",{"link_id":"L0"},"set",2),), snapshot_id=snapshot.snapshot_id)

def toy(*, events=(), max_steps=4):
    links=(Link("L0","R0","R1",0,LinkAttributes(physical_bandwidth_bps=10,bandwidth_bps=10,capacity_max_bps=10,capacity_bps=10)), Link("L1","R1","R2",0,LinkAttributes(physical_bandwidth_bps=10,bandwidth_bps=10,capacity_max_bps=10,capacity_bps=10)))
    topology=Topology("T:test","test","synthetic","x",(Node("R0","0"),Node("R1","1"),Node("R2","2")),links)
    traffic=TrafficMatrix("TM",("R0","R1","R2"),(TrafficDemand("R0",1,destination="R2"),))
    return NetworkScenario("S:test",topology,traffic,(Policy("P","reachable",{"source":"R0","destination":"R2"}),),events=events,max_steps=max_steps)

def test_configuration_field_ratio_all_agent_types_and_restore():
    s=toy(); base=NetworkConfiguration.initial(s.topology)
    bgp=BGPConfiguration((BGPRoute("R0","192.0.2.0/24","R1",100,(64512,),0),))
    changed=base.with_updates(ospf_weights={"L0":2}, performance={"L1":LinkAttributes(physical_bandwidth_bps=10,bandwidth_bps=9,capacity_max_bps=10,capacity_bps=9)}, bgp=bgp)
    result=configuration_change(base,changed)
    assert result["count"] == 6 and result["denominator"] == 18
    assert configuration_change(base,base)["count"] == 0
    restored=changed.with_updates(ospf_weights={"L0":1}, performance={"L1":base.performance["L1"]}, bgp=base.bgp)
    assert configuration_change(base,restored)["count"] == 0

def test_static_runner_metrics_runtime_success_and_snapshot_immutability():
    scenario=toy(max_steps=2); outcome=EvaluationRunner(hold_steps=9).run_static(OneOspf(),scenario,seed=2)
    assert outcome.summary["status"] == "truncated" and outcome.summary["action_count"] == 1
    assert outcome.summary["configuration_change_count"] == 1
    assert outcome.summary["wall_time_ms"] >= outcome.summary["simulator_time_ms"] >= 0
    assert all(x["decision_time_ms"] >= 0 and x["before_snapshot_id"] != x["after_snapshot_id"] for x in outcome.steps)
    assert scenario.configuration is None
    from dataclasses import replace
    terminated=EvaluationRunner(hold_steps=9).run_static(NoUpdate(),replace(scenario,target_mlu=1.0),seed=2)
    assert terminated.summary["status"] == "truncated" and terminated.summary["success"] is False
    assert EvaluationRunner().run_static(BadMethod(),scenario,seed=2).summary["status"] == "failed"

def test_static_termination_does_not_bypass_three_state_hold():
    from dataclasses import replace
    scenario=replace(toy(max_steps=4),target_mlu=1.0)
    outcome=EvaluationRunner(hold_steps=3).run_static(NoUpdate(),scenario,seed=2)
    assert outcome.summary["success"] is True
    assert outcome.summary["convergence_step"] == 3
    assert len(outcome.steps) == 3

def test_dynamic_event_recovery_and_no_preconsume():
    from netkeeper_sim.schemas import Event
    scenario=toy(events=(Event("E0",1,"traffic_scale",{"factor":2}),),max_steps=4)
    outcome=EvaluationRunner(hold_steps=9).run_dynamic(NoUpdate(),scenario,seed=3,sequence_id="DS",recovery_budget_steps=2)
    event=outcome.event_recovery[0]
    assert event["event_step"] == 1 and event["recovered"] is False and event["recovery_steps"] is None
    assert outcome.steps[0]["metrics"]["total_input_bps"] == 1
    assert outcome.steps[1]["metrics"]["total_input_bps"] == 2

def test_dynamic_consumes_full_horizon_and_pairs_fault_recovery_window():
    from netkeeper_sim.schemas import Event
    scenario=toy(events=(Event("down",1,"link_down",target_id="L0"),Event("up",3,"link_up",target_id="L0")),max_steps=6)
    logical=({"kind":"link_failure_recovery","event_ids":["down","up"]},)
    outcome=EvaluationRunner(hold_steps=3).run_dynamic(NoUpdate(),scenario,seed=3,sequence_id="DS",recovery_budget_steps=3,logical_events=logical)
    assert len(outcome.steps) == 6
    event=outcome.event_recovery[0]
    assert event["event_step"] == 1 and event["recovery_start_step"] == 3
    assert event["recovered"] is True and event["recovery_steps"] == 3
    assert event["worst"]["policy_consistency"] < event["pre_event_baseline"]["policy_consistency"]

def test_store_nan_rejection_resume_and_stats(tmp_path):
    store=ResultStore(tmp_path); row={"run_key":"a","x":float("inf")}
    assert store.append_unique("episodes.jsonl",row,key="a") and not store.append_unique("episodes.jsonl",row,key="a")
    assert store.read("episodes.jsonl")[0]["x"] == {"non_finite":"positive_infinity"}
    rows=[{"method_name":"m","status":"completed","success":True,"censored":False,"policy_consistency_final":1.,"mlu_final":.5,"configuration_change_ratio":.1,"wall_time_ms":1.},{"method_name":"m","status":"failed","success":False,"censored":True}]
    agg=aggregate_runs(rows)[0]; assert agg["attempted"]==2 and agg["failed"]==1 and agg["policy_consistency_final"]["std"] is None
    rows.append({"method_name":"m","status":"completed","success":True,"censored":False,"policy_consistency_final":.5,"mlu_final":.4,"configuration_change_ratio":.2,"wall_time_ms":2.})
    assert aggregate_runs(rows)[0]["policy_consistency_final"]["std"] is not None
    assert run_key({"config_hash":"a"},"S",None,1,"x") != run_key({"config_hash":"b"},"S",None,1,"x")
    persisted=EvaluationRunner(hold_steps=9).run_static(NoUpdate(),toy(max_steps=1),seed=1)
    assert EvaluationRunner().persist(store,persisted) and not EvaluationRunner().persist(store,persisted)

def test_run_commit_replaces_interrupted_rows_and_serializes_workers(tmp_path):
    outcome=EvaluationRunner(hold_steps=9,evaluator_config_hash="cfg").run_static(NoUpdate(),toy(max_steps=3),seed=1)
    key=outcome.summary["run_key"]
    store=ResultStore(tmp_path)
    # Simulate a killed old worker which wrote one corrupt pre-terminal row.
    store.append_unique("steps.jsonl",{"run_key":f"{key}:0","step":9,"before_snapshot_id":"bad","after_snapshot_id":"bad"},key=f"{key}:0")
    with ThreadPoolExecutor(max_workers=2) as pool:
        committed=list(pool.map(lambda _:EvaluationRunner().persist(store,outcome),range(2)))
    assert sorted(committed)==[False,True]
    rows=store.read("steps.jsonl")
    assert [row["step"] for row in rows]==[0,1,2]
    assert all(row["run_key"]==f"{key}:{index}" for index,row in enumerate(rows))
    assert len(store.read("episodes.jsonl"))==1

def test_manifest_and_missing_checkpoint_are_stable():
    root=Path(__file__).resolve().parents[2]/"data"/"netkeeper_lite"
    first=generate_evaluation_manifest(root,seeds=(1,2)); second=generate_evaluation_manifest(root,seeds=(1,2))
    assert len(first["static"])==500 and len(first["dynamic"])==100 and first["manifest_hash"]==second["manifest_hash"]
    from netkeeper_sim.simulator import UnifiedNetworkEnvironment
    snap, observation=UnifiedNetworkEnvironment().reset(toy())
    method=DispatcherMethodAdapter("/not/a/checkpoint.pt"); decision=method.act(snap,observation,None)
    assert decision.status=="unavailable" and decision.failure_code=="checkpoint_unavailable"
    dispatched=EvaluationRunner().run_static(method,toy(max_steps=1),seed=3)
    assert dispatched.summary["status"]=="unavailable" and dispatched.summary["failure_code"]=="checkpoint_unavailable"

def test_checkpoint_method_identity_excludes_evaluation_plan_fields():
    first=DispatcherMethodAdapter("/not/a/checkpoint.pt",config={"mode":"static","max_steps":50})
    second=DispatcherMethodAdapter("/not/a/checkpoint.pt",config={"mode":"dynamic","max_steps":240})
    assert first.metadata.version == "legacy-inference-adapter"
    assert first.metadata.config_hash == second.metadata.config_hash

def test_formal_manifest_load_groups_and_method_specific_seeds():
    root=Path(__file__).resolve().parents[2]/"data"/"netkeeper_lite"
    manifest=generate_evaluation_manifest(root)
    groups=manifest["analysis_groups"]["load"]["scenario_ids"]
    assert {key:len(value) for key,value in groups.items()} == {"Normal":166,"Hotspot":125,"Burst":42,"High-load":167}
    assert len({item for values in groups.values() for item in values}) == 500
    config={"methods":["no_update","random"],"deterministic_seed":20260714,"random_seeds":[20260714,20260715,20260716],"mode":"static","max_steps":50,"hold_steps":3,"local_search_budget":64,"local_search_deltas":[1,2,4,8],"checkpoint":None,"checkpoint_status":"debug_unconverged","scenario_ids":[],"sequence_ids":[],"device":"cpu","recovery_budget":30}
    run,tasks=plan(root,config)
    assert len(tasks) == 2000
    assert run["evaluator_version"] == EVALUATOR_VERSION
    assert run["evaluator_config_hash"] == canonical_hash({"evaluator_version":EVALUATOR_VERSION,"config":config})
    assert {x["seed"] for x in tasks if x["method"]=="no_update"} == {20260714}
    assert {x["seed"] for x in tasks if x["method"]=="random"} == {20260714,20260715,20260716}

def test_no_update_random_and_default_permissions_and_replay():
    from netkeeper_sim.schemas import Event
    from dataclasses import replace
    dynamic=toy(events=(Event("E",0,"traffic_scale",{"factor":2}),),max_steps=2)
    no=EvaluationRunner(hold_steps=9).run_dynamic(NoUpdateMethod(),dynamic,seed=4,sequence_id="D",recovery_budget_steps=2)
    assert no.summary["action_count"]==0 and no.summary["configuration_change_count"]==0
    assert no.steps[0]["metrics"]["total_input_bps"]==2
    first=EvaluationRunner(hold_steps=9).run_static(RandomMethod(),toy(max_steps=2),seed=7)
    second=EvaluationRunner(hold_steps=9).run_static(RandomMethod(),toy(max_steps=2),seed=7)
    other=EvaluationRunner(hold_steps=9).run_static(RandomMethod(),toy(max_steps=2),seed=8)
    assert [x["action"] for x in first.steps]==[x["action"] for x in second.steps]
    assert [x["action"] for x in first.steps] != [x["action"] for x in other.steps]
    base=NetworkConfiguration.initial(toy().topology).with_updates(ospf_weights={"L0":5,"L1":6})
    default=EvaluationRunner(hold_steps=9).run_static(OSPFDefaultMethod(),replace(toy(max_steps=3),configuration=base),seed=1)
    assert default.summary["action_count"]==2 and default.summary["configuration_change_count"]==2
    assert all("ospf_weight" in [a["parameter_type"] for a in row["action"]["actions"]] or not row["action"]["actions"] for row in default.steps)

def test_local_search_hand_calculated_choice_and_sandbox_leakage():
    links=(Link("L0","R0","R1",0,LinkAttributes(physical_bandwidth_bps=10,bandwidth_bps=10,capacity_max_bps=10,capacity_bps=10)),Link("L1","R1","R3",0,LinkAttributes(physical_bandwidth_bps=10,bandwidth_bps=10,capacity_max_bps=10,capacity_bps=10)),Link("L2","R0","R2",0,LinkAttributes(physical_bandwidth_bps=10,bandwidth_bps=10,capacity_max_bps=10,capacity_bps=10,ospf_weight=2)),Link("L3","R2","R3",0,LinkAttributes(physical_bandwidth_bps=10,bandwidth_bps=10,capacity_max_bps=10,capacity_bps=10)))
    topology=Topology("T:diamond","diamond","synthetic","x",(Node("R0","0"),Node("R1","1"),Node("R2","2"),Node("R3","3")),links)
    scenario=NetworkScenario("S:diamond",topology,TrafficMatrix("TM",("R0","R1","R2","R3"),(TrafficDemand("R0",1,destination="R3"),)),(Policy("P","forward_pass",{"source":"R0","destination":"R3","waypoint":"R2"}),),max_steps=1)
    method=LocalSearchOSPFMethod(candidate_budget=64)
    result=EvaluationRunner(hold_steps=9).run_static(method,scenario,seed=1)
    action=result.steps[0]["action"]["actions"][0]
    assert action["target"]["link_id"]=="L0" and action["value"]==3
    assert result.steps[0]["method_diagnostics"]["simulator_calls"] > 1
    # The sandbox did not mutate the scenario or its initial snapshot; a fresh
    # replay makes the identical action and metric transition.
    replay=EvaluationRunner(hold_steps=9).run_static(LocalSearchOSPFMethod(candidate_budget=64),scenario,seed=1)
    assert replay.steps[0]["before_snapshot_id"]==result.steps[0]["before_snapshot_id"]
    assert replay.steps[0]["action"]==result.steps[0]["action"]
