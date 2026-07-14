"""Frozen Block-6 baselines, all expressed as unified schema JointActions."""
from __future__ import annotations

import random
from dataclasses import replace
from time import perf_counter_ns
from typing import Any

from netkeeper_sim.evaluation.methods import EvaluationContext, MethodDecision, MethodMetadata, canonical_hash
from netkeeper_sim.rl.schema_adapter import action_masks, candidate_to_joint_action, snapshot_to_graph
from netkeeper_sim.schemas import AtomicAction, JointAction
from netkeeper_sim.simulator import UnifiedNetworkEnvironment

OSPF_DEFAULT_WEIGHT = 1
LOCAL_SEARCH_DELTAS = (1, 2, 4, 8)
LOCAL_SEARCH_CANDIDATE_BUDGET = 64

ALL_RL_PARAMETERS = ("ospf_weight", "local_preference", "as_path_length", "med", "bandwidth_bps", "capacity_bps", "queue_packets")

class NoUpdateMethod:
    metadata = MethodMetadata("no_update", "baseline-v1", canonical_hash({}), None, True, (), False)
    def reset(self, context: EvaluationContext) -> None: pass
    def act(self, snapshot, observation, context) -> MethodDecision:
        started=perf_counter_ns()
        return MethodDecision(JointAction((), requested_by="no_update", snapshot_id=snapshot.snapshot_id), perf_counter_ns()-started)

class RandomMethod:
    """Uniform over the exact Block-5 candidate masks, independently per agent."""
    metadata = MethodMetadata("random", "baseline-v1", canonical_hash({"sampling":"uniform_valid_candidates_including_no_update"}), None, False, ALL_RL_PARAMETERS, False)
    def reset(self, context: EvaluationContext) -> None:
        self.rng=random.Random(int(canonical_hash({"seed":context.seed,"scenario":context.scenario_id,"sequence":context.sequence_id,"method":self.metadata.identity})[:16],16))
    def act(self, snapshot, observation, context) -> MethodDecision:
        started=perf_counter_ns(); graph=snapshot_to_graph(snapshot); masks=action_masks(snapshot,graph)
        choices={}
        for agent in ("ospf","bgp","performance"):
            valid=[int(x) for x in masks[agent].nonzero(as_tuple=False).reshape(-1).tolist()]
            choices[agent]=self.rng.choice(valid) if valid else 0
        action=candidate_to_joint_action(snapshot,graph,choices)
        return MethodDecision(action,perf_counter_ns()-started,{"candidates":{key:int(mask.sum()) for key,mask in masks.items()},"choices":choices})

class OSPFDefaultMethod:
    """One legal OSPF set per step in stable link order; never adapts after reset."""
    metadata = MethodMetadata("ospf_default", "baseline-v1", canonical_hash({"default_weight":OSPF_DEFAULT_WEIGHT,"submission":"one_sorted_link_per_step"}), None, True, ("ospf_weight",), False)
    def reset(self, context: EvaluationContext) -> None: self.defaulted=set()
    def act(self, snapshot, observation, context) -> MethodDecision:
        started=perf_counter_ns(); graph=snapshot_to_graph(snapshot); masks=action_masks(snapshot,graph)
        # Mask semantics are the common action legality source.  A down link
        # is deferred rather than force-written; this method never responds to
        # a later fault by changing a weight already made default.
        legal_links=[]
        for index, link_id in enumerate(graph.link_ids):
            first=1 + index * 64
            if bool(masks["ospf"][first:first+64].any()): legal_links.append(link_id)
        candidates=[link for link in sorted(legal_links) if link not in self.defaulted and snapshot.configuration.ospf_weights[link] != OSPF_DEFAULT_WEIGHT]
        if not candidates:
            return MethodDecision(JointAction((),requested_by="ospf_default",snapshot_id=snapshot.snapshot_id),perf_counter_ns()-started)
        link_id=candidates[0]; self.defaulted.add(link_id)
        return MethodDecision(JointAction((AtomicAction("ospf","ospf_weight",{"link_id":link_id},"set",OSPF_DEFAULT_WEIGHT),),requested_by="ospf_default",snapshot_id=snapshot.snapshot_id),perf_counter_ns()-started,{"default_weight":OSPF_DEFAULT_WEIGHT,"submission_rule":"one_sorted_link_per_step"})

class LocalSearchOSPFMethod:
    """Strict one-step lexicographic OSPF hill climb with isolated replays."""
    metadata = MethodMetadata("local_search_ospf", "baseline-v1", canonical_hash({"deltas":LOCAL_SEARCH_DELTAS,"budget":LOCAL_SEARCH_CANDIDATE_BUDGET,"objective":"max_pc,min_mlu,min_project_shift,min_changed_fields"}), None, True, ("ospf_weight",), True)
    def __init__(self, *, candidate_budget: int = LOCAL_SEARCH_CANDIDATE_BUDGET, deltas: tuple[int,...] = LOCAL_SEARCH_DELTAS) -> None:
        if candidate_budget <= 0: raise ValueError("candidate_budget must be positive")
        self.candidate_budget,self.deltas=candidate_budget,tuple(deltas)
        self.metadata=replace(self.metadata,config_hash=canonical_hash({"deltas":self.deltas,"budget":candidate_budget,"objective":"max_pc,min_mlu,min_project_shift,min_changed_fields"}))
    def reset(self, context: EvaluationContext) -> None: pass
    def _sandbox(self, context, snapshot, action):
        sandbox=UnifiedNetworkEnvironment(); sandbox.reset(context.scenario,seed=context.seed)
        return sandbox.step(snapshot,action)
    @staticmethod
    def _objective(result):
        metric=result.metrics
        return (-metric.policy_consistency,metric.maximum_link_utilization,metric.traffic_shift_step_project_v1 or 0.0,len(result.changed_config))
    def act(self,snapshot,observation,context) -> MethodDecision:
        started=perf_counter_ns(); graph=snapshot_to_graph(snapshot); masks=action_masks(snapshot,graph)
        legal=[]
        for entity,link_id in enumerate(graph.link_ids):
            first=1+entity*64
            if not bool(masks["ospf"][first:first+64].any()): continue
            current=int(snapshot.configuration.ospf_weights[link_id])
            for delta in self.deltas:
                for value in (current-delta,current+delta):
                    if 1 <= value <= 64 and value != current:
                        legal.append((link_id,value))
        unique_candidates=sorted(set(legal)); candidates=unique_candidates[:self.candidate_budget]
        noop=JointAction((),requested_by="local_search_ospf",snapshot_id=snapshot.snapshot_id)
        before=self._sandbox(context,snapshot,noop); best_result,best_action,best_key=before,noop,self._objective(before)
        evaluated=1
        for link_id,value in candidates:
            action=JointAction((AtomicAction("ospf","ospf_weight",{"link_id":link_id},"set",value),),requested_by="local_search_ospf",snapshot_id=snapshot.snapshot_id)
            result=self._sandbox(context,snapshot,action); evaluated+=1; key=self._objective(result)
            if key < best_key: best_result,best_action,best_key=result,action,key
        accepted=best_action.actions[0].to_dict() if best_action.actions else None
        return MethodDecision(best_action,perf_counter_ns()-started,{"candidate_total":len(unique_candidates),"candidate_evaluations":evaluated,"candidate_budget":self.candidate_budget,"simulator_calls":evaluated,"accepted_candidate":accepted,"objective_before":self._objective(before),"objective_after":best_key})
