"""Reproducible bounded training runner for the unified COMA environment."""
from __future__ import annotations
import csv, json, os, random, time, uuid, struct, zlib, hashlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
import numpy as np
import torch, yaml
from netkeeper_sim.rl.config import GraphNetworkConfig
from netkeeper_sim.rl.multi_agent_env import UnifiedRLEnvironment
from netkeeper_sim.rl.trainer import COMATrainer
from netkeeper_sim.rl.schema_adapter import action_masks, candidate_to_joint_action, masked_policy, monotonic_candidate_filter, snapshot_to_graph
from netkeeper_sim.rl.networks.multi_agent_actor import AGENTS
from netkeeper_sim.schemas.ids import SCHEMA_VERSION
from netkeeper_sim.simulator import UnifiedNetworkEnvironment

MODEL_VERSION = "rl-coma-v3"
TRAINING_SEMANTICS_VERSION = "coma-counterfactual-v4"
CHECKPOINT_STATE_VERSION = "training-state-v2"
CHECKPOINT_CANDIDATE = "validation_selected_candidate"

def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

@dataclass(frozen=True)
class TrainingConfig:
    episodes:int; max_steps:int; batch_size:int; warmup:int; updates_per_step:int
    actor_lr:float; critic_lr:float; gamma:float; target_interval:int; entropy_coef:float
    epsilon_start:float; epsilon_end:float; epsilon_decay:float; temperature:float
    validation_every:int; validation_scenarios:int; patience:int; seed:int; replay_capacity:int
    hidden_dim:int; gnn_layers:int; transformer_layers:int; heads:int; dropout:float=.1; gradient_clip:float=1.; amp:bool=True; device:str="auto"
    static_patience:int=5; improvement_tolerance:float=1e-6
    formal_min_validation_delta:float=0.; formal_min_agent_actions:int=1
    counterfactual_replay:bool=True
    counterfactual_candidate_samples:int=8
    monotonic_shield:bool=True
    @classmethod
    def load(cls,path:str|Path)->"TrainingConfig": return cls(**yaml.safe_load(Path(path).read_text())["training"])
    def graph(self): return GraphNetworkConfig(18,11,self.hidden_dim,self.gnn_layers,self.transformer_layers,self.heads,self.dropout)

def seed_all(seed:int): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
def rng_state(): return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
def set_rng_state(state): random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"]); torch.cuda.set_rng_state_all(state["cuda"]) if state.get("cuda") is not None and torch.cuda.is_available() else None
def aggregate_trajectory(results):
    """Shared train/validation statistic: totals, per-step means and breakdown."""
    names=("policy_reward","mlu_reward","traffic_shift_reward","configuration_change_penalty","illegal_action_penalty","dropped_traffic_penalty","total_reward")
    steps=len(results); breakdown={name:sum(getattr(item.rewards,name) for item in results) for name in names}
    return {"reward":breakdown["total_reward"],"mean_reward":breakdown["total_reward"]/max(steps,1),"steps":steps,"breakdown_total":breakdown,"breakdown_mean":{key:value/max(steps,1) for key,value in breakdown.items()}}

class TrainingRunner:
    def __init__(self, root:str|Path, config:TrainingConfig, output:str|Path, *, resume:str|Path|None=None) -> None:
        self.root=Path(root); self.cfg=config; seed_all(config.seed)
        self.output=Path(output); self.output.mkdir(parents=True,exist_ok=False) if not resume else self.output.mkdir(parents=True,exist_ok=True)
        self.env=UnifiedRLEnvironment(dataset_root=self.root,split="train",seed=config.seed)
        self.trainer=COMATrainer(self.env,config.graph(),gamma=config.gamma,target_interval=config.target_interval,amp=config.amp,seed=config.seed,device=config.device,entropy_coef=config.entropy_coef,gradient_clip=config.gradient_clip)
        self.trainer.actor_optimizer.param_groups[0]["lr"]=config.actor_lr; self.trainer.critic_optimizer.param_groups[0]["lr"]=config.critic_lr
        from netkeeper_sim.rl.replay_buffer import ReplayBuffer
        self.trainer.buffer=ReplayBuffer(config.replay_capacity,config.seed)
        self.episode=0; self.global_step=0; self.best_validation=float("-inf"); self.best_validation_episode=None; self.bad_validations=0; self.epsilon=config.epsilon_start; self.records=[]; self.validation_summary=None
        (self.output/"resolved_config.yaml").write_text(yaml.safe_dump(asdict(config),sort_keys=True));
        if resume: self.load(resume)

    def actions(self, graph,masks,training=True):
        self.trainer.actor.train(training)
        with torch.no_grad(): out=self.trainer.actor.forward_graph(graph,masks)
        result={}
        for agent in out.logits:
            logits=out.logits[agent]/max(self.cfg.temperature,1e-6); p,_=masked_policy(logits,masks[agent])
            if training and random.random()<self.epsilon:
                valid=torch.nonzero(masks[agent],as_tuple=False).reshape(-1); result[agent]=int(valid[random.randrange(valid.numel())])
            else: result[agent]=int(torch.multinomial(p,1).item() if training else p.argmax(-1).item())
        return result

    def run_episode(self, training=True, *, force_noop=False, step_limit=None, allow_early_stop=True):
        graph,_,masks=self.env.reset(seed=self.cfg.seed+self.episode if training else self.cfg.seed)
        transitions=[]; losses=[]; started=time.perf_counter(); result=None; stagnant_steps=0; stop_reason=None
        action_counts={agent:0 for agent in AGENTS}; no_op_counts={agent:0 for agent in AGENTS}; unique_actions={agent:set() for agent in AGENTS}; agent_rewards={agent:0.0 for agent in AGENTS}
        limit=self.cfg.max_steps if step_limit is None else min(self.cfg.max_steps,int(step_limit))
        for step in range(limit):
            before_snapshot=self.env.snapshot
            actions={agent:0 for agent in AGENTS} if force_noop else self.actions(graph,masks,training)
            if not force_noop and self.cfg.monotonic_shield and before_snapshot is not None:
                actions=monotonic_candidate_filter(before_snapshot,graph,actions,reward_config=self.env.environment.reward_config)
            for agent,candidate in actions.items():
                if candidate == 0: no_op_counts[agent]+=1
                else: action_counts[agent]+=1; unique_actions[agent].add(candidate)
            result=self.env.step(actions); transitions.append(result.result)
            for agent,value in result.result.rewards.per_agent.items(): agent_rewards[agent]+=float(value)
            if training:
                from netkeeper_sim.rl.replay_buffer import Transition
                counterfactuals=self.counterfactual_transitions(before_snapshot,graph,masks,actions) if self.cfg.counterfactual_replay and before_snapshot is not None else []
                paired={agent:item.reward[agent] for agent,item in counterfactuals}
                advantages={agent:float(result.result.rewards.per_agent[agent])-float(paired[agent]) for agent in paired}
                self.trainer.buffer.add(Transition(graph,masks,actions,dict(result.result.rewards.per_agent),result.graph,result.masks,result.result.terminated,result.result.truncated,graph.snapshot_id,result.graph.snapshot_id,advantages))
                if self.cfg.counterfactual_replay and before_snapshot is not None:
                    for agent,item in counterfactuals:
                        reverse=float(item.reward[agent])-float(result.result.rewards.per_agent[agent])
                        self.trainer.buffer.add(replace(item,advantages={agent:reverse},actor_weight=1.0))
                if len(self.trainer.buffer)>=max(self.cfg.warmup,self.cfg.batch_size):
                    for _ in range(self.cfg.updates_per_step): losses.append(self.trainer.update(self.trainer.buffer.sample(self.cfg.batch_size)))
                self.global_step+=1
            graph,masks=result.graph,result.masks
            if result.result.terminated or result.result.truncated: break
            objective_gain=result.result.rewards.policy_reward+result.result.rewards.mlu_reward
            stagnant_steps=stagnant_steps+1 if objective_gain <= self.cfg.improvement_tolerance else 0
            scenario=self.env.environment.scenario
            if allow_early_stop and scenario is not None and not scenario.events and stagnant_steps>=self.cfg.static_patience:
                stop_reason="static_no_improvement"; break
        assert result is not None
        metrics=result.result.metrics; aggregate=aggregate_trajectory(transitions); steps=aggregate["steps"]
        initial=self.env.environment.initial_snapshot
        cumulative_changes=len(initial.configuration.diff(result.result.next_snapshot.configuration)) if initial is not None else len(result.result.changed_config)
        diagnostics={agent:{"action_count":action_counts[agent],"no_op_count":no_op_counts[agent],"action_rate":action_counts[agent]/max(steps,1),"unique_action_count":len(unique_actions[agent]),"reward_total":agent_rewards[agent],"reward_mean":agent_rewards[agent]/max(steps,1)} for agent in AGENTS}
        return {**aggregate,"policy_consistency":metrics.policy_consistency,"mlu":metrics.maximum_link_utilization,"traffic_shift":metrics.traffic_shift_step_project_v1 or 0.,"config_changes":cumulative_changes,"terminated":result.result.terminated,"truncated":result.result.truncated,"early_stop":stop_reason is not None,"stop_reason":stop_reason,"agent_diagnostics":diagnostics,"breakdown":result.result.rewards.to_dict(),"actor_loss":losses[-1].actor_loss if losses else None,"critic_loss":losses[-1].critic_loss if losses else None,"entropy":losses[-1].entropy if losses else None,"grad_norm":losses[-1].grad_norm if losses else None,"epsilon":self.epsilon,"seconds":time.perf_counter()-started,"gpu_peak":torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}

    def counterfactual_transitions(self,snapshot,graph,masks,actions):
        from netkeeper_sim.rl.replay_buffer import Transition
        scenario=self.env.environment.scenario
        if scenario is None or scenario.events: return []
        rows=[]
        for agent in AGENTS:
            alternatives=torch.nonzero(masks[agent],as_tuple=False).reshape(-1).tolist()
            alternatives=[int(value) for value in alternatives if int(value)!=int(actions[agent])]
            if not alternatives: continue
            if int(actions[agent])!=0: selected=[0]
            else:
                count=min(len(alternatives),max(1,int(self.cfg.counterfactual_candidate_samples)))
                selected=random.sample(alternatives,count)
            candidates=[]
            for candidate in selected:
                revised=dict(actions); revised[agent]=candidate
                kernel=UnifiedNetworkEnvironment(self.env.environment.reward_config); kernel.reset(scenario,seed=self.env.seed)
                outcome=kernel.step(snapshot,candidate_to_joint_action(snapshot,graph,revised))
                next_graph=snapshot_to_graph(outcome.next_snapshot); next_masks=action_masks(outcome.next_snapshot,next_graph)
                item=Transition(graph,masks,revised,dict(outcome.rewards.per_agent),next_graph,next_masks,outcome.terminated,outcome.truncated,graph.snapshot_id,next_graph.snapshot_id,{},0.0)
                candidates.append(item)
            rows.append((agent,max(candidates,key=lambda item:float(item.reward[agent]))))
        return rows

    def validate(self):
        original=self.env; sampler=UnifiedRLEnvironment(dataset_root=self.root,split="validation",seed=self.cfg.seed).sampler
        assert sampler is not None
        scenarios=[sampler.next_scenario() for _ in range(self.cfg.validation_scenarios)]; rows=[]
        self.trainer.actor.eval()
        try:
            for scenario in scenarios:
                self.env=UnifiedRLEnvironment(scenario=scenario,seed=self.cfg.seed)
                policy=self.run_episode(False)
                self.env=UnifiedRLEnvironment(scenario=scenario,seed=self.cfg.seed)
                baseline=self.run_episode(False,force_noop=True,step_limit=policy["steps"],allow_early_stop=False)
                policy["baseline_reward"]=baseline["reward"]; policy["reward_delta"]=policy["reward"]-baseline["reward"]
                rows.append(policy)
        finally: self.env=original
        return sum(x["reward_delta"] for x in rows)/len(rows),rows

    def train(self):
        for ep in range(self.episode,self.cfg.episodes):
            pending=[]; self.episode=ep; row=self.run_episode(True); row.update({"kind":"train","episode":ep}); pending.append(row); self.records.append(row)
            self.epsilon=max(self.cfg.epsilon_end,self.epsilon*self.cfg.epsilon_decay)
            selected=False
            if (ep+1)%self.cfg.validation_every==0:
                score, rows=self.validate(); self.validation_summary=summarize_validation(score,rows); record={"kind":"validation","episode":ep,"reward":score,"scenarios":rows}; pending.append(record); self.records.append(record)
                if score>self.best_validation:
                    self.best_validation=score; self.best_validation_episode=ep; self.bad_validations=0
                    selected=True
                else: self.bad_validations+=1
            # latest.pt is the transaction commit point.  Metrics are emitted
            # only after it succeeds and can be reconstructed from its records.
            self.save("latest.pt")
            if selected: self.save("best.pt", checkpoint_status=CHECKPOINT_CANDIDATE)
            self.append_metrics(pending)
            if self.bad_validations>=self.cfg.patience: break
        self.save("latest.pt"); self.curves()

    def append_metrics(self,rows):
        with (self.output/"metrics.jsonl").open("a") as h:
            for row in rows: h.write(json.dumps(row,default=str)+"\n")
        self.rewrite_csv()

    def rewrite_metrics(self):
        tmp=self.output/("metrics.jsonl."+uuid.uuid4().hex+".tmp")
        tmp.write_text("".join(json.dumps(row,default=str)+"\n" for row in self.records)); os.replace(tmp,self.output/"metrics.jsonl")
        self.rewrite_csv()

    def rewrite_csv(self):
        rows=[{k:v for k,v in row.items() if not isinstance(v,(dict,list))} for row in self.records]
        fields=sorted({key for row in rows for key in row})
        tmp=self.output/("metrics.csv."+uuid.uuid4().hex+".tmp")
        with tmp.open("w",newline="") as h:
            if fields:
                w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(rows)
        os.replace(tmp,self.output/"metrics.csv")

    def save(self,name, *, checkpoint_status="training_resume"):
        path=self.output/name; tmp=self.output/(name+"."+uuid.uuid4().hex+".tmp")
        dataset_path=self.root/"metadata/manifest.json"; dataset_text=dataset_path.read_text() if dataset_path.exists() else None
        torch.save({"model_version":MODEL_VERSION,"training_semantics_version":TRAINING_SEMANTICS_VERSION,"checkpoint_state_version":CHECKPOINT_STATE_VERSION,"schema_version":SCHEMA_VERSION,"checkpoint_status":checkpoint_status,"config":asdict(self.cfg),"actor":self.trainer.actor.state_dict(),"critic":self.trainer.critic.state_dict(),"target":self.trainer.target_critic.state_dict(),"actor_optimizer":self.trainer.actor_optimizer.state_dict(),"critic_optimizer":self.trainer.critic_optimizer.state_dict(),"scaler":self.trainer.scaler.state_dict(),"episode":self.episode,"global_step":self.global_step,"best_validation":self.best_validation,"best_validation_episode":self.best_validation_episode,"bad_validations":self.bad_validations,"validation_summary":self.validation_summary,"epsilon":self.epsilon,"rng":rng_state(),"sampler":self.env.sampler.state_dict() if self.env.sampler else None,"replay":self.trainer.buffer.state_dict(),"records":self.records,"provenance":{"training_split":"scenarios/train.jsonl","validation_split":"scenarios/validation.jsonl","dataset_manifest_sha256":hashlib.sha256(dataset_text.encode()).hexdigest() if dataset_text is not None else None,"seed":self.cfg.seed,"selection_metric":"validation_reward_delta_vs_no_update","selection_rule":"strictly highest paired validation reward delta; test split is never loaded"},"dataset":{"manifest":dataset_text}},tmp); os.replace(tmp,path)
        if name == "best.pt":
            (self.output/"best_resolved_config.yaml").write_text(yaml.safe_dump(asdict(self.cfg),sort_keys=True))

    def load(self,path):
        item=torch.load(path,map_location="cpu",weights_only=False)
        saved=dict(item.get("config") or {}); requested=asdict(self.cfg)
        compatible={key:value for key,value in saved.items() if key!="episodes"} == {key:value for key,value in requested.items() if key!="episodes"}
        if item.get("model_version")!=MODEL_VERSION or item.get("training_semantics_version")!=TRAINING_SEMANTICS_VERSION or item.get("checkpoint_state_version")!=CHECKPOINT_STATE_VERSION or not compatible or requested["episodes"] < int(saved.get("episodes",0)):
            raise ValueError({"code":"INCOMPATIBLE_CHECKPOINT","message":"model/config mismatch or episode budget decreases"})
        for key,module in (("actor",self.trainer.actor),("critic",self.trainer.critic),("target",self.trainer.target_critic)): module.load_state_dict(item[key],strict=True)
        loaded_episode=int(item["episode"])
        self.trainer.actor_optimizer.load_state_dict(item["actor_optimizer"]); self.trainer.critic_optimizer.load_state_dict(item["critic_optimizer"]); self.trainer.scaler.load_state_dict(item["scaler"]); self.episode=loaded_episode; self.global_step=item["global_step"]; self.best_validation=item["best_validation"]; self.best_validation_episode=item.get("best_validation_episode"); self.bad_validations=int(item["bad_validations"]); self.validation_summary=item.get("validation_summary"); self.epsilon=item["epsilon"]; self.records=list(item["records"]); self.trainer.buffer.load_state_dict(item["replay"]); set_rng_state(item["rng"])
        if item.get("sampler") is not None:
            if self.env.sampler is None: raise ValueError("checkpoint requires dataset sampler")
            self.env.sampler.load_state_dict(item["sampler"])
        if Path(path).name=="latest.pt" and self.best_validation_episode==int(item["episode"]):
            best_path=self.output/"best.pt"; repair=True
            if best_path.is_file():
                try:
                    best=torch.load(best_path,map_location="cpu",weights_only=False)
                    repair=best.get("episode")!=self.best_validation_episode or best.get("best_validation")!=self.best_validation
                except Exception: repair=True
            if repair: self.save("best.pt",checkpoint_status=CHECKPOINT_CANDIDATE)
        self.episode=loaded_episode+1
        self.rewrite_metrics()

    def curves(self):
        try:
            import matplotlib.pyplot as plt
            rows=[x for x in self.records if x["kind"]=="train"]; plt.plot([x["episode"] for x in rows],[x["reward"] for x in rows]); plt.xlabel("episode");plt.ylabel("reward");plt.savefig(self.output/"training_reward.png");plt.close()
        except ImportError:
            rows=[x for x in self.records if x["kind"]=="train"]
            _simple_png(self.output/"training_reward.png",[x["reward"] for x in rows])

def _simple_png(path, values):
    """Dependency-free fallback chart for machines without matplotlib."""
    w,h=640,360; pixels=bytearray([255,255,255,255]*w*h)
    if values:
        lo,hi=min(values),max(values); span=max(hi-lo,1e-9)
        for i,value in enumerate(values):
            x=20+round(i*(w-40)/max(1,len(values)-1)); y=20+round((hi-value)/span*(h-40))
            for dx in range(-2,3):
                for dy in range(-2,3):
                    xx,yy=x+dx,y+dy
                    if 0<=xx<w and 0<=yy<h: pixels[(yy*w+xx)*4:(yy*w+xx)*4+4]=bytes((30,90,200,255))
    raw=b''.join(b'\0'+bytes(pixels[y*w*4:(y+1)*w*4]) for y in range(h)); chunk=lambda kind,data: struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,6,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b''))

def summarize_validation(score, rows):
    agents={name:{"action_count":0,"no_op_count":0} for name in AGENTS}
    for row in rows:
        for name,diagnostic in row["agent_diagnostics"].items():
            agents[name]["action_count"]+=int(diagnostic["action_count"])
            agents[name]["no_op_count"]+=int(diagnostic["no_op_count"])
    return {"reward_delta_vs_no_update":float(score),"scenario_count":len(rows),"positive_scenario_count":sum(float(row["reward_delta"])>0 for row in rows),"agents":agents}
