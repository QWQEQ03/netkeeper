from __future__ import annotations
import argparse, hashlib, json, os, uuid
from pathlib import Path
from netkeeper_sim.rl.training import CHECKPOINT_STATE_VERSION, TRAINING_SEMANTICS_VERSION, TrainingConfig, TrainingRunner
from netkeeper_sim.rl.dispatcher import TrainedPolicyDispatcher
from netkeeper_sim.dataset.scenarios import scenario_from_record
from netkeeper_sim.simulator import UnifiedNetworkEnvironment
from netkeeper_sim.api.models import OptimizationRequest
import torch
from netkeeper_sim.schemas.ids import SCHEMA_VERSION

def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def freeze_checkpoint(checkpoint, dataset_root):
    """Strict-load and validation-dispatch a validation-selected checkpoint, then freeze it."""
    checkpoint=Path(checkpoint); root=Path(dataset_root); data=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if data.get('checkpoint_status') != 'validation_selected_candidate':
        raise ValueError('checkpoint is not a validation-selected candidate')
    if data.get('model_version') != 'rl-coma-v3' or data.get('training_semantics_version') != TRAINING_SEMANTICS_VERSION or data.get('checkpoint_state_version') != CHECKPOINT_STATE_VERSION or data.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('checkpoint model/schema version is incomplete')
    if int(data.get('global_step',0)) <= 0 or not isinstance(data.get('best_validation'),(int,float)) or not torch.isfinite(torch.tensor(float(data['best_validation']))):
        raise ValueError('checkpoint has no finite trained validation selection')
    provenance=data.get('provenance') or {}
    if provenance.get('training_split')!='scenarios/train.jsonl' or provenance.get('validation_split')!='scenarios/validation.jsonl':
        raise ValueError('checkpoint split provenance is incomplete')
    config_path=checkpoint.parent/'best_resolved_config.yaml'
    if not config_path.is_file(): raise ValueError('best_resolved_config.yaml is required to freeze a checkpoint')
    config=data.get('config') or {}; summary=data.get('validation_summary') or {}
    minimum=float(config.get('formal_min_validation_delta',0.0)); score=float(data['best_validation'])
    if score <= minimum or float(summary.get('reward_delta_vs_no_update',float('-inf'))) != score:
        raise ValueError(f'checkpoint validation delta {score} does not exceed formal minimum {minimum}')
    if int(summary.get('scenario_count',0)) != int(config.get('validation_scenarios',0)):
        raise ValueError('checkpoint validation summary is incomplete')
    minimum_actions=int(config.get('formal_min_agent_actions',1)); agents=summary.get('agents') or {}
    inactive=[name for name in ('ospf','bgp','performance') if int((agents.get(name) or {}).get('action_count',0)) < minimum_actions]
    if inactive: raise ValueError(f'checkpoint has inactive validation agents: {inactive}')
    dispatcher=TrainedPolicyDispatcher(str(checkpoint),expected_dataset_manifest=str(root/'metadata/manifest.json'))
    first=json.loads((root/'scenarios/validation.jsonl').read_text().splitlines()[0])
    scenario=scenario_from_record(root,first); environment=UnifiedNetworkEnvironment(); snapshot,_=environment.reset(scenario,seed=int(data['config']['seed']))
    action=dispatcher.dispatch(snapshot,OptimizationRequest(snapshot.snapshot_id,('policy_consistency','mlu')))
    result=environment.step(snapshot,action)
    if result.errors: raise ValueError(f'validation dispatcher action rejected:{[x.to_dict() for x in result.errors]}')
    data['checkpoint_status']='formal_validation_selected'
    data['formalization']={'validation_smoke_scenario_id':scenario.scenario_id,'validation_smoke_seed':int(data['config']['seed']),'strict_load':True,'greedy_dispatch':True,'test_split_accessed':False}
    tmp=checkpoint.with_suffix(checkpoint.suffix+'.freeze.tmp'); torch.save(data,tmp); os.replace(tmp,checkpoint)
    bundle={'checkpoint_path':str(checkpoint),'checkpoint_sha256':_sha(checkpoint),'resolved_config_path':str(config_path),'resolved_config_sha256':_sha(config_path),'dataset_manifest_sha256':_sha(root/'metadata/manifest.json'),'checkpoint_status':'formal_validation_selected','model_version':data['model_version'],'training_semantics_version':data['training_semantics_version'],'checkpoint_state_version':data['checkpoint_state_version'],'schema_version':data['schema_version'],'validation_summary':summary,'provenance':data['provenance'],'formalization':data['formalization']}
    (checkpoint.parent/'checkpoint_manifest.json').write_text(json.dumps(bundle,sort_keys=True,indent=2)+'\n')
    print(json.dumps(bundle,ensure_ascii=False))
def main():
 p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--dataset-root',default='../data/netkeeper_lite');p.add_argument('--output-root',default='runs');p.add_argument('--episodes',type=int);p.add_argument('--seed',type=int);p.add_argument('--resume');p.add_argument('--freeze-checkpoint');a=p.parse_args()
 if a.freeze_checkpoint:
  freeze_checkpoint(a.freeze_checkpoint,a.dataset_root); return
 if not a.config: p.error('--config is required unless --freeze-checkpoint is used')
 c=TrainingConfig.load(a.config)
 if a.episodes is not None: c=type(c)(**{**c.__dict__,'episodes':a.episodes})
 if a.seed is not None: c=type(c)(**{**c.__dict__,'seed':a.seed})
 out=Path(a.resume).resolve().parent if a.resume else Path(a.output_root)/(f'rl-{uuid.uuid4().hex[:10]}'); TrainingRunner(a.dataset_root,c,out,resume=a.resume).train()
if __name__=='__main__': main()
