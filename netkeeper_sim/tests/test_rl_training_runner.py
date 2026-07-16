from __future__ import annotations
from pathlib import Path
import json
import torch
import pytest
from netkeeper_sim.rl.training import TrainingConfig, TrainingRunner, aggregate_trajectory
from netkeeper_sim.rl.dispatcher import TrainedPolicyDispatcher
from netkeeper_sim.dataset.scenarios import scenario_from_record
from netkeeper_sim.simulator import UnifiedNetworkEnvironment
from netkeeper_sim.api.models import OptimizationRequest

ROOT=Path(__file__).resolve().parents[2]/"data"/"netkeeper_lite"
def tiny():
 c=TrainingConfig.load(Path(__file__).resolve().parents[1]/"configs/rl_debug_train.yaml")
 return type(c)(**{**c.__dict__,"episodes":2,"max_steps":2,"validation_every":1,"validation_scenarios":1,"patience":9})
def test_checkpoint_resume_validation_and_dispatcher(tmp_path):
 r=TrainingRunner(ROOT,tiny(),tmp_path/"run");r.train(); assert (r.output/"best.pt").exists() and (r.output/"latest.pt").exists()
 restored=TrainingRunner(ROOT,tiny(),r.output,resume=r.output/"latest.pt"); assert restored.global_step==r.global_step and len(restored.trainer.buffer)==len(r.trainer.buffer)
 scenario=scenario_from_record(ROOT,json.loads((ROOT/"scenarios/train.jsonl").read_text().splitlines()[0])); env=UnifiedNetworkEnvironment();snapshot,_=env.reset(scenario); action=TrainedPolicyDispatcher(str(r.output/"best.pt")).dispatch(snapshot,OptimizationRequest(snapshot.snapshot_id,("mlu",)))
 assert action.snapshot_id==snapshot.snapshot_id; assert env.step(snapshot,action).next_snapshot.snapshot_id != snapshot.snapshot_id

def test_resume_permits_only_an_increased_episode_budget(tmp_path):
    first=tiny(); runner=TrainingRunner(ROOT,first,tmp_path/'run'); runner.train()
    extended=type(first)(**{**first.__dict__,"episodes":3})
    resumed=TrainingRunner(ROOT,extended,runner.output,resume=runner.output/'latest.pt')
    assert resumed.episode==2
    assert resumed.bad_validations==runner.bad_validations
    assert resumed.records==runner.records

def test_resume_rebuilds_logs_and_matches_uninterrupted_training(tmp_path):
    base=type(tiny())(**{**tiny().__dict__,"episodes":2,"validation_every":1,"patience":9})
    complete=type(base)(**{**base.__dict__,"episodes":3})
    uninterrupted=TrainingRunner(ROOT,complete,tmp_path/'continuous'); uninterrupted.train()
    interrupted=TrainingRunner(ROOT,base,tmp_path/'resumed'); interrupted.train()
    with (interrupted.output/'metrics.jsonl').open('a') as handle:
        handle.write(json.dumps({"kind":"train","episode":999,"reward":999})+'\n')
    resumed=TrainingRunner(ROOT,complete,interrupted.output,resume=interrupted.output/'latest.pt'); resumed.train()
    restored=[json.loads(line) for line in (resumed.output/'metrics.jsonl').read_text().splitlines()]
    assert all(row.get('episode') != 999 for row in restored)
    assert [(x.get('kind'),x.get('episode'),x.get('reward')) for x in restored] == [(x.get('kind'),x.get('episode'),x.get('reward')) for x in uninterrupted.records]
    assert all(torch.equal(left,right) for left,right in zip(resumed.trainer.actor.state_dict().values(),uninterrupted.trainer.actor.state_dict().values()))

def test_freeze_checkpoint_requires_validation_selected_candidate(tmp_path):
    runner=TrainingRunner(ROOT,tiny(),tmp_path/'run'); runner.train()
    from netkeeper_sim.rl.cli import freeze_checkpoint
    item=torch.load(runner.output/'best.pt',map_location='cpu',weights_only=False)
    item['best_validation']=-0.01; item['validation_summary']['reward_delta_vs_no_update']=-0.01
    torch.save(item,runner.output/'best.pt')
    with pytest.raises(ValueError,match='does not exceed formal minimum'):
        freeze_checkpoint(runner.output/'best.pt',ROOT)
    item['best_validation']=0.01; item['validation_summary']['reward_delta_vs_no_update']=0.01
    for agent in ('ospf','bgp','performance'): item['validation_summary']['agents'][agent]['action_count']=1
    torch.save(item,runner.output/'best.pt')
    freeze_checkpoint(runner.output/'best.pt',ROOT)
    item=torch.load(runner.output/'best.pt',map_location='cpu',weights_only=False)
    assert item['checkpoint_status']=='formal_validation_selected'
    assert (runner.output/'best_resolved_config.yaml').is_file()
    assert (runner.output/'checkpoint_manifest.json').is_file()

def test_resume_repairs_best_checkpoint_if_latest_committed_first(tmp_path):
    config=type(tiny())(**{**tiny().__dict__,"episodes":1,"validation_every":1})
    runner=TrainingRunner(ROOT,config,tmp_path/'run'); runner.train()
    (runner.output/'best.pt').unlink()
    restored=TrainingRunner(ROOT,config,runner.output,resume=runner.output/'latest.pt')
    item=torch.load(restored.output/'best.pt',map_location='cpu',weights_only=False)
    assert item['checkpoint_status']=='validation_selected_candidate' and item['episode']==0

def test_train_validation_trajectory_aggregation_is_identical():
    scenario=scenario_from_record(ROOT,json.loads((ROOT/"scenarios/train.jsonl").read_text().splitlines()[0])); env=UnifiedNetworkEnvironment(); snapshot,_=env.reset(scenario)
    from netkeeper_sim.schemas import JointAction
    trajectory=[]
    for _ in range(2):
        result=env.step(snapshot,JointAction((),snapshot_id=snapshot.snapshot_id)); trajectory.append(result);snapshot=result.next_snapshot
    train=aggregate_trajectory(trajectory); validation=aggregate_trajectory(trajectory)
    assert train==validation and train["reward"]==sum(x.rewards.total_reward for x in trajectory) and train["mean_reward"]==train["reward"]/2

def test_static_no_improvement_stops_early_and_reports_each_agent(tmp_path):
    config=type(tiny())(**{**tiny().__dict__,"max_steps":10,"static_patience":2})
    runner=TrainingRunner(ROOT,config,tmp_path/"diagnostic")
    row=runner.run_episode(False,force_noop=True)
    assert row["steps"]==2 and row["early_stop"] and row["stop_reason"]=="static_no_improvement"
    assert set(row["agent_diagnostics"])=={"ospf","bgp","performance"}
    assert all(item["no_op_count"]==2 and item["action_count"]==0 for item in row["agent_diagnostics"].values())
