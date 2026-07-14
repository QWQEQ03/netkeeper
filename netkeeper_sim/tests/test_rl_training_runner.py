from __future__ import annotations
from pathlib import Path
import json
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

def test_freeze_checkpoint_requires_validation_selected_candidate(tmp_path):
    runner=TrainingRunner(ROOT,tiny(),tmp_path/'run'); runner.train()
    from netkeeper_sim.rl.cli import freeze_checkpoint
    freeze_checkpoint(runner.output/'best.pt',ROOT)
    import torch
    item=torch.load(runner.output/'best.pt',map_location='cpu',weights_only=False)
    assert item['checkpoint_status']=='formal_validation_selected'
    assert (runner.output/'best_resolved_config.yaml').is_file()
    assert (runner.output/'checkpoint_manifest.json').is_file()

def test_train_validation_trajectory_aggregation_is_identical():
    scenario=scenario_from_record(ROOT,json.loads((ROOT/"scenarios/train.jsonl").read_text().splitlines()[0])); env=UnifiedNetworkEnvironment(); snapshot,_=env.reset(scenario)
    from netkeeper_sim.schemas import JointAction
    trajectory=[]
    for _ in range(2):
        result=env.step(snapshot,JointAction((),snapshot_id=snapshot.snapshot_id)); trajectory.append(result);snapshot=result.next_snapshot
    train=aggregate_trajectory(trajectory); validation=aggregate_trajectory(trajectory)
    assert train==validation and train["reward"]==sum(x.rewards.total_reward for x in trajectory) and train["mean_reward"]==train["reward"]/2
