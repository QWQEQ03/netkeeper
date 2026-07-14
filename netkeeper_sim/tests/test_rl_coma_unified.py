from __future__ import annotations
import json
from pathlib import Path
import torch
import pytest
from netkeeper_sim.dataset.scenarios import scenario_from_record
from netkeeper_sim.rl.algorithms.coma import coma_actor_loss, coma_counterfactual, td_target
from netkeeper_sim.rl.config import GraphNetworkConfig
from netkeeper_sim.rl.multi_agent_env import MultiAgentNetworkEnvironment
from netkeeper_sim.rl.networks.centralized_critic import CentralizedCritic
from netkeeper_sim.rl.networks.multi_agent_actor import MultiAgentActor
from netkeeper_sim.rl.networks.target_network import clone_target_network
from netkeeper_sim.rl.networks.graph_encoder import SharedGraphTransformerEncoder
from netkeeper_sim.rl.schema_adapter import masked_policy
from netkeeper_sim.rl.trainer import COMATrainer

ROOT=Path(__file__).resolve().parents[2]/"data"/"netkeeper_lite"
def env():
    s=scenario_from_record(ROOT,json.loads((ROOT/"scenarios/train.jsonl").read_text().splitlines()[0]))
    return MultiAgentNetworkEnvironment(scenario=s,seed=7)
def cfg(): return GraphNetworkConfig(17,11,hidden_dim=64,gcn_layers=2,transformer_layers=2,transformer_heads=4,dropout=.1)

def test_toy_coma_baseline_advantage_loss_and_chosen_not_argmax():
    logits=torch.tensor([[2.,0.,7.]],requires_grad=True); q=torch.tensor([[1.,3.,99.]],requires_grad=True); mask=torch.tensor([[True,True,False]])
    base, chosen_q, adv, logp=coma_counterfactual(logits,q,torch.tensor([1]),mask)
    p=torch.softmax(torch.tensor([2.,0.]),0); assert base.item()==pytest.approx((p*torch.tensor([1.,3.])).sum().item())
    assert chosen_q.item()==3 and adv.requires_grad is False and logp.item()==pytest.approx(torch.log(p[1]).item())
    loss, _, entropy=coma_actor_loss(logits,q,torch.tensor([1]),mask); loss.backward()
    assert logits.grad[0,1] != 0 and q.grad is None and torch.isfinite(entropy).all()

def test_masks_normalize_and_td_distinguishes_terminal_truncation():
    p,_=masked_policy(torch.tensor([[0.,5.,10.]]),torch.tensor([[True,False,False]])); assert p.tolist()==[[1.,0.,0.]]
    assert td_target(torch.tensor([1.]),torch.tensor([True]),torch.tensor([9.])).item()==1.
    assert td_target(torch.tensor([1.]),torch.tensor([False]),torch.tensor([9.])).item()==pytest.approx(1+0.85*9)

def test_actor_entity_logits_are_distinct_and_ownership_isolated():
    e=env(); graph,_,masks=e.reset(); actor=MultiAgentActor(cfg()); critic=CentralizedCritic(cfg()); out=actor.forward_graph(graph,masks)
    assert out.logits["ospf"].shape == masks["ospf"].shape
    assert out.logits["performance"].shape == masks["performance"].shape
    assert not torch.allclose(out.logits["ospf"][1:65],out.logits["ospf"][65:129])
    actor_ids={id(x) for x in actor.parameters()}; critic_ids={id(x) for x in critic.parameters()}
    assert actor_ids.isdisjoint(critic_ids) and len(actor_ids)==len(list(actor.parameters()))
    target=clone_target_network(critic); assert all(not p.requires_grad for p in target.parameters())

def test_encoder_transformer_concat_and_multi_graph_batch_shape():
    e=env(); one,_,_=e.reset(); other,_,_=MultiAgentNetworkEnvironment(dataset_root=ROOT,split="train",seed=9).reset()
    encoder=SharedGraphTransformerEncoder(cfg())
    x=torch.cat((one.node_features,other.node_features)); edge=torch.cat((one.edge_index,other.edge_index+len(one.node_ids)),1); attrs=torch.cat((one.edge_features,other.edge_features)); batch=torch.cat((torch.zeros(len(one.node_ids),dtype=torch.long),torch.ones(len(other.node_ids),dtype=torch.long)))
    result=encoder(x,edge,attrs,batch=batch)
    assert result.node_embeddings.shape == (x.size(0),64) and result.graph_embedding.shape == (2,64)

def test_critic_holds_other_actions_fixed_and_target_update_boundary():
    e=env(); graph,_,masks=e.reset(); critic=CentralizedCritic(cfg());
    q_a=critic.forward_graph(graph,torch.tensor([0,0,0]),"ospf"); q_b=critic.forward_graph(graph,torch.tensor([0,4,0]),"ospf")
    assert q_a.shape==q_b.shape==masks["ospf"].shape and not torch.allclose(q_a,q_b)
    trainer=COMATrainer(e,cfg(),target_interval=2); item=trainer.collect_one(); before=next(trainer.target_critic.parameters()).detach().clone(); trainer.update([item]); assert torch.allclose(before,next(trainer.target_critic.parameters()))
    trainer.update([item]); assert not torch.allclose(before,next(trainer.target_critic.parameters()))

def test_short_real_transition_update_changes_only_actor_critic():
    trainer=COMATrainer(env(),cfg()); item=trainer.collect_one(); a=next(trainer.actor.parameters()).detach().clone(); c=next(trainer.critic.parameters()).detach().clone(); t=next(trainer.target_critic.parameters()).detach().clone()
    stats=trainer.update([item,item]); assert all(torch.isfinite(torch.tensor(v)) for v in stats.__dict__.values())
    assert not torch.allclose(a,next(trainer.actor.parameters())) and not torch.allclose(c,next(trainer.critic.parameters())) and torch.allclose(t,next(trainer.target_critic.parameters()))
    assert all(parameter.grad is None for parameter in trainer.target_critic.parameters())

def test_two_update_real_train_smoke_replay_sample_is_finite():
    trainer=COMATrainer(MultiAgentNetworkEnvironment(dataset_root=ROOT,split="train",seed=17),cfg(),seed=17)
    trainer.collect_one(); trainer.collect_one(); batch=trainer.buffer.sample(2)
    first=trainer.update(batch); second=trainer.update(batch)
    assert all(torch.isfinite(torch.tensor(v)) for v in (*first.__dict__.values(),*second.__dict__.values()))
