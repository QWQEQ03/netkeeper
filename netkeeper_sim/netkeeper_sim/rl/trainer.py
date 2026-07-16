"""Short, testable COMA update loop; not an experiment runner."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
import random
import torch
from torch import nn
from netkeeper_sim.rl.config import GraphNetworkConfig
from netkeeper_sim.rl.multi_agent_env import MultiAgentNetworkEnvironment
from netkeeper_sim.rl.networks.multi_agent_actor import AGENTS, MultiAgentActor
from netkeeper_sim.rl.networks.centralized_critic import CentralizedCritic
from netkeeper_sim.rl.networks.target_network import clone_target_network
from netkeeper_sim.rl.schema_adapter import masked_policy, sample_candidates
from netkeeper_sim.rl.algorithms.coma import coma_actor_loss, td_target
from netkeeper_sim.rl.replay_buffer import ReplayBuffer, Transition

class EpsilonGreedyExplorer:
    """Compatibility sampler; unified runner owns exploration scheduling."""
    def __init__(self,start=1.,end=.01,decay=.99,seed=None): self.epsilon=start;self.end=end;self.decay=decay;self.rng=random.Random(seed)
    def select(self,logits,mask,deterministic=False):
        valid=torch.as_tensor(mask,dtype=torch.bool,device=logits.device); out=torch.full((logits.size(0),),-1,dtype=torch.long,device=logits.device)
        for i in torch.nonzero(valid,as_tuple=False).reshape(-1).tolist(): out[i]=torch.argmax(logits[i]) if deterministic or self.rng.random()>=self.epsilon else self.rng.randrange(logits.size(-1))
        return out

@dataclass(frozen=True)
class UpdateStats:
    actor_loss: float; critic_loss: float; entropy: float; grad_norm: float

class COMATrainer:
    def __init__(self, env: MultiAgentNetworkEnvironment, config: GraphNetworkConfig, *, gamma: float = .85, target_interval: int = 16, amp: bool = True, seed: int = 42, device: str = "auto", entropy_coef: float = .01, gradient_clip: float = 1.0) -> None:
        self.env, self.gamma, self.target_interval, self.step_count = env, gamma, target_interval, 0
        if device not in {"auto","cpu","cuda"}: raise ValueError("device must be auto/cpu/cuda")
        if device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
        self.device = torch.device("cuda" if (device=="cuda" or device=="auto" and torch.cuda.is_available()) else "cpu")
        self.entropy_coef,self.gradient_clip=float(entropy_coef),float(gradient_clip)
        self.amp_enabled = amp and self.device.type == "cuda"; self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.actor = MultiAgentActor(config).to(self.device); self.critic = CentralizedCritic(config).to(self.device); self.target_critic = clone_target_network(self.critic).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), 1e-4); self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), 2e-4)
        self.buffer = ReplayBuffer(seed=seed); self.rng = random.Random(seed)

    @torch.no_grad()
    def rollout_actions(self, graph, masks, greedy=False) -> dict[str, int]:
        output = self.actor.forward_graph(graph, masks)
        return {agent: int(sample_candidates(output.logits[agent], output.masks[agent], greedy=greedy)[0]) for agent in AGENTS}

    def collect_one(self) -> Transition:
        graph, _obs, masks = self.env.reset(); actions = self.rollout_actions(graph, masks)
        result = self.env.step(actions)
        item = Transition(graph, masks, actions, dict(result.result.rewards.per_agent), result.graph, result.masks, result.result.terminated, result.result.truncated, graph.snapshot_id, result.graph.snapshot_id)
        self.buffer.add(item); return item

    def update(self, transitions: list[Transition]) -> UpdateStats:
        self.actor.train(); self.critic.train(); self.actor_optimizer.zero_grad(set_to_none=True); self.critic_optimizer.zero_grad(set_to_none=True)
        def calculate_losses(use_autocast: bool):
            actor_terms=[]; critic_terms=[]; entropies=[]
            with torch.autocast(device_type=self.device.type, enabled=use_autocast):
                for item in transitions:
                    output = self.actor.forward_graph(item.graph, item.masks)
                    joint = torch.tensor([item.actions[x] for x in AGENTS], device=self.device)
                    with torch.no_grad():
                        next_output = self.actor.forward_graph(item.next_graph, item.next_masks)
                        next_probabilities = {
                            name: masked_policy(next_output.logits[name], item.next_masks[name].to(self.device))[0]
                            for name in AGENTS
                        }
                        next_joint = torch.tensor(
                            [int(next_probabilities[name].argmax(-1).item()) for name in AGENTS],
                            device=self.device,
                        )
                    for agent in AGENTS:
                        q = self.critic.forward_graph(item.graph, joint, agent)
                        with torch.no_grad():
                            next_q_all = self.target_critic.forward_graph(item.next_graph, next_joint, agent)
                            next_q = (next_probabilities[agent] * next_q_all).sum(-1)
                            reward=item.reward[agent] if isinstance(item.reward,Mapping) else item.reward
                            target = td_target(torch.tensor([reward], device=self.device), torch.tensor([item.terminated], device=self.device), next_q, self.gamma)
                        chosen = joint[AGENTS.index(agent)].reshape(1)
                        critic_terms.append(nn.functional.smooth_l1_loss(q.gather(0, chosen), target))
                        if item.actor_weight and item.advantages is not None and agent in item.advantages:
                            probabilities,log_probabilities=masked_policy(output.logits[agent],output.masks[agent])
                            entropy=-(probabilities*torch.where(torch.isfinite(log_probabilities),log_probabilities,torch.zeros_like(log_probabilities))).sum(-1)
                            logp=log_probabilities.gather(1,chosen[:,None]).squeeze(1)
                            advantage=torch.tensor([float(item.advantages[agent])],device=self.device)
                            aloss=-(advantage*logp+self.entropy_coef*entropy).mean()
                        elif item.actor_weight:
                            aloss, _adv, entropy = coma_actor_loss(output.logits[agent], q.detach(), chosen, output.masks[agent], entropy_coef=self.entropy_coef)
                        else: continue
                        actor_terms.append(aloss); entropies.append(entropy.mean())
                actor_loss=torch.stack(actor_terms).mean() if actor_terms else torch.zeros((),device=self.device,requires_grad=True)
                critic_loss=torch.stack(critic_terms).mean()
            return actor_loss,critic_loss,entropies

        actor_loss,critic_loss,entropies=calculate_losses(self.amp_enabled)
        if not torch.isfinite(actor_loss + critic_loss) and self.amp_enabled:
            actor_loss,critic_loss,entropies=calculate_losses(False)
        if not torch.isfinite(actor_loss + critic_loss): raise FloatingPointError("non-finite COMA loss after FP32 retry")
        self.scaler.scale(critic_loss).backward(); self.scaler.scale(actor_loss).backward()
        self.scaler.unscale_(self.critic_optimizer); self.scaler.unscale_(self.actor_optimizer)
        norm=max(float(nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip)), float(nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)))
        self.scaler.step(self.critic_optimizer); self.scaler.step(self.actor_optimizer); self.scaler.update()
        self.step_count += 1
        if self.step_count % self.target_interval == 0: self.target_critic.load_state_dict(self.critic.state_dict()); self.target_critic.requires_grad_(False)
        return UpdateStats(float(actor_loss.detach()), float(critic_loss.detach()), float(torch.stack(entropies).mean().detach()) if entropies else 0.0, norm)
