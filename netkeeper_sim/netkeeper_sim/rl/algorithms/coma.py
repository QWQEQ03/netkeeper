"""Numerically stable macro-action COMA utilities."""
from __future__ import annotations
import torch
from netkeeper_sim.rl.schema_adapter import masked_entropy, masked_policy


def coma_counterfactual(logits: torch.Tensor, q_candidates: torch.Tensor, chosen: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return baseline, chosen Q, detached advantage and chosen log-prob.

    Rows are batch/factors; candidate axis is the current factor only while all
    other rollout factors remain encoded in `q_candidates`' critic context.
    """
    probabilities, log_probs = masked_policy(logits, mask)
    if q_candidates.ndim == 1: q_candidates = q_candidates.unsqueeze(0)
    chosen = torch.as_tensor(chosen, device=q_candidates.device, dtype=torch.long).reshape(-1)
    if chosen.numel() != q_candidates.size(0): raise ValueError("chosen action batch mismatch")
    baseline = (probabilities * q_candidates).sum(-1)
    selected_q = q_candidates.gather(1, chosen[:, None]).squeeze(1)
    selected_logp = log_probs.gather(1, chosen[:, None]).squeeze(1)
    return baseline, selected_q, (selected_q - baseline).detach(), selected_logp


def coma_actor_loss(logits: torch.Tensor, q_candidates: torch.Tensor, chosen: torch.Tensor, mask: torch.Tensor, entropy_coef: float = 0.0, valid_rows: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    baseline, _q, advantage, logp = coma_counterfactual(logits, q_candidates.detach(), chosen, mask)
    entropy = masked_entropy(logits, mask)
    rows = torch.ones_like(advantage, dtype=torch.bool) if valid_rows is None else valid_rows.bool()
    denominator = rows.float().sum().clamp_min(1.0)
    loss = -((advantage * logp + entropy_coef * entropy) * rows).sum() / denominator
    return loss, advantage, entropy


def td_target(reward: torch.Tensor, terminated: torch.Tensor, next_q: torch.Tensor, gamma: float = 0.85) -> torch.Tensor:
    """Time-limit truncation deliberately bootstraps; true terminal does not."""
    return reward.reshape(-1) + gamma * (1.0 - terminated.float().reshape(-1)) * next_q.reshape(-1)

# Compatibility helpers retained for old unit imports; new training uses the
# macro-candidate functions above.
def legal_action_probabilities(logits, entity_mask):
    mask=torch.as_tensor(entity_mask,dtype=torch.bool,device=logits.device).reshape(-1)
    return torch.softmax(logits,dim=-1)*mask[:,None]
def selected_log_probabilities(logits, selected):
    selected=torch.as_tensor(selected,dtype=torch.long,device=logits.device).reshape(-1)
    return torch.log_softmax(logits,dim=-1).gather(1,selected.clamp_min(0)[:,None]).squeeze(1)
def coma_counterfactual_baseline(actor_logits,critic_q_values,selected_actions,entity_mask,detach_advantage=True):
    p=legal_action_probabilities(actor_logits,entity_mask); selected=torch.as_tensor(selected_actions,dtype=torch.long,device=actor_logits.device).reshape(-1); valid=torch.as_tensor(entity_mask,device=actor_logits.device,dtype=torch.bool).reshape(-1)
    if torch.any((selected[valid]<0)|(selected[valid]>=actor_logits.size(-1))): raise ValueError("invalid selected action")
    baseline=(p*critic_q_values).sum(-1); q=critic_q_values.gather(1,selected.clamp_min(0)[:,None]).squeeze(1); adv=(q-baseline)*valid.float(); return baseline,q,adv.detach() if detach_advantage else adv
