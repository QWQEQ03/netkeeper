from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from netkeeper_sim.policies import ForwardPolicy
from netkeeper_sim.rl.checkpoints import load_checkpoint, save_checkpoint
from netkeeper_sim.rl.config import COMATrainingConfig, RLConfig
from netkeeper_sim.rl.multi_agent_env import MultiAgentNetworkEnvironment
from netkeeper_sim.rl.trainer import COMATrainer
from netkeeper_sim.topology.model import build_topology_from_edges
from netkeeper_sim.traffic.matrix import TrafficDemand, TrafficMatrix


def test_one_training_update_changes_actor_and_critic_parameters():
    trainer = COMATrainer(_env(), COMATrainingConfig.debug())
    state, observations = trainer.env.reset(seed=1)
    action, action_indices = trainer.select_action(state)
    result = trainer.env.step(action)
    from netkeeper_sim.rl.replay_buffer import Transition

    transition = Transition(
        state=state,
        observations=observations,
        action=action,
        action_indices=action_indices,
        previous_action=None,
        previous_action_indices=None,
        rewards=result.rewards,
        next_state=result.state,
        next_observations=result.observations,
        terminated=result.terminated,
        truncated=result.truncated,
        action_masks=state.parameter_masks,
        next_action_masks=result.state.parameter_masks,
        topology_identifier="smoke",
    )
    actor_before = next(trainer.actor.parameters()).detach().clone()
    critic_before = next(trainer.critic.parameters()).detach().clone()

    actor_loss, critic_loss, _norm = trainer.update([transition, transition])

    assert torch.isfinite(actor_loss)
    assert torch.isfinite(critic_loss)
    assert not torch.allclose(actor_before, next(trainer.actor.parameters()).detach())
    assert not torch.allclose(critic_before, next(trainer.critic.parameters()).detach())
    assert trainer.actor.encoder.node_projection.weight.grad is not None
    assert all(parameter.grad is None for parameter in trainer.target_critic.parameters())


def test_debug_training_smoke_and_checkpoint_roundtrip(tmp_path):
    trainer = COMATrainer(_env(), COMATrainingConfig.debug())

    logs = trainer.train()

    assert logs
    assert len(logs) <= 10
    checkpoint = tmp_path / "coma.pt"
    save_checkpoint(
        checkpoint,
        trainer.actor,
        trainer.critic,
        trainer.target_critic,
        trainer.actor_optimizer,
        trainer.critic_optimizer,
        metadata={"logs": len(logs)},
    )
    state = trainer.env.current_state
    assert state is not None
    before_action, before_output = trainer.evaluate_deterministic()

    reloaded = COMATrainer(_env(), COMATrainingConfig.debug())
    metadata = load_checkpoint(
        checkpoint,
        reloaded.actor,
        reloaded.critic,
        reloaded.target_critic,
        reloaded.actor_optimizer,
        reloaded.critic_optimizer,
    )
    reloaded.env.reset(seed=42)
    after_action, after_output = reloaded.evaluate_deterministic()

    assert metadata["logs"] == len(logs)
    assert before_action.keys() == after_action.keys()
    assert before_output.ospf.logits["ospf_weight"].shape == after_output.ospf.logits["ospf_weight"].shape


def _env() -> MultiAgentNetworkEnvironment:
    topology = build_topology_from_edges(
        [
            ("R1", "R2", 1),
            ("R2", "R4", 1),
            ("R1", "R3", 1),
            ("R3", "R4", 1),
        ]
    )
    return MultiAgentNetworkEnvironment(
        topology=topology,
        traffic_matrix=TrafficMatrix((TrafficDemand("R1", "R4", 100.0),)),
        policies=[ForwardPolicy("must-use-r3", "R1", "R4", "R3")],
        config=RLConfig(max_steps=5),
    )
