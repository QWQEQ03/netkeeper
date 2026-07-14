from __future__ import annotations
import pytest
pytestmark = pytest.mark.skip(reason="critic migration is deferred to the COMA block")

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from netkeeper_sim.rl import MultiAgentNetworkEnvironment
from netkeeper_sim.rl.config import GraphNetworkConfig
from netkeeper_sim.rl.networks import CentralizedCritic
from netkeeper_sim.rl.networks.target_network import clone_target_network
from netkeeper_sim.rl.algorithms.target_update import hard_update, soft_update


def test_critic_outputs_all_candidate_action_q_values(diamond_topology):
    state, _ = MultiAgentNetworkEnvironment(topology=diamond_topology).reset()
    critic = CentralizedCritic(
        GraphNetworkConfig.debug(state.node_features.shape[1], state.edge_features.shape[1])
    )

    output = critic.forward_state(state, "ospf", "ospf_weight")

    assert output.logits["ospf_weight"].shape == (len(state.link_ids), 64)


def test_target_critic_updates_and_has_no_training_gradients(diamond_topology):
    state, _ = MultiAgentNetworkEnvironment(topology=diamond_topology).reset()
    critic = CentralizedCritic(
        GraphNetworkConfig.debug(state.node_features.shape[1], state.edge_features.shape[1])
    )
    target = clone_target_network(critic)
    with torch.no_grad():
        next(critic.parameters()).add_(1.0)

    hard_update(target, critic)
    assert all(not parameter.requires_grad for parameter in target.parameters())
    for target_parameter, source_parameter in zip(target.parameters(), critic.parameters()):
        assert torch.allclose(target_parameter, source_parameter)

    with torch.no_grad():
        next(critic.parameters()).add_(1.0)
    before = next(target.parameters()).clone()
    soft_update(target, critic, tau=0.5)
    after = next(target.parameters())
    assert not torch.allclose(before, after)
