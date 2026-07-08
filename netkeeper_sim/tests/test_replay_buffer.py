from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from netkeeper_sim.rl import MultiAgentNetworkEnvironment
from netkeeper_sim.rl.replay_buffer import ReplayBuffer, Transition
from netkeeper_sim.rl.trainer import EpsilonGreedyExplorer


def test_replay_buffer_capacity_and_sampling(single_path_topology, diamond_topology):
    first, _ = MultiAgentNetworkEnvironment(topology=single_path_topology).reset()
    second, _ = MultiAgentNetworkEnvironment(topology=diamond_topology).reset()
    buffer = ReplayBuffer(capacity=2, seed=1)

    buffer.add(_transition(first, "a"))
    buffer.add(_transition(second, "b"))
    buffer.add(_transition(first, "c"))

    assert len(buffer) == 2
    sample = buffer.sample(2)
    assert {item.topology_identifier for item in sample} <= {"b", "c"}
    assert len({item.state.node_features.shape[0] for item in sample}) >= 1


def test_epsilon_greedy_never_selects_invalid_actions():
    explorer = EpsilonGreedyExplorer(start=1.0, seed=3)
    logits = torch.zeros((3, 4))
    mask = torch.tensor([True, False, True])

    actions = explorer.select(logits, mask)

    assert actions[0] >= 0
    assert actions[1] == -1
    assert actions[2] >= 0


def _transition(state, topology_identifier: str) -> Transition:
    return Transition(
        state=state,
        observations=None,
        action={},
        action_indices={},
        previous_action=None,
        previous_action_indices=None,
        rewards={"ospf": 0.0, "bgp": 0.0, "performance": 0.0},
        next_state=state,
        next_observations=None,
        terminated=False,
        truncated=False,
        action_masks=state.parameter_masks,
        next_action_masks=state.parameter_masks,
        topology_identifier=topology_identifier,
    )
