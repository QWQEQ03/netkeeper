from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from netkeeper_sim.rl.algorithms.losses import actor_loss, critic_loss, td_target


def test_actor_loss_direction_prefers_positive_advantage_action():
    logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
    q_values = torch.tensor([[0.0, 2.0]])
    selected = torch.tensor([1])
    mask = torch.tensor([True])

    loss = actor_loss(logits, q_values, selected, mask)
    loss.backward()

    assert loss.item() > 0.0
    assert logits.grad[0, 1] < 0.0


def test_td_target_bootstraps_and_stops_on_done():
    rewards = torch.tensor([1.0, 1.0])
    dones = torch.tensor([0.0, 1.0])
    next_q = torch.tensor([[2.0, 4.0], [10.0, 20.0]])
    mask = torch.tensor([True, True])

    target = td_target(rewards, dones, next_q, mask, gamma=0.5)

    assert torch.allclose(target, torch.tensor([3.0, 1.0]))


def test_critic_loss_gathers_selected_q():
    q_values = torch.tensor([[1.0, 5.0], [3.0, 7.0]], requires_grad=True)
    selected = torch.tensor([1, 0])
    target = torch.tensor([6.0, 1.0])
    mask = torch.tensor([True, True])

    loss = critic_loss(q_values, selected, target, mask, loss="mse")
    loss.backward()

    assert torch.isclose(loss, torch.tensor((1.0 + 4.0) / 2.0))
    assert q_values.grad[0, 1] != 0.0
    assert q_values.grad[0, 0] == 0.0


def test_losses_reject_invalid_selected_action_for_valid_entity():
    q_values = torch.tensor([[1.0, 2.0]])
    logits = torch.tensor([[0.0, 0.0]])
    selected = torch.tensor([-1])
    mask = torch.tensor([True])

    with pytest.raises(ValueError):
        actor_loss(logits, q_values, selected, mask)
    with pytest.raises(ValueError):
        critic_loss(q_values, selected, torch.tensor([1.0]), mask)
