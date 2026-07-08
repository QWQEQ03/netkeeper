from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from netkeeper_sim.rl.algorithms.coma import coma_counterfactual_baseline


def test_coma_baseline_selected_q_and_advantage_are_numerically_correct():
    logits = torch.zeros((2, 3))
    q_values = torch.tensor([[1.0, 2.0, 4.0], [10.0, 20.0, 30.0]])
    selected = torch.tensor([2, 0])
    mask = torch.tensor([True, True])

    baseline, selected_q, advantage = coma_counterfactual_baseline(
        logits,
        q_values,
        selected,
        mask,
    )

    assert torch.allclose(baseline, torch.tensor([7.0 / 3.0, 20.0]))
    assert torch.allclose(selected_q, torch.tensor([4.0, 10.0]))
    assert torch.allclose(advantage, torch.tensor([5.0 / 3.0, -10.0]))
    assert advantage.requires_grad is False


def test_coma_baseline_masks_invalid_entities():
    logits = torch.zeros((2, 2))
    q_values = torch.tensor([[1.0, 3.0], [100.0, 300.0]])
    selected = torch.tensor([1, 1])
    mask = torch.tensor([True, False])

    baseline, selected_q, advantage = coma_counterfactual_baseline(
        logits,
        q_values,
        selected,
        mask,
    )

    assert baseline.tolist() == [2.0, 0.0]
    assert selected_q.tolist() == [3.0, 300.0]
    assert advantage.tolist() == [1.0, 0.0]
