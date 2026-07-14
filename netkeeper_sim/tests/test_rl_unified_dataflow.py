from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch

from netkeeper_sim.dataset.scenarios import scenario_from_record
from netkeeper_sim.rl.multi_agent_env import BalancedScenarioSampler, MultiAgentNetworkEnvironment
from netkeeper_sim.rl.schema_adapter import ACTION_VALUES, action_masks, candidate_to_joint_action, masked_entropy, masked_policy, sample_candidates, snapshot_to_graph, to_pyg_data
from netkeeper_sim.simulator import RewardConfig, UnifiedNetworkEnvironment
from netkeeper_sim.simulator.unified_environment import _reward
from netkeeper_sim.schemas import Metrics

ROOT = Path(__file__).resolve().parents[2] / "data" / "netkeeper_lite"


def _scenario(split="train", index=0):
    row = json.loads((ROOT / "scenarios" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()[index])
    return scenario_from_record(ROOT, row)


def test_train_validation_only_sampler_is_seeded_and_never_opens_test():
    first = BalancedScenarioSampler(ROOT, "train", seed=9)
    second = BalancedScenarioSampler(ROOT, "train", seed=9)
    assert [first.next_record()["scenario_id"] for _ in range(12)] == [second.next_record()["scenario_id"] for _ in range(12)]
    assert {row["split"] for row in first.rows} == {"train"}
    validation = BalancedScenarioSampler(ROOT, "validation", seed=9)
    assert {row["split"] for row in validation.rows} == {"validation"}
    with pytest.raises(ValueError): BalancedScenarioSampler(ROOT, "test")


def test_schema_graph_is_stable_batched_and_has_fixed_dimensions():
    snapshots = []
    for index in (0, 1):
        env = UnifiedNetworkEnvironment(); snapshot, _ = env.reset(_scenario("train", index), seed=7); snapshots.append(snapshot)
    first, second = (snapshot_to_graph(item) for item in snapshots)
    assert first.node_features.shape[1] == 17 and first.edge_features.shape[1] == 11
    assert first.node_ids == tuple(sorted(first.node_ids, key=lambda x: int(x[1:])))
    assert len(first.edge_ids) == 2 * len(first.link_ids)
    assert torch.isfinite(first.node_features).all() and torch.isfinite(first.edge_features).all()
    batch = Batch.from_data_list([to_pyg_data(first), to_pyg_data(second)])
    assert batch.x.shape[0] == len(first.node_ids) + len(second.node_ids)
    assert batch.edge_attr.shape[1] == 11


def test_agent_masks_and_roundtrip_joint_actions_use_only_owned_parameters():
    env = UnifiedNetworkEnvironment(); snapshot, _ = env.reset(_scenario(), seed=3); graph = snapshot_to_graph(snapshot); masks = action_masks(snapshot, graph)
    assert all(mask[0] for mask in masks.values())
    for agent, mask in masks.items():
        valid = next((index for index in range(1, mask.numel()) if mask[index]), 0)
        action = candidate_to_joint_action(snapshot, graph, {agent: valid})
        assert action.snapshot_id == snapshot.snapshot_id
        if valid:
            atomic = action.actions[0]
            allowed = {"ospf": {"ospf_weight"}, "bgp": {"local_preference", "as_path_length", "med"}, "performance": {"bandwidth_bps", "capacity_bps", "queue_packets"}}
            assert atomic.parameter_type in allowed[agent]
    assert candidate_to_joint_action(snapshot, graph, {"ospf": 0, "bgp": 0, "performance": 0}).actions == ()


def test_down_and_all_masked_leave_only_no_update():
    scenario = _scenario(); env = UnifiedNetworkEnvironment(); snapshot, _ = env.reset(scenario)
    actions = tuple()
    # Down every link using schema events/actions through successive snapshots.
    from netkeeper_sim.schemas import AtomicAction, JointAction
    for link in snapshot.topology.links:
        result = env.step(snapshot, JointAction((AtomicAction("baseline", "link_state", {"link_id": link.link_id}, "set", "down"),), snapshot_id=snapshot.snapshot_id)); snapshot = result.next_snapshot
    graph = snapshot_to_graph(snapshot); masks = action_masks(snapshot, graph)
    assert masks["ospf"].sum().item() == 1 and masks["performance"].sum().item() == 1
    assert candidate_to_joint_action(snapshot, graph, {"ospf": 1, "performance": 1}).actions == ()


def test_single_candidate_mask_semantics_normalize_and_never_nan():
    logits = torch.tensor([[2.0, 99.0, -4.0], [1.0, 2.0, 3.0]])
    probabilities, log_probs = masked_policy(logits, torch.tensor([[True, False, True], [False, False, False]]))
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2))
    assert probabilities[0, 1] == 0 and probabilities[1, 0] == 1
    assert torch.isfinite(probabilities).all() and torch.isfinite(masked_entropy(logits, torch.tensor([[True, False, True], [False, False, False]]))).all()
    assert sample_candidates(logits, torch.tensor([[True, False, True], [False, False, False]]), greedy=True).tolist() == [0, 0]


def test_unified_env_reset_step_reward_and_lifecycle_are_replayable():
    scenario = _scenario("train", 2)
    env = MultiAgentNetworkEnvironment(scenario=scenario, seed=11, reward_config=RewardConfig(include_traffic_shift=False))
    graph, observation, masks = env.reset(seed=11)
    outcome = env.step({"ospf": 0, "bgp": 0, "performance": 0})
    reward = outcome.result.rewards
    assert outcome.result.previous_snapshot_id == graph.snapshot_id
    assert reward.traffic_shift_reward == 0.0
    assert reward.total_reward == pytest.approx(sum((reward.policy_reward, reward.mlu_reward, reward.traffic_shift_reward, reward.configuration_change_penalty, reward.illegal_action_penalty, reward.dropped_traffic_penalty)))
    assert not outcome.result.terminated or scenario.target_mlu is not None
    assert outcome.result.truncated is False
    again = UnifiedNetworkEnvironment(RewardConfig(include_traffic_shift=False)); old, _ = again.reset(scenario, seed=11); replay = again.step(old, candidate_to_joint_action(old, snapshot_to_graph(old), {"ospf": 0, "bgp": 0, "performance": 0}))
    assert replay.next_snapshot.snapshot_id == outcome.result.next_snapshot.snapshot_id


def test_reward_weights_have_bounded_directions_and_shift_switch():
    env = UnifiedNetworkEnvironment(); previous, _ = env.reset(_scenario())
    before = replace(previous, metrics=Metrics(policy_consistency=0.25, maximum_link_utilization=0.80, total_input_bps=100.0, dropped_bps=0.0))
    after = replace(previous, metrics=Metrics(policy_consistency=0.75, maximum_link_utilization=0.30, traffic_shift_step_project_v1=0.20, total_input_bps=100.0, dropped_bps=10.0))
    weighted = _reward(before, after, {"x": (1, 2), "y": (1, 2)}, 2, RewardConfig())
    assert weighted.policy_reward > 0 and weighted.mlu_reward > 0
    assert weighted.traffic_shift_reward < 0 and weighted.configuration_change_penalty < 0
    assert weighted.illegal_action_penalty < 0 and weighted.dropped_traffic_penalty < 0
    without_shift = _reward(before, after, {}, 0, RewardConfig(include_traffic_shift=False))
    assert without_shift.traffic_shift_reward == 0.0


def test_random_fixed_policy_smoke_on_multiple_train_topologies():
    seen = set(); sampler = BalancedScenarioSampler(ROOT, "train", seed=4)
    for _ in range(40):
        scenario = sampler.next_scenario()
        if scenario.topology.topology_id in seen: continue
        seen.add(scenario.topology.topology_id)
        env = MultiAgentNetworkEnvironment(scenario=scenario, seed=4); graph, _, _ = env.reset()
        result = env.step({"ospf": 0, "bgp": 0, "performance": 0}).result
        assert torch.isfinite(torch.tensor(result.rewards.total_reward))
        if len(seen) >= 3: break
    assert len(seen) == 3
