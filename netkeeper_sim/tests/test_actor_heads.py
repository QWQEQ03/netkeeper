from __future__ import annotations
import pytest
pytestmark = pytest.mark.skip(reason="legacy per-entity action heads are superseded")

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from netkeeper_sim.rl import MultiAgentNetworkEnvironment
from netkeeper_sim.rl.config import GraphNetworkConfig
from netkeeper_sim.rl.networks import MultiAgentActor
from netkeeper_sim.rl.networks.actor_head import (
    argmax_masked_actions,
    masked_action_probabilities,
    sample_masked_actions,
)
from netkeeper_sim.routing.bgp import BGPRoute


def test_actor_logits_shapes(single_path_topology):
    state = _state_with_bgp(single_path_topology)
    actor = _actor_for_state(state)

    output = actor.forward_state(state)

    link_count = len(state.link_ids)
    route_count = len(state.bgp_route_targets)
    assert output.ospf.logits["ospf_weight"].shape == (link_count, 64)
    assert output.bgp.logits["local_preference"].shape == (route_count, 64)
    assert output.bgp.logits["as_path_length"].shape == (route_count, 64)
    assert output.bgp.logits["med"].shape == (route_count, 64)
    assert output.performance.logits["bandwidth"].shape == (link_count, 64)
    assert output.performance.logits["capacity"].shape == (link_count, 64)
    assert output.performance.logits["queue_length"].shape == (link_count, 64)


def test_action_mask_sets_invalid_entity_logits_to_small_value(single_path_topology):
    state = _state_with_bgp(single_path_topology)
    masks = {name: value.clone() for name, value in state.parameter_masks.items()}
    masks["ospf_weight_mask"][0] = False
    actor = _actor_for_state(state)

    output = actor(
        node_features=state.node_features,
        edge_index=state.edge_index,
        edge_features=state.edge_features,
        policy_observation=state.policy_observation,
        utilization_observation=state.utilization_observation,
        masks=masks,
        node_mask=state.node_mask,
    )

    assert torch.all(output.ospf.logits["ospf_weight"][0] < -1.0e8)


def test_invalid_action_probability_is_zero(single_path_topology):
    state = _state_with_bgp(single_path_topology)
    masks = {name: value.clone() for name, value in state.parameter_masks.items()}
    masks["capacity_mask"][0] = False
    actor = _actor_for_state(state)
    output = actor(
        state.node_features,
        state.edge_index,
        state.edge_features,
        state.policy_observation,
        state.utilization_observation,
        masks,
        node_mask=state.node_mask,
    )

    probabilities = masked_action_probabilities(
        output.performance.logits["capacity"],
        masks["capacity_mask"],
    )

    assert probabilities[0].sum().item() == 0.0


def test_sample_and_argmax_run(single_path_topology):
    state = _state_with_bgp(single_path_topology)
    actor = _actor_for_state(state)
    output = actor.forward_state(state)

    sampled = sample_masked_actions(
        output.ospf.logits["ospf_weight"],
        state.parameter_masks["ospf_weight_mask"],
    )
    greedy = argmax_masked_actions(
        output.ospf.logits["ospf_weight"],
        state.parameter_masks["ospf_weight_mask"],
    )

    assert sampled.shape == greedy.shape == (len(state.link_ids),)
    assert torch.all(sampled >= 0)
    assert torch.all(greedy >= 0)


def _state_with_bgp(topology):
    prefix = "203.0.113.0/24"
    candidates = {
        "R1": {
            prefix: [BGPRoute(prefix, "R2", 10, (65001,), 10, "R2", "peer")]
        }
    }
    state, _ = MultiAgentNetworkEnvironment(
        topology=topology,
        bgp_candidate_routes=candidates,
    ).reset()
    return state


def _actor_for_state(state):
    return MultiAgentActor(
        GraphNetworkConfig.debug(
            node_feature_dim=state.node_features.shape[1],
            edge_feature_dim=state.edge_features.shape[1],
        )
    )
