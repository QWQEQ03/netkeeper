from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from netkeeper_sim.rl import MultiAgentNetworkEnvironment
from netkeeper_sim.rl.config import GraphNetworkConfig
from netkeeper_sim.rl.networks import MultiAgentActor
from netkeeper_sim.routing.bgp import BGPRoute


def test_three_actors_share_encoder_object(single_path_topology):
    state = _state_with_bgp(single_path_topology)
    actor = _actor_for_state(state)

    assert actor.ospf_actor.encoder is actor.encoder
    assert actor.bgp_actor.encoder is actor.encoder
    assert actor.performance_actor.encoder is actor.encoder


def test_actor_heads_are_independent(single_path_topology):
    state = _state_with_bgp(single_path_topology)
    actor = _actor_for_state(state)

    ospf_params = {id(param) for param in actor.ospf_actor.ospf_weight_head.parameters()}
    bgp_params = {id(param) for param in actor.bgp_actor.local_preference_head.parameters()}
    perf_params = {id(param) for param in actor.performance_actor.capacity_head.parameters()}

    assert ospf_params.isdisjoint(bgp_params)
    assert ospf_params.isdisjoint(perf_params)
    assert bgp_params.isdisjoint(perf_params)


def test_gradient_flows_from_actor_head_to_shared_encoder(single_path_topology):
    state = _state_with_bgp(single_path_topology)
    actor = _actor_for_state(state)

    output = actor.forward_state(state)
    loss = (
        output.ospf.logits["ospf_weight"].sum()
        + output.bgp.logits["local_preference"].sum()
        + output.performance.logits["capacity"].sum()
    )
    loss.backward()

    gradient = actor.encoder.node_projection.weight.grad
    assert gradient is not None
    assert torch.any(gradient != 0)


def test_debug_configuration_runs_on_cpu(single_path_topology):
    state = _state_with_bgp(single_path_topology)
    actor = _actor_for_state(state).cpu()

    output = actor.forward_state(state)

    assert output.encoder_output.graph_embedding.device.type == "cpu"
    assert output.ospf.logits["ospf_weight"].device.type == "cpu"


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
