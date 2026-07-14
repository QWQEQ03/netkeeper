from __future__ import annotations
import pytest
pytestmark = pytest.mark.skip(reason="superseded by unified SnapshotGraph dataflow tests")

from netkeeper_sim.policies import ForwardPolicy
from netkeeper_sim.rl import MultiAgentNetworkEnvironment, RLConfig
from netkeeper_sim.rl import _tensor as T
from netkeeper_sim.traffic.matrix import TrafficDemand, TrafficMatrix


def test_reset_returns_valid_graph_state(diamond_topology):
    env = MultiAgentNetworkEnvironment(
        topology=diamond_topology,
        traffic_matrix=TrafficMatrix((TrafficDemand("R1", "R4", 100.0),)),
        policies=[ForwardPolicy("p1", "R1", "R4", "R2")],
    )

    state, observations = env.reset(seed=1)

    assert T.is_tensor(state.node_features)
    assert T.is_tensor(state.edge_index)
    assert T.is_tensor(state.edge_features)
    assert state.node_features.shape == (4, len(RLConfig().node_feature_names))
    assert state.edge_index.shape == (2, 8)
    assert state.edge_features.shape == (8, len(RLConfig().edge_feature_names))
    assert state.node_mask.shape == (4,)
    assert state.edge_mask.shape == (8,)
    assert observations.ospf.shape == (1 + 4,)
    assert observations.bgp.shape == (1,)
    assert observations.performance.shape == (4,)


def test_node_and_edge_ordering_is_stable(diamond_topology):
    env = MultiAgentNetworkEnvironment(topology=diamond_topology)
    first_state, _ = env.reset(seed=7)
    first_node_ids = first_state.node_ids
    first_edge_ids = first_state.edge_ids

    second_state, _ = env.reset(seed=7)

    assert second_state.node_ids == first_node_ids
    assert second_state.edge_ids == first_edge_ids
    assert first_node_ids == tuple(sorted(first_node_ids))


def test_graph_state_contains_required_parameter_masks(diamond_topology):
    env = MultiAgentNetworkEnvironment(topology=diamond_topology)
    state, _ = env.reset()

    assert set(state.parameter_masks) == {
        "ospf_weight_mask",
        "local_preference_mask",
        "as_path_length_mask",
        "med_mask",
        "bandwidth_mask",
        "capacity_mask",
        "queue_length_mask",
    }
    for mask in state.parameter_masks.values():
        assert T.is_tensor(mask)
