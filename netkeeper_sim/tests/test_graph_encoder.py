from __future__ import annotations
import pytest
pytestmark = pytest.mark.skip(reason="legacy graph-state encoder contract is superseded")

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from netkeeper_sim.rl import MultiAgentNetworkEnvironment, RLConfig
from netkeeper_sim.rl.config import GraphNetworkConfig
from netkeeper_sim.rl.networks import SharedGraphTransformerEncoder


def test_encoder_single_graph_forward(diamond_topology):
    state, _ = MultiAgentNetworkEnvironment(topology=diamond_topology).reset()
    encoder = SharedGraphTransformerEncoder(
        GraphNetworkConfig.debug(
            node_feature_dim=state.node_features.shape[1],
            edge_feature_dim=state.edge_features.shape[1],
        )
    )

    output = encoder(state.node_features, state.edge_index, state.edge_features)

    assert output.node_embeddings.shape == (4, 64)
    assert output.graph_embedding.shape == (1, 64)


def test_encoder_batch_multiple_graphs(single_path_topology, diamond_topology):
    first, _ = MultiAgentNetworkEnvironment(topology=single_path_topology).reset()
    second, _ = MultiAgentNetworkEnvironment(topology=diamond_topology).reset()
    node_features = torch.cat((first.node_features, second.node_features), dim=0)
    edge_features = torch.cat((first.edge_features, second.edge_features), dim=0)
    second_edge_index = second.edge_index + first.node_features.shape[0]
    edge_index = torch.cat((first.edge_index, second_edge_index), dim=1)
    batch = torch.cat(
        (
            torch.zeros(first.node_features.shape[0], dtype=torch.long),
            torch.ones(second.node_features.shape[0], dtype=torch.long),
        )
    )
    encoder = SharedGraphTransformerEncoder(
        GraphNetworkConfig.debug(
            node_feature_dim=first.node_features.shape[1],
            edge_feature_dim=first.edge_features.shape[1],
        )
    )

    output = encoder(node_features, edge_index, edge_features, batch=batch)

    assert output.node_embeddings.shape == (7, 64)
    assert output.graph_embedding.shape == (2, 64)


def test_encoder_handles_different_node_counts(single_path_topology, diamond_topology):
    config = RLConfig()
    encoder = SharedGraphTransformerEncoder(
        GraphNetworkConfig.debug(
            node_feature_dim=len(config.node_feature_names),
            edge_feature_dim=len(config.edge_feature_names),
        )
    )
    first, _ = MultiAgentNetworkEnvironment(topology=single_path_topology).reset()
    second, _ = MultiAgentNetworkEnvironment(topology=diamond_topology).reset()

    first_output = encoder(first.node_features, first.edge_index, first.edge_features)
    second_output = encoder(second.node_features, second.edge_index, second.edge_features)

    assert first_output.node_embeddings.shape[0] == 3
    assert second_output.node_embeddings.shape[0] == 4


def test_graph_embedding_is_permutation_invariant(diamond_topology):
    state, _ = MultiAgentNetworkEnvironment(topology=diamond_topology).reset(seed=3)
    encoder = SharedGraphTransformerEncoder(
        GraphNetworkConfig.debug(
            node_feature_dim=state.node_features.shape[1],
            edge_feature_dim=state.edge_features.shape[1],
        )
    )
    encoder.eval()
    original = encoder(state.node_features, state.edge_index, state.edge_features).graph_embedding

    permutation = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    permuted_nodes = state.node_features[permutation]
    permuted_edges = inverse[state.edge_index]
    permuted = encoder(permuted_nodes, permuted_edges, state.edge_features).graph_embedding

    assert torch.allclose(original, permuted, atol=1e-5)
