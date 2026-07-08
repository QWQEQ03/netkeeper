from __future__ import annotations

import torch
from torch import nn

from netkeeper_sim.rl.config import GraphNetworkConfig
from netkeeper_sim.rl.types import EncoderOutput

try:
    from torch_geometric.nn import GINEConv, TransformerConv
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard.
    raise ModuleNotFoundError(
        "SharedGraphTransformerEncoder requires torch-geometric. "
        "Install the RL extra in a compatible training environment."
    ) from exc


class SharedGraphTransformerEncoder(nn.Module):
    """Shared graph encoder for decentralized actors.

    This is an engineering approximation of the paper's GraphTrans block:
    edge-aware GINE layers provide GCN-like local message passing, followed by
    edge-aware TransformerConv layers and explicit masked mean graph pooling.
    """

    def __init__(self, config: GraphNetworkConfig) -> None:
        super().__init__()
        if config.hidden_dim % config.transformer_heads != 0:
            raise ValueError("hidden_dim must be divisible by transformer_heads")
        if config.pooling != "mean":
            raise ValueError("Only mean pooling is currently supported")

        self.config = config
        hidden_dim = config.hidden_dim
        self.node_projection = nn.Linear(config.node_feature_dim, hidden_dim)
        self.edge_projection = nn.Linear(config.edge_feature_dim, hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.gcn_layers = nn.ModuleList(
            GINEConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                edge_dim=hidden_dim,
            )
            for _ in range(config.gcn_layers)
        )
        self.gcn_norms = nn.ModuleList(
            nn.LayerNorm(hidden_dim) if config.layer_norm else nn.Identity()
            for _ in range(config.gcn_layers)
        )
        self.transformer_layers = nn.ModuleList(
            TransformerConv(
                hidden_dim,
                hidden_dim // config.transformer_heads,
                heads=config.transformer_heads,
                concat=True,
                edge_dim=hidden_dim,
                dropout=config.dropout,
                beta=True,
            )
            for _ in range(config.transformer_layers)
        )
        self.transformer_norms = nn.ModuleList(
            nn.LayerNorm(hidden_dim) if config.layer_norm else nn.Identity()
            for _ in range(config.transformer_layers)
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        batch: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> EncoderOutput:
        if batch is None:
            batch = torch.zeros(node_features.size(0), dtype=torch.long, device=node_features.device)
        if node_mask is None:
            node_mask = torch.ones(node_features.size(0), dtype=torch.bool, device=node_features.device)
        else:
            node_mask = node_mask.to(device=node_features.device, dtype=torch.bool)

        x = self.node_projection(node_features.float())
        edge_attr = self.edge_projection(edge_features.float())

        for layer, norm in zip(self.gcn_layers, self.gcn_norms):
            residual = x
            x = layer(x, edge_index.long(), edge_attr)
            x = norm(residual + self.dropout(torch.relu(x)))

        for layer, norm in zip(self.transformer_layers, self.transformer_norms):
            residual = x
            x = layer(x, edge_index.long(), edge_attr)
            x = norm(residual + self.dropout(torch.relu(x)))

        x = x * node_mask.unsqueeze(-1)
        graph_embedding = _masked_mean_pool(x, batch.long(), node_mask)
        return EncoderOutput(node_embeddings=x, graph_embedding=graph_embedding)


def _masked_mean_pool(
    node_embeddings: torch.Tensor,
    batch: torch.Tensor,
    node_mask: torch.Tensor,
) -> torch.Tensor:
    if node_embeddings.numel() == 0:
        return node_embeddings.new_zeros((0, node_embeddings.size(-1)))
    batch_size = int(batch.max().item()) + 1
    masked = node_embeddings * node_mask.unsqueeze(-1)
    pooled = node_embeddings.new_zeros((batch_size, node_embeddings.size(-1)))
    pooled.index_add_(0, batch, masked)
    counts = node_embeddings.new_zeros((batch_size, 1))
    counts.index_add_(0, batch, node_mask.float().unsqueeze(-1))
    return pooled / counts.clamp_min(1.0)
