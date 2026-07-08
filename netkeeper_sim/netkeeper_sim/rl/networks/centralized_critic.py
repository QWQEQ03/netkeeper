from __future__ import annotations

from typing import Any

import torch
from torch import nn

from netkeeper_sim.rl.config import GraphNetworkConfig
from netkeeper_sim.rl.networks.actor_head import apply_entity_mask
from netkeeper_sim.rl.networks.embeddings import ParameterIDEmbedding
from netkeeper_sim.rl.networks.graph_encoder import SharedGraphTransformerEncoder
from netkeeper_sim.rl.networks.multi_agent_actor import _as_tensor
from netkeeper_sim.rl.types import ActorLogits


AGENT_TO_ID = {"ospf": 0, "bgp": 1, "performance": 2}


class CentralizedCritic(nn.Module):
    """Centralized critic that returns Q-values for all candidate actions."""

    def __init__(
        self,
        config: GraphNetworkConfig,
        action_count: int = 64,
        hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = SharedGraphTransformerEncoder(config)
        self.parameter_embedding = ParameterIDEmbedding(config.hidden_dim)
        self.agent_embedding = nn.Embedding(len(AGENT_TO_ID), config.hidden_dim)
        self.action_count = action_count
        input_dim = (3 * config.hidden_dim) + 6
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(max(1, hidden_layers)):
            layers.extend((nn.Linear(current_dim, config.hidden_dim), nn.ReLU()))
            current_dim = config.hidden_dim
        layers.append(nn.Linear(current_dim, action_count))
        self.q_head = nn.Sequential(*layers)

    def forward_state(
        self,
        state: Any,
        agent: str,
        parameter: str,
        current_actions: torch.Tensor | None = None,
        previous_actions: torch.Tensor | None = None,
    ) -> ActorLogits:
        device = next(self.parameters()).device
        encoding = self.encoder(
            node_features=_as_tensor(state.node_features, device, torch.float32),
            edge_index=_as_tensor(state.edge_index, device, torch.long),
            edge_features=_as_tensor(state.edge_features, device, torch.float32),
            node_mask=_as_tensor(state.node_mask, device, torch.bool),
        )
        mask = _as_tensor(state.parameter_masks[f"{parameter}_mask"], device, torch.bool).reshape(-1)
        q_values = self.forward_from_graph_embedding(
            graph_embedding=encoding.graph_embedding[0],
            agent=agent,
            parameter=parameter,
            entity_mask=mask,
            current_actions=current_actions,
            previous_actions=previous_actions,
        )
        return ActorLogits({parameter: q_values})

    def forward_from_graph_embedding(
        self,
        graph_embedding: torch.Tensor,
        agent: str,
        parameter: str,
        entity_mask: torch.Tensor,
        current_actions: torch.Tensor | None = None,
        previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = torch.as_tensor(entity_mask, device=graph_embedding.device, dtype=torch.bool).reshape(-1)
        entity_count = mask.numel()
        if entity_count == 0:
            return graph_embedding.new_zeros((0, self.action_count))
        agent_ids = torch.full(
            (entity_count,),
            AGENT_TO_ID[agent],
            dtype=torch.long,
            device=graph_embedding.device,
        )
        current_summary = _action_summary(current_actions, entity_count, graph_embedding.device)
        previous_summary = _action_summary(previous_actions, entity_count, graph_embedding.device)
        features = torch.cat(
            (
                graph_embedding.unsqueeze(0).expand(entity_count, -1),
                self.agent_embedding(agent_ids),
                self.parameter_embedding(parameter, entity_count, graph_embedding.device),
                current_summary,
                previous_summary,
            ),
            dim=-1,
        )
        return apply_entity_mask(self.q_head(features), mask, parameter)


def _action_summary(
    actions: torch.Tensor | None,
    entity_count: int,
    device: torch.device,
) -> torch.Tensor:
    if actions is None:
        return torch.zeros((entity_count, 3), dtype=torch.float32, device=device)
    tensor = torch.as_tensor(actions, dtype=torch.float32, device=device).reshape(-1)
    if tensor.numel() < entity_count:
        padded = torch.full((entity_count,), -1.0, dtype=torch.float32, device=device)
        padded[: tensor.numel()] = tensor
        tensor = padded
    else:
        tensor = tensor[:entity_count]
    valid = tensor >= 0
    normalized = torch.where(valid, tensor / 63.0, torch.zeros_like(tensor))
    return torch.stack((normalized, valid.float(), torch.ones_like(normalized)), dim=-1)
