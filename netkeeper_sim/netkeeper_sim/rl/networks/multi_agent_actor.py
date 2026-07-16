"""Shared-encoder, entity-aware actors for unified macro candidates."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn

from netkeeper_sim.rl.schema_adapter import ACTION_VALUES, PARAMETERS, SnapshotGraph
from netkeeper_sim.rl.networks.graph_encoder import SharedGraphTransformerEncoder
from netkeeper_sim.rl.config import GraphNetworkConfig

AGENTS = ("ospf", "bgp", "performance")

@dataclass(frozen=True)
class ActorOutput:
    logits: dict[str, torch.Tensor]  # each [1 + entity_count * parameter_count * 64]
    masks: dict[str, torch.Tensor]
    graph_embedding: torch.Tensor


class CandidateHead(nn.Module):
    """Two-layer head; entity embeddings make different links/routes differ."""
    def __init__(self, hidden: int, parameters: int) -> None:
        super().__init__()
        self.no_update = nn.Linear(hidden, 1)
        self.parameter = nn.Embedding(parameters, hidden)
        self.value = nn.Embedding(ACTION_VALUES, hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, graph: torch.Tensor, entities: torch.Tensor) -> torch.Tensor:
        no_op = self.no_update(graph).reshape(1)
        if entities.numel() == 0: return no_op
        count, hidden = entities.shape
        params = self.parameter.weight
        values = self.value.weight
        item = torch.cat((entities[:, None, None, :].expand(count, params.size(0), ACTION_VALUES, hidden),
                          params[None, :, None, :].expand(count, -1, ACTION_VALUES, -1),
                          values[None, None, :, :].expand(count, params.size(0), -1, -1)), dim=-1)
        return torch.cat((no_op, self.mlp(item).reshape(-1)), dim=0)


class MultiAgentActor(nn.Module):
    def __init__(self, config: GraphNetworkConfig) -> None:
        super().__init__()
        self.encoder = SharedGraphTransformerEncoder(config)
        h = config.hidden_dim
        self.link_projection = nn.Linear(h + config.edge_feature_dim, h)
        self.route_projection = nn.Linear(h + 4, h)
        self.heads = nn.ModuleDict({agent: CandidateHead(h, len(PARAMETERS[agent])) for agent in AGENTS})

    def forward_graph(self, graph: SnapshotGraph, masks: dict[str, torch.Tensor]) -> ActorOutput:
        device = next(self.parameters()).device
        encoded = self.encoder(graph.node_features.to(device), graph.edge_index.to(device), graph.edge_features.to(device), node_mask=graph.node_mask.to(device))
        node = encoded.node_embeddings
        index = {name: pos for pos, name in enumerate(graph.node_ids)}
        link_base = torch.stack([(node[index[_endpoints(graph, link_id)[0]]] + node[index[_endpoints(graph, link_id)[1]]]) / 2 for link_id in graph.link_ids]) if graph.link_ids else node.new_zeros((0, node.size(-1)))
        link_raw=torch.stack([graph.edge_features[next(i for i,name in enumerate(graph.edge_ids) if name.startswith(f"{link_id}:"))] for link_id in graph.link_ids]).to(device) if graph.link_ids else node.new_zeros((0,graph.edge_features.size(-1)))
        link=self.link_projection(torch.cat((link_base,link_raw),dim=-1)) if graph.link_ids else link_base
        route_base = torch.stack([(node[index[item.router_id]] + node[index[item.next_hop]]) / 2 for item in graph.route_targets]) if graph.route_targets else node.new_zeros((0, node.size(-1)))
        route_raw=graph.local_observations["bgp"]["route_features"].to(device)
        route=self.route_projection(torch.cat((route_base,route_raw),dim=-1)) if graph.route_targets else route_base
        masks = {key: value.to(device) for key, value in masks.items()}
        logits={"ospf": self.heads["ospf"](encoded.graph_embedding[0], link), "bgp": self.heads["bgp"](encoded.graph_embedding[0], route), "performance": self.heads["performance"](encoded.graph_embedding[0], link)}
        logits={name:_cardinality_correct(values,masks[name]) for name,values in logits.items()}
        return ActorOutput(logits, masks, encoded.graph_embedding[0])


def _cardinality_correct(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Give the macro no-op prior mass comparable to all valid edits."""
    valid_non_noop=int(mask[1:].sum().item())
    if valid_non_noop<=0: return logits
    corrected=logits.clone(); corrected[0]=corrected[0]+math.log(valid_non_noop); return corrected


def _endpoints(graph: SnapshotGraph, link_id: str) -> tuple[str, str]:
    # edge IDs are stable `link:source->target`; use first directed occurrence.
    prefix = f"{link_id}:"
    edge = next(item for item in graph.edge_ids if item.startswith(prefix))
    values = edge[len(prefix):].split("->")
    return values[0], values[1]
