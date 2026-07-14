"""Independent centralized critic for COMA macro actions."""
from __future__ import annotations
import torch
from torch import nn

from netkeeper_sim.rl.config import GraphNetworkConfig
from netkeeper_sim.rl.schema_adapter import ACTION_VALUES, PARAMETERS, SnapshotGraph
from netkeeper_sim.rl.networks.graph_encoder import SharedGraphTransformerEncoder
from netkeeper_sim.rl.networks.multi_agent_actor import AGENTS, CandidateHead, _endpoints


class CentralizedCritic(nn.Module):
    """Q_i(s, u_-i, a_i) for all candidates a_i.

    `joint_actions` contains the actual rollout choices in ospf/bgp/performance
    order.  The queried agent's own choice is deliberately omitted from context;
    its candidate is supplied by the output axis.
    """
    def __init__(self, config: GraphNetworkConfig) -> None:
        super().__init__()
        self.encoder = SharedGraphTransformerEncoder(config)
        h = config.hidden_dim
        self.agent_embedding = nn.Embedding(3, h)
        self.action_projection = nn.Sequential(nn.Linear(6, h), nn.ReLU(), nn.Linear(h, h))
        self.heads = nn.ModuleDict({agent: CandidateHead(h * 2, len(PARAMETERS[agent])) for agent in AGENTS})

    def forward_graph(self, graph: SnapshotGraph, joint_actions: torch.Tensor, agent: str) -> torch.Tensor:
        device = next(self.parameters()).device
        encoded = self.encoder(graph.node_features.to(device), graph.edge_index.to(device), graph.edge_features.to(device), node_mask=graph.node_mask.to(device))
        node = encoded.node_embeddings; index = {name: pos for pos, name in enumerate(graph.node_ids)}
        links = torch.stack([(node[index[_endpoints(graph, x)[0]]] + node[index[_endpoints(graph, x)[1]]]) / 2 for x in graph.link_ids]) if graph.link_ids else node.new_zeros((0, node.size(-1)))
        routes = torch.stack([(node[index[x.router_id]] + node[index[x.next_hop]]) / 2 for x in graph.route_targets]) if graph.route_targets else node.new_zeros((0, node.size(-1)))
        actions = torch.as_tensor(joint_actions, dtype=torch.float32, device=device).reshape(3)
        context = torch.stack((actions / 20000.0, (actions == 0).float()), dim=-1).reshape(-1)
        context = self.action_projection(context)
        agent_id = AGENTS.index(agent)
        query = torch.cat((encoded.graph_embedding[0], context + self.agent_embedding.weight[agent_id]), dim=-1)
        entities = links if agent != "bgp" else routes
        # CandidateHead expects the same dimensionality for graph and entities.
        augmented_entities = torch.cat((entities, torch.zeros_like(entities)), dim=-1)
        return self.heads[agent](query, augmented_entities)
