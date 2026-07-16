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
        self.link_projection = nn.Linear(h + config.edge_feature_dim, h)
        self.route_projection = nn.Linear(h + 4, h)
        self.agent_embedding = nn.Embedding(3, h)
        self.action_projection = nn.Sequential(nn.Linear(6, h), nn.ReLU(), nn.Linear(h, h))
        self.heads = nn.ModuleDict({agent: CandidateHead(h * 2, len(PARAMETERS[agent])) for agent in AGENTS})

    def forward_graph(self, graph: SnapshotGraph, joint_actions: torch.Tensor, agent: str) -> torch.Tensor:
        device = next(self.parameters()).device
        encoded = self.encoder(graph.node_features.to(device), graph.edge_index.to(device), graph.edge_features.to(device), node_mask=graph.node_mask.to(device))
        node = encoded.node_embeddings; index = {name: pos for pos, name in enumerate(graph.node_ids)}
        link_base = torch.stack([(node[index[_endpoints(graph, x)[0]]] + node[index[_endpoints(graph, x)[1]]]) / 2 for x in graph.link_ids]) if graph.link_ids else node.new_zeros((0, node.size(-1)))
        link_raw=torch.stack([graph.edge_features[next(i for i,name in enumerate(graph.edge_ids) if name.startswith(f"{link_id}:"))] for link_id in graph.link_ids]).to(device) if graph.link_ids else node.new_zeros((0,graph.edge_features.size(-1)))
        links=self.link_projection(torch.cat((link_base,link_raw),dim=-1)) if graph.link_ids else link_base
        route_base = torch.stack([(node[index[x.router_id]] + node[index[x.next_hop]]) / 2 for x in graph.route_targets]) if graph.route_targets else node.new_zeros((0, node.size(-1)))
        route_raw=graph.local_observations["bgp"]["route_features"].to(device)
        routes=self.route_projection(torch.cat((route_base,route_raw),dim=-1)) if graph.route_targets else route_base
        actions = torch.as_tensor(joint_actions, dtype=torch.long, device=device).reshape(3)
        # COMA's counterfactual critic must condition on u_-i only.  Encoding
        # the queried agent's sampled action here would leak a_i into both the
        # context and candidate axis and invalidate the counterfactual baseline.
        context = _joint_action_features(graph, actions, excluded_agent=agent)
        context = self.action_projection(context)
        agent_id = AGENTS.index(agent)
        query = torch.cat((encoded.graph_embedding[0], context + self.agent_embedding.weight[agent_id]), dim=-1)
        entities = links if agent != "bgp" else routes
        # CandidateHead expects the same dimensionality for graph and entities.
        augmented_entities = torch.cat((entities, torch.zeros_like(entities)), dim=-1)
        return self.heads[agent](query, augmented_entities)


def _joint_action_features(graph: SnapshotGraph, actions: torch.Tensor, *, excluded_agent: str) -> torch.Tensor:
    """Return topology-relative semantic features for the other agents' actions.

    Candidate IDs are categorical and topology dependent, so treating the raw
    integer as a continuous magnitude is misleading.  Each action is instead
    represented by normalized entity position and parameter/value position.
    The queried agent is represented by zeros (omitted context).
    """
    rows = []
    for index, name in enumerate(AGENTS):
        candidate = int(actions[index].item())
        if name == excluded_agent or candidate == 0:
            rows.extend((0.0, 0.0))
            continue
        parameter_count = len(PARAMETERS[name])
        entity_count = len(graph.route_targets) if name == "bgp" else len(graph.link_ids)
        offset = candidate - 1
        value_index = offset % ACTION_VALUES
        group = offset // ACTION_VALUES
        parameter_index = group % parameter_count
        entity_index = group // parameter_count
        entity_feature = (entity_index + 1) / max(entity_count, 1)
        parameter_value_feature = (parameter_index + (value_index + 1) / ACTION_VALUES) / parameter_count
        rows.extend((entity_feature, parameter_value_feature))
    return actions.new_tensor(rows, dtype=torch.float32)
