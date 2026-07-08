from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from netkeeper_sim.rl.config import GraphNetworkConfig
from netkeeper_sim.rl.networks.actor_head import FactorizedParameterHead, apply_entity_mask
from netkeeper_sim.rl.networks.embeddings import ParameterIDEmbedding
from netkeeper_sim.rl.networks.graph_encoder import SharedGraphTransformerEncoder
from netkeeper_sim.rl.types import ActorLogits, EncoderOutput, MultiAgentActorOutput


class OSPFActor(nn.Module):
    def __init__(
        self,
        encoder: SharedGraphTransformerEncoder,
        parameter_embedding: ParameterIDEmbedding,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.parameter_embedding = parameter_embedding
        hidden_dim = encoder.config.hidden_dim
        self.ospf_weight_head = FactorizedParameterHead(input_dim=(2 * hidden_dim) + 5)

    def forward_from_encoding(
        self,
        encoding: EncoderOutput,
        policy_observation: torch.Tensor,
        utilization_observation: torch.Tensor,
        masks: Any,
    ) -> ActorLogits:
        graph_embedding = _single_graph_embedding(encoding)
        link_count = _mask_tensor(_get_mask(masks, "ospf_weight_mask"), graph_embedding.device).numel()
        target_features = _link_target_features(
            graph_embedding,
            _summary(policy_observation, graph_embedding.device),
            utilization_observation,
            self.parameter_embedding("ospf_weight", link_count, graph_embedding.device),
        )
        logits = self.ospf_weight_head(target_features)
        logits = apply_entity_mask(logits, _get_mask(masks, "ospf_weight_mask"), "ospf_weight")
        return ActorLogits({"ospf_weight": logits})


class BGPActor(nn.Module):
    def __init__(
        self,
        encoder: SharedGraphTransformerEncoder,
        parameter_embedding: ParameterIDEmbedding,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.parameter_embedding = parameter_embedding
        hidden_dim = encoder.config.hidden_dim
        input_dim = (2 * hidden_dim) + 4
        self.local_preference_head = FactorizedParameterHead(input_dim=input_dim)
        self.as_path_length_head = FactorizedParameterHead(input_dim=input_dim)
        self.med_head = FactorizedParameterHead(input_dim=input_dim)

    def forward_from_encoding(
        self,
        encoding: EncoderOutput,
        policy_observation: torch.Tensor,
        masks: Any,
    ) -> ActorLogits:
        graph_embedding = _single_graph_embedding(encoding)
        route_count = _mask_tensor(_get_mask(masks, "local_preference_mask"), graph_embedding.device).numel()
        policy_summary = _summary(policy_observation, graph_embedding.device)
        logits: dict[str, torch.Tensor] = {}
        for name, head in (
            ("local_preference", self.local_preference_head),
            ("as_path_length", self.as_path_length_head),
            ("med", self.med_head),
        ):
            target_features = _route_target_features(
                graph_embedding,
                policy_summary,
                self.parameter_embedding(name, route_count, graph_embedding.device),
            )
            logits[name] = apply_entity_mask(head(target_features), _get_mask(masks, f"{name}_mask"), name)
        return ActorLogits(logits)


class PerformanceActor(nn.Module):
    def __init__(
        self,
        encoder: SharedGraphTransformerEncoder,
        parameter_embedding: ParameterIDEmbedding,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.parameter_embedding = parameter_embedding
        hidden_dim = encoder.config.hidden_dim
        input_dim = (2 * hidden_dim) + 1
        self.bandwidth_head = FactorizedParameterHead(input_dim=input_dim)
        self.capacity_head = FactorizedParameterHead(input_dim=input_dim)
        self.queue_length_head = FactorizedParameterHead(input_dim=input_dim)

    def forward_from_encoding(
        self,
        encoding: EncoderOutput,
        utilization_observation: torch.Tensor,
        masks: Any,
    ) -> ActorLogits:
        graph_embedding = _single_graph_embedding(encoding)
        link_count = _mask_tensor(_get_mask(masks, "bandwidth_mask"), graph_embedding.device).numel()
        logits: dict[str, torch.Tensor] = {}
        for name, head in (
            ("bandwidth", self.bandwidth_head),
            ("capacity", self.capacity_head),
            ("queue_length", self.queue_length_head),
        ):
            target_features = _link_target_features(
                graph_embedding,
                None,
                utilization_observation,
                self.parameter_embedding(name, link_count, graph_embedding.device),
            )
            logits[name] = apply_entity_mask(head(target_features), _get_mask(masks, f"{name}_mask"), name)
        return ActorLogits(logits)


class MultiAgentActor(nn.Module):
    def __init__(self, config: GraphNetworkConfig) -> None:
        super().__init__()
        self.encoder = SharedGraphTransformerEncoder(config)
        self.parameter_embedding = ParameterIDEmbedding(config.hidden_dim)
        self.ospf_actor = OSPFActor(self.encoder, self.parameter_embedding)
        self.bgp_actor = BGPActor(self.encoder, self.parameter_embedding)
        self.performance_actor = PerformanceActor(self.encoder, self.parameter_embedding)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        policy_observation: torch.Tensor,
        utilization_observation: torch.Tensor,
        masks: Any,
        batch: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> MultiAgentActorOutput:
        encoding = self.encoder(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            batch=batch,
            node_mask=node_mask,
        )
        return MultiAgentActorOutput(
            encoder_output=encoding,
            ospf=self.ospf_actor.forward_from_encoding(
                encoding,
                policy_observation,
                utilization_observation,
                masks,
            ),
            bgp=self.bgp_actor.forward_from_encoding(
                encoding,
                policy_observation,
                masks,
            ),
            performance=self.performance_actor.forward_from_encoding(
                encoding,
                utilization_observation,
                masks,
            ),
        )

    def forward_state(self, state: Any) -> MultiAgentActorOutput:
        device = next(self.parameters()).device
        return self.forward(
            node_features=_as_tensor(state.node_features, device, torch.float32),
            edge_index=_as_tensor(state.edge_index, device, torch.long),
            edge_features=_as_tensor(state.edge_features, device, torch.float32),
            policy_observation=_as_tensor(state.policy_observation, device, torch.float32),
            utilization_observation=_as_tensor(state.utilization_observation, device, torch.float32),
            masks=state.parameter_masks,
            node_mask=_as_tensor(state.node_mask, device, torch.bool),
        )


def _single_graph_embedding(encoding: EncoderOutput) -> torch.Tensor:
    if encoding.graph_embedding.size(0) != 1:
        raise ValueError("Actor heads currently expect a single graph embedding")
    return encoding.graph_embedding[0]


def _summary(values: torch.Tensor, device: torch.device) -> torch.Tensor:
    tensor = _as_tensor(values, device, torch.float32).reshape(-1)
    if tensor.numel() == 0:
        return torch.zeros((4,), dtype=torch.float32, device=device)
    return torch.stack(
        (
            tensor.mean(),
            tensor.min(),
            tensor.max(),
            torch.tensor(min(1.0, tensor.numel() / 100.0), dtype=torch.float32, device=device),
        )
    )


def _link_target_features(
    graph_embedding: torch.Tensor,
    policy_summary: torch.Tensor | None,
    utilization_observation: torch.Tensor,
    parameter_embedding: torch.Tensor,
) -> torch.Tensor:
    count = parameter_embedding.size(0)
    if count == 0:
        feature_dim = graph_embedding.numel() + parameter_embedding.size(-1) + 1
        if policy_summary is not None:
            feature_dim += policy_summary.numel()
        return graph_embedding.new_zeros((0, feature_dim))
    utilization = _as_tensor(utilization_observation, graph_embedding.device, torch.float32).reshape(-1)
    if utilization.numel() < count:
        padded = graph_embedding.new_zeros((count,))
        padded[: utilization.numel()] = utilization
        utilization = padded
    else:
        utilization = utilization[:count]
    pieces = [
        graph_embedding.unsqueeze(0).expand(count, -1),
        parameter_embedding,
        utilization.unsqueeze(-1),
    ]
    if policy_summary is not None:
        pieces.insert(1, policy_summary.unsqueeze(0).expand(count, -1))
    return torch.cat(pieces, dim=-1)


def _route_target_features(
    graph_embedding: torch.Tensor,
    policy_summary: torch.Tensor,
    parameter_embedding: torch.Tensor,
) -> torch.Tensor:
    count = parameter_embedding.size(0)
    if count == 0:
        return graph_embedding.new_zeros((0, (2 * graph_embedding.numel()) + policy_summary.numel()))
    return torch.cat(
        (
            graph_embedding.unsqueeze(0).expand(count, -1),
            policy_summary.unsqueeze(0).expand(count, -1),
            parameter_embedding,
        ),
        dim=-1,
    )


def _get_mask(masks: Any, name: str) -> Any:
    if isinstance(masks, Mapping):
        return masks[name]
    return getattr(masks, name)


def _mask_tensor(mask: Any, device: torch.device) -> torch.Tensor:
    return _as_tensor(mask, device, torch.bool)


def _as_tensor(value: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, dtype=dtype, device=device)
