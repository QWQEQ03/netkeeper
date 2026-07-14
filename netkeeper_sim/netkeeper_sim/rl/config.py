from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


RewardMode = Literal["paper", "normalized"]
GraphPooling = Literal["mean"]


@dataclass(frozen=True)
class ParameterRange:
    minimum: int
    maximum: int

    def contains(self, value: int) -> bool:
        return self.minimum <= value <= self.maximum


@dataclass(frozen=True)
class RLConfig:
    """Configuration for the lightweight multi-agent RL adapter."""

    max_steps: int = 200
    reward_k: float = 10.0
    reward_mode: RewardMode = "normalized"
    maximum_link_utilization_normalizer: float = 1.0

    ospf_weight: ParameterRange = ParameterRange(1, 64)
    local_preference: ParameterRange = ParameterRange(1, 64)
    as_path_length: ParameterRange = ParameterRange(1, 64)
    med: ParameterRange = ParameterRange(1, 64)
    bandwidth: ParameterRange = ParameterRange(65, 128)
    capacity: ParameterRange = ParameterRange(65, 128)
    queue_length: ParameterRange = ParameterRange(65, 128)

    node_feature_names: tuple[str, ...] = (
        "type_router",
        "type_prefix",
        "type_external_as",
        "type_policy",
        "type_other",
        "avg_ospf_weight",
        "avg_local_preference",
        "avg_as_path_length",
        "avg_med",
        "avg_bandwidth",
        "avg_capacity",
        "avg_queue_length",
        "avg_loss_rate",
        "active_ratio",
        "degree",
        "is_bgp_speaker",
        "is_policy_endpoint",
    )

    edge_feature_names: tuple[str, ...] = (
        "edge_type_link",
        "edge_type_policy",
        "edge_type_bgp",
        "edge_type_other",
        "ospf_weight",
        "bandwidth",
        "capacity",
        "queue_length",
        "loss_rate",
        "active",
        "utilization",
    )


@dataclass(frozen=True)
class GraphNetworkConfig:
    node_feature_dim: int
    edge_feature_dim: int
    hidden_dim: int = 64
    gcn_layers: int = 2
    transformer_layers: int = 2
    transformer_heads: int = 4
    dropout: float = 0.1
    layer_norm: bool = True
    pooling: GraphPooling = "mean"

    @classmethod
    def debug(
        cls,
        node_feature_dim: int,
        edge_feature_dim: int,
        dropout: float = 0.0,
    ) -> "GraphNetworkConfig":
        return cls(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=32,
            gcn_layers=2,
            transformer_layers=1,
            transformer_heads=2,
            dropout=dropout,
        )

    @classmethod
    def paper(
        cls,
        node_feature_dim: int,
        edge_feature_dim: int,
        dropout: float = 0.0,
    ) -> "GraphNetworkConfig":
        return cls(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=128,
            gcn_layers=8,
            transformer_layers=8,
            transformer_heads=8,
            dropout=dropout,
        )


@dataclass(frozen=True)
class COMATrainingConfig:
    episodes: int = 10
    max_steps: int = 200
    replay_buffer_size: int = 100
    batch_size: int = 16
    gamma: float = 0.85
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.99
    target_update_interval: int = 16
    target_update_tau: float | None = None
    actor_lr: float = 0.0001
    critic_lr: float = 0.0002
    gradient_clip_norm: float = 1.0
    loss: Literal["huber", "mse"] = "huber"
    reward_normalization: bool = False
    deterministic_seed: int = 42
    device: Literal["auto", "cpu", "cuda"] = "auto"
    hidden_dim: int = 128
    gcn_layers: int = 8
    transformer_layers: int = 8
    transformer_heads: int = 8
    decoder_linear_layers: int = 4

    @classmethod
    def debug(cls) -> "COMATrainingConfig":
        return cls(
            episodes=2,
            max_steps=5,
            replay_buffer_size=100,
            batch_size=2,
            hidden_dim=64,
            gcn_layers=2,
            transformer_layers=2,
            transformer_heads=4,
            decoder_linear_layers=2,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "COMATrainingConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            data: dict[str, Any] = yaml.safe_load(handle) or {}
        return cls(**data.get("training", data))

    def graph_config(self, node_feature_dim: int, edge_feature_dim: int) -> GraphNetworkConfig:
        return GraphNetworkConfig(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=self.hidden_dim,
            gcn_layers=self.gcn_layers,
            transformer_layers=self.transformer_layers,
            transformer_heads=self.transformer_heads,
        )
