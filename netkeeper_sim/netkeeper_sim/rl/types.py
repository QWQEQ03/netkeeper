from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EncoderOutput:
    node_embeddings: torch.Tensor
    graph_embedding: torch.Tensor


@dataclass(frozen=True)
class ActorLogits:
    logits: dict[str, torch.Tensor]


@dataclass(frozen=True)
class MultiAgentActorOutput:
    encoder_output: EncoderOutput
    ospf: ActorLogits
    bgp: ActorLogits
    performance: ActorLogits
