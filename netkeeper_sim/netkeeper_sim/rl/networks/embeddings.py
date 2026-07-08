from __future__ import annotations

from enum import IntEnum

import torch
from torch import nn


class ParameterID(IntEnum):
    OSPF_WEIGHT = 0
    BANDWIDTH = 1
    CAPACITY = 2
    QUEUE_LENGTH = 3
    LOCAL_PREFERENCE = 4
    AS_PATH_LENGTH = 5
    MED = 6


PARAMETER_NAME_TO_ID = {
    "ospf_weight": ParameterID.OSPF_WEIGHT,
    "bandwidth": ParameterID.BANDWIDTH,
    "capacity": ParameterID.CAPACITY,
    "queue_length": ParameterID.QUEUE_LENGTH,
    "local_preference": ParameterID.LOCAL_PREFERENCE,
    "as_path_length": ParameterID.AS_PATH_LENGTH,
    "med": ParameterID.MED,
}


class ParameterIDEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(len(ParameterID), embedding_dim)

    def forward(self, parameter_name: str, count: int, device: torch.device | None = None) -> torch.Tensor:
        parameter_id = PARAMETER_NAME_TO_ID[parameter_name]
        ids = torch.full((count,), int(parameter_id), dtype=torch.long, device=device)
        return self.embedding(ids)
