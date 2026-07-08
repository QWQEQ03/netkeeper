from __future__ import annotations

import copy

from torch import nn


def clone_target_network(module: nn.Module) -> nn.Module:
    target = copy.deepcopy(module)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    target.eval()
    return target
