from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DestinationType = Literal["node", "prefix"]
PathMode = Literal["any_path", "all_paths"]
IsolationMode = Literal["forbidden_node", "path_disjoint"]


@dataclass(frozen=True)
class ForwardPolicy:
    policy_id: str
    source: str
    destination: str
    required_next_hop: str
    destination_type: DestinationType = "node"


@dataclass(frozen=True)
class ReachablePolicy:
    policy_id: str
    source: str
    destination: str
    must_pass: str
    mode: PathMode = "any_path"
    destination_type: DestinationType = "node"

    def __post_init__(self) -> None:
        if self.mode not in ("any_path", "all_paths"):
            raise ValueError("ReachablePolicy mode must be 'any_path' or 'all_paths'")


@dataclass(frozen=True)
class IsolationPolicy:
    policy_id: str
    first_source: str
    first_destination: str
    second_source: str
    second_destination: str
    forbidden_node: str | None = None
    mode: IsolationMode = "forbidden_node"
    first_destination_type: DestinationType = "node"
    second_destination_type: DestinationType = "node"

    def __post_init__(self) -> None:
        if self.mode not in ("forbidden_node", "path_disjoint"):
            raise ValueError("IsolationPolicy mode must be 'forbidden_node' or 'path_disjoint'")
        if self.mode == "forbidden_node" and self.forbidden_node is None:
            raise ValueError("IsolationPolicy forbidden_node mode requires forbidden_node")


Policy = ForwardPolicy | ReachablePolicy | IsolationPolicy
