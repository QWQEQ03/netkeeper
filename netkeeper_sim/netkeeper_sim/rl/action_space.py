from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from netkeeper_sim.rl import _tensor as T
from netkeeper_sim.routing.bgp import BGPRoute
from netkeeper_sim.topology.model import Topology


@dataclass(frozen=True, order=True)
class BGPRouteTarget:
    router: str
    prefix: str
    route_index: int


@dataclass(frozen=True)
class ActionMasks:
    ospf_weight_mask: T.Tensor
    local_preference_mask: T.Tensor
    as_path_length_mask: T.Tensor
    med_mask: T.Tensor
    bandwidth_mask: T.Tensor
    capacity_mask: T.Tensor
    queue_length_mask: T.Tensor
    link_ids: tuple[str, ...]
    bgp_route_targets: tuple[BGPRouteTarget, ...]

    def as_dict(self) -> dict[str, T.Tensor]:
        return {
            "ospf_weight_mask": self.ospf_weight_mask,
            "local_preference_mask": self.local_preference_mask,
            "as_path_length_mask": self.as_path_length_mask,
            "med_mask": self.med_mask,
            "bandwidth_mask": self.bandwidth_mask,
            "capacity_mask": self.capacity_mask,
            "queue_length_mask": self.queue_length_mask,
        }


def sorted_link_ids(topology: Topology) -> tuple[str, ...]:
    return tuple(sorted(topology.links))


def sorted_bgp_route_targets(
    candidate_routes: Mapping[str, Mapping[str, list[BGPRoute]]] | None,
) -> tuple[BGPRouteTarget, ...]:
    if not candidate_routes:
        return ()
    targets: list[BGPRouteTarget] = []
    for router in sorted(candidate_routes):
        for prefix in sorted(candidate_routes[router]):
            for route_index, _route in enumerate(candidate_routes[router][prefix]):
                targets.append(BGPRouteTarget(router, prefix, route_index))
    return tuple(targets)


def build_action_masks(
    topology: Topology,
    candidate_routes: Mapping[str, Mapping[str, list[BGPRoute]]] | None = None,
) -> ActionMasks:
    link_ids = sorted_link_ids(topology)
    link_mask = [topology.links[link_id].is_active for link_id in link_ids]
    route_targets = sorted_bgp_route_targets(candidate_routes)
    route_mask = [
        target.router in topology.nodes
        and _route_for_target(candidate_routes, target).next_hop in topology.nodes
        for target in route_targets
    ]
    return ActionMasks(
        ospf_weight_mask=T.tensor(link_mask, "bool"),
        local_preference_mask=T.tensor(route_mask, "bool"),
        as_path_length_mask=T.tensor(route_mask, "bool"),
        med_mask=T.tensor(route_mask, "bool"),
        bandwidth_mask=T.tensor(link_mask, "bool"),
        capacity_mask=T.tensor(link_mask, "bool"),
        queue_length_mask=T.tensor(link_mask, "bool"),
        link_ids=link_ids,
        bgp_route_targets=route_targets,
    )


def canonical_joint_action(joint_action: Mapping[str, Any]) -> tuple[Any, ...]:
    return _freeze(joint_action)


def action_values(
    raw_action: Any,
    expected_length: int,
) -> np.ndarray:
    """Return an int vector; scalar values are broadcast to every target."""
    if raw_action is None:
        return np.zeros(expected_length, dtype=np.int64)
    array = T.to_numpy(raw_action)
    if array.ndim == 0:
        return np.full(expected_length, int(array), dtype=np.int64)
    flat = array.astype(np.int64, copy=False).reshape(-1)
    if len(flat) < expected_length:
        padded = np.zeros(expected_length, dtype=np.int64)
        padded[: len(flat)] = flat
        return padded
    return flat[:expected_length]


def _route_for_target(
    candidate_routes: Mapping[str, Mapping[str, list[BGPRoute]]] | None,
    target: BGPRouteTarget,
) -> BGPRoute:
    assert candidate_routes is not None
    return candidate_routes[target.router][target.prefix][target.route_index]


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple((key, _freeze(value[key])) for key in sorted(value))
    if isinstance(value, np.ndarray):
        return tuple(value.reshape(-1).tolist())
    array = T.to_numpy(value)
    if array.ndim > 0:
        return tuple(array.reshape(-1).tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value
