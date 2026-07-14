"""Compatibility conversion from legacy actor dictionaries to JointAction.

The adapter is intentionally outside Actor/COMA: it preserves old output
dimensions while giving the schema environment explicit target/value actions.
"""
from __future__ import annotations

from typing import Any, Mapping

from netkeeper_sim.schemas import AtomicAction, JointAction


def legacy_action_to_joint_action(
    action: Mapping[str, Mapping[str, Any]],
    *,
    link_ids: tuple[str, ...],
    bgp_targets: tuple[tuple[str, str, str], ...] = (),
    snapshot_id: str | None = None,
) -> JointAction:
    actions: list[AtomicAction] = []
    for parameter, agent, targets in (
        ("ospf_weight", "ospf", tuple({"link_id": link_id} for link_id in link_ids)),
        ("bandwidth_bps", "performance", tuple({"link_id": link_id} for link_id in link_ids)),
        ("capacity_bps", "performance", tuple({"link_id": link_id} for link_id in link_ids)),
        ("queue_packets", "performance", tuple({"link_id": link_id} for link_id in link_ids)),
    ):
        for target, value in zip(targets, _values(action.get(agent, {}).get(parameter))):
            actions.append(_legacy_atomic(agent, parameter, target, value))
    for parameter in ("local_preference", "as_path_length", "med"):
        for target, value in zip(bgp_targets, _values(action.get("bgp", {}).get(parameter))):
            actions.append(_legacy_atomic("bgp", parameter, {"router_id": target[0], "prefix": target[1], "next_hop": target[2]}, value))
    return JointAction(tuple(actions), requested_by="legacy_actor", snapshot_id=snapshot_id)


def _values(value):
    if value is None: return ()
    try:
        from netkeeper_sim.rl import _tensor as T
        return tuple(T.to_numpy(value).reshape(-1).tolist())
    except Exception:
        return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def _legacy_atomic(agent, parameter, target, value):
    # Legacy zero encodes "do not change"; nonzero values remain set values.
    return AtomicAction(agent, parameter, target, "no_update" if int(value) == 0 else "set", None if int(value) == 0 else int(value))
