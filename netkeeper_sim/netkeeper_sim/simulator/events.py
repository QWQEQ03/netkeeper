"""Atomic, ordered Event application for immutable schema snapshots."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from netkeeper_sim.schemas import ErrorRecord, Event, LinkAttributes, NetworkSnapshot, Policy, TrafficDemand, TrafficMatrix


@dataclass(frozen=True)
class EventApplicationResult:
    snapshot: NetworkSnapshot
    applied_event_ids: tuple[str, ...]
    errors: tuple[ErrorRecord, ...]


def apply_scheduled_events(snapshot: NetworkSnapshot, events: Iterable[Event]) -> EventApplicationResult:
    """Apply events scheduled for ``snapshot.step`` in ``(step, event_id)`` order.

    Each event is transactional: validation is performed against a candidate
    copy and an invalid event leaves the accumulated state untouched.
    """
    current = snapshot
    applied: list[str] = []
    errors: list[ErrorRecord] = []
    for event in sorted((item for item in events if item.step == snapshot.step), key=lambda item: (item.step, item.event_id)):
        try:
            current = _apply_one(current, event)
            applied.append(event.event_id)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(ErrorRecord("invalid_event", str(exc), field_path=f"event:{event.event_id}"))
    return EventApplicationResult(current, tuple(applied), tuple(errors))


def _apply_one(snapshot: NetworkSnapshot, event: Event) -> NetworkSnapshot:
    config, traffic, policies = snapshot.configuration, snapshot.traffic, list(snapshot.policies)
    target = event.target_id or event.payload.get("target_id")
    if event.kind in {"link_down", "link_up"}:
        if target not in {link.link_id for link in snapshot.topology.links}:
            raise ValueError("unknown_link_id")
        config = config.with_updates(link_states={target: "down" if event.kind == "link_down" else "up"}, step=snapshot.step)
    elif event.kind in {"node_down", "node_up"}:
        if target not in {node.node_id for node in snapshot.topology.nodes}:
            raise ValueError("unknown_node_id")
        config = config.with_updates(node_states={target: "down" if event.kind == "node_down" else "up"}, step=snapshot.step)
    elif event.kind == "policy_add":
        raw = event.payload.get("policy")
        if not isinstance(raw, dict): raise ValueError("policy_add_requires_policy")
        policy = Policy.from_dict(raw)
        if any(item.policy_id == policy.policy_id for item in policies): raise ValueError("duplicate_policy_id")
        policies.append(policy)
    elif event.kind == "policy_remove":
        if not isinstance(target, str) or not any(item.policy_id == target for item in policies): raise ValueError("unknown_policy_id")
        policies = [item for item in policies if item.policy_id != target]
    elif event.kind == "traffic_set" or event.kind == "traffic_replace":
        raw = event.payload.get("traffic")
        if not isinstance(raw, dict): raise ValueError("traffic_set_requires_traffic")
        traffic = TrafficMatrix.from_dict(raw)
        _validate_traffic(snapshot, traffic)
    elif event.kind == "traffic_scale":
        factor = event.payload.get("factor")
        if not isinstance(factor, (int, float)) or factor <= 0: raise ValueError("traffic_scale_requires_positive_factor")
        traffic = replace(traffic, load_multiplier=traffic.load_multiplier * float(factor))
    elif event.kind == "hotspot_change":
        raw = event.payload.get("demand")
        if not isinstance(raw, dict): raise ValueError("hotspot_change_requires_demand")
        replacement = TrafficDemand.from_dict(raw)
        replaced = False
        demands = []
        for demand in traffic.demands:
            same = demand.source == replacement.source and demand.destination == replacement.destination and demand.prefix == replacement.prefix
            demands.append(replacement if same else demand); replaced = replaced or same
        if not replaced: raise ValueError("hotspot_demand_not_found")
        traffic = replace(traffic, demands=tuple(demands))
    elif event.kind == "config_patch":
        config = _config_patch(config, event.payload, snapshot)
    else:
        raise ValueError("unsupported_event_kind")
    # Snapshot ID includes topology-state version; event-only traffic/policy
    # changes must therefore advance it just like a topology state transition.
    return snapshot.next(configuration=config, traffic=traffic, policies=tuple(policies), topology_state_version=snapshot.topology_state_version + 1, step=snapshot.step)


def _config_patch(config, payload, snapshot):
    allowed = {"ospf_weights", "link_states", "node_states", "performance"}
    if set(payload) - allowed: raise ValueError("config_patch_contains_unsupported_field")
    link_ids = {link.link_id for link in snapshot.topology.links}
    node_ids = {node.node_id for node in snapshot.topology.nodes}
    for key in ("ospf_weights", "link_states", "performance"):
        if key in payload and (not isinstance(payload[key], dict) or set(payload[key]) - link_ids):
            raise ValueError(f"{key}_contains_unknown_or_invalid_link_id")
    if "node_states" in payload and (not isinstance(payload["node_states"], dict) or set(payload["node_states"]) - node_ids):
        raise ValueError("node_states_contains_unknown_or_invalid_node_id")
    performance = payload.get("performance")
    if performance is not None:
        if not isinstance(performance, dict): raise ValueError("performance_must_be_mapping")
        performance = {key: LinkAttributes.from_dict(value) if isinstance(value, dict) else value for key, value in performance.items()}
    return config.with_updates(
        ospf_weights=payload.get("ospf_weights"), link_states=payload.get("link_states"),
        node_states=payload.get("node_states"), performance=performance,
        step=config.step,
    )


def _validate_traffic(snapshot, traffic):
    nodes = {node.node_id for node in snapshot.topology.nodes}
    if set(traffic.node_order) != nodes:
        raise ValueError("traffic_node_order_must_match_topology")
    for demand in traffic.demands:
        if demand.source not in nodes or (demand.destination is not None and demand.destination not in nodes):
            raise ValueError("traffic_demand_contains_unknown_node")
