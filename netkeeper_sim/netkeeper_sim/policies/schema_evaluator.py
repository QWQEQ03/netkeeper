"""Deterministic policy semantics for immutable schema snapshots."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Iterable, Literal

from netkeeper_sim.schemas import NetworkConfiguration, Policy, RoutingEntry, Topology
from netkeeper_sim.simulator.deterministic import SelectedBGPRoute


PolicyStatus = Literal["satisfied", "unsatisfied", "conflict", "infeasible_due_to_failure", "invalid"]


@dataclass(frozen=True)
class SchemaPolicyEvaluation:
    policy_id: str
    kind: str
    status: PolicyStatus
    reason: str | None
    paths_examined: int


@dataclass(frozen=True)
class PolicyConsistencyReport:
    overall_consistency: float
    feasible_only_consistency: float
    numerator: int
    denominator: int
    feasible_numerator: int
    feasible_denominator: int
    excluded_count: int
    by_kind: dict[str, float]
    evaluations: tuple[SchemaPolicyEvaluation, ...]
    evaluated_policies: tuple[Policy, ...]


def evaluate_schema_policies(
    policies: Iterable[Policy],
    routing: Iterable[RoutingEntry],
    configuration: NetworkConfiguration,
    topology: Topology,
    selected_bgp_routes: Iterable[SelectedBGPRoute] = (),
    *,
    max_paths: int = 1000,
) -> PolicyConsistencyReport:
    """Evaluate all enabled policies with explicit all-path ECMP semantics.

    Waypoints and forbidden nodes may not be endpoints: endpoints are excluded
    from forward checks by design, preventing vacuous endpoint satisfaction.
    Isolation means intermediate-node disjointness for every pair of ECMP
    paths; endpoints are ignored.
    """
    route_map = {(entry.router_id, entry.destination): entry for entry in routing}
    bgp_map = {(item.router_id, item.prefix): item.next_hop for item in selected_bgp_routes}
    policy_list = tuple(policies)
    conflicts = _conflicts(policy_list)
    evaluated: list[Policy] = []
    details: list[SchemaPolicyEvaluation] = []
    for policy in policy_list:
        if not policy.enabled:
            evaluated.append(replace(policy, status="pending", reason_code="disabled", conflict_with=()))
            continue
        if policy.policy_id in conflicts:
            peers = tuple(sorted(conflicts[policy.policy_id]))
            evaluated.append(replace(policy, status="conflict", reason_code="contradictory_forward_constraints", conflict_with=peers))
            details.append(SchemaPolicyEvaluation(policy.policy_id, policy.kind, "conflict", "contradictory_forward_constraints", 0))
            continue
        status, reason, path_count = _evaluate(policy, route_map, bgp_map, configuration, topology, max_paths)
        evaluated.append(replace(policy, status=status, reason_code=reason, conflict_with=()))
        details.append(SchemaPolicyEvaluation(policy.policy_id, policy.kind, status, reason, path_count))

    enabled = [item for item in details]
    numerator = sum(item.status == "satisfied" for item in enabled)
    denominator = len(enabled)
    feasible = [item for item in enabled if item.status not in {"conflict", "infeasible_due_to_failure", "invalid"}]
    feasible_numerator = sum(item.status == "satisfied" for item in feasible)
    by_kind: dict[str, float] = {}
    for kind in sorted({item.kind for item in enabled}):
        group = [item for item in enabled if item.kind == kind]
        by_kind[kind] = sum(item.status == "satisfied" for item in group) / len(group) if group else 1.0
    return PolicyConsistencyReport(
        numerator / denominator if denominator else 1.0,
        feasible_numerator / len(feasible) if feasible else 1.0,
        numerator, denominator, feasible_numerator, len(feasible), denominator - len(feasible), by_kind,
        tuple(details), tuple(evaluated),
    )


def _evaluate(policy, route_map, bgp_map, configuration, topology, max_paths):
    try:
        if policy.kind == "isolation":
            return _isolation(policy, route_map, bgp_map, configuration, topology, max_paths)
        source, destination, error = _relation(policy.fields, route_map, bgp_map)
        if error:
            return "invalid", error, 0
        paths = _paths(source, destination, route_map, max_paths)
        if not paths:
            return _no_path_status(configuration), "no_reachable_path", 0
        if policy.kind == "reachable":
            return "satisfied", None, len(paths)
        mode = policy.fields.get("path_mode", "all_path")
        if mode not in {"all_path", "any_path"}:
            return "invalid", "invalid_path_mode", len(paths)
        key = "waypoint" if policy.kind == "forward_pass" else "forbidden_node"
        checked = policy.fields.get(key)
        if not isinstance(checked, str) or checked in {source, destination}:
            return "invalid", f"invalid_{key}", len(paths)
        matches = [checked in path[1:-1] for path in paths]
        if policy.kind == "forward_avoid":
            matches = [not match for match in matches]
        ok = any(matches) if mode == "any_path" else all(matches)
        return ("satisfied", None if ok else f"{policy.kind}_{mode}_failed", len(paths)) if ok else ("unsatisfied", f"{policy.kind}_{mode}_failed", len(paths))
    except ValueError as exc:
        return "invalid", str(exc), 0


def _isolation(policy, route_map, bgp_map, configuration, topology, max_paths):
    first_source, first_destination, first_error = _relation(policy.fields, route_map, bgp_map, prefix="first_")
    second_source, second_destination, second_error = _relation(policy.fields, route_map, bgp_map, prefix="second_")
    if first_error or second_error:
        return "invalid", first_error or second_error, 0
    if policy.fields.get("resource", "node") != "node":
        return "invalid", "only_node_isolation_is_supported", 0
    first_paths = _paths(first_source, first_destination, route_map, max_paths)
    second_paths = _paths(second_source, second_destination, route_map, max_paths)
    if not first_paths or not second_paths:
        return _no_path_status(configuration), "no_reachable_path", len(first_paths) + len(second_paths)
    for left in first_paths:
        for right in second_paths:
            if set(left[1:-1]) & set(right[1:-1]):
                return "unsatisfied", "paths_share_intermediate_node", len(first_paths) + len(second_paths)
    return "satisfied", None, len(first_paths) + len(second_paths)


def _relation(fields, route_map, bgp_map, prefix=""):
    source = fields.get(f"{prefix}source")
    destination = fields.get(f"{prefix}destination")
    destination_type = fields.get(f"{prefix}destination_type", "node")
    if not isinstance(source, str) or not isinstance(destination, str):
        return "", "", "missing_source_or_destination"
    if destination_type == "prefix":
        destination = bgp_map.get((source, destination))
        if destination is None:
            return source, "", "missing_bgp_route"
    elif destination_type != "node":
        return source, destination, "invalid_destination_type"
    if source != destination and (source, destination) not in route_map:
        return source, destination, "missing_forwarding_entry"
    return source, destination, None


def _paths(source, destination, route_map, max_paths):
    if source == destination:
        return ((source,),)
    paths: list[tuple[str, ...]] = []
    def walk(node, path):
        if len(paths) >= max_paths:
            raise ValueError("forwarding_path_limit_exceeded")
        if node == destination:
            paths.append(path)
            return
        entry = route_map.get((node, destination))
        if entry is None or not entry.reachable or not entry.next_hops:
            return
        for next_hop in sorted(entry.next_hops):
            if next_hop not in path:
                walk(next_hop, (*path, next_hop))
    walk(source, (source,))
    return tuple(paths)


def _no_path_status(configuration):
    return "infeasible_due_to_failure" if "down" in (*configuration.link_states.values(), *configuration.node_states.values()) else "unsatisfied"


def _conflicts(policies):
    conflicts: dict[str, set[str]] = defaultdict(set)
    passes: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    avoids: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for policy in policies:
        if not policy.enabled or policy.kind not in {"forward_pass", "forward_avoid"}:
            continue
        source, destination = policy.fields.get("source"), policy.fields.get("destination")
        field = "waypoint" if policy.kind == "forward_pass" else "forbidden_node"
        node = policy.fields.get(field)
        if all(isinstance(value, str) for value in (source, destination, node)):
            (passes if policy.kind == "forward_pass" else avoids)[source, destination, node].append(policy.policy_id)
    for key in set(passes) & set(avoids):
        for left in passes[key]:
            for right in avoids[key]:
                conflicts[left].add(right); conflicts[right].add(left)
    return conflicts
