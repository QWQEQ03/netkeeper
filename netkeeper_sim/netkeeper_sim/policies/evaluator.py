from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from netkeeper_sim.policies.model import (
    DestinationType,
    ForwardPolicy,
    IsolationPolicy,
    Policy,
    ReachablePolicy,
)
from netkeeper_sim.routing.bgp import BGPRouteTable, best_route_for_prefix
from netkeeper_sim.routing.ospf import ForwardingEntry, ForwardingTable


class ForwardingPathLimitExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class PolicyEvaluationResult:
    policy_id: str
    satisfied: bool
    reason: str


@dataclass(frozen=True)
class PolicyConsistencyResult:
    consistency: float
    total: int
    satisfied: int
    unsatisfied: int
    results: list[PolicyEvaluationResult]


def evaluate_policy_consistency(
    policies: Iterable[Policy],
    forwarding_table: ForwardingTable,
    bgp_routes: BGPRouteTable | None = None,
    max_hops: int | None = None,
    max_paths: int = 1000,
) -> PolicyConsistencyResult:
    policy_list = list(policies)
    results = [
        evaluate_policy(
            policy,
            forwarding_table,
            bgp_routes=bgp_routes,
            max_hops=max_hops,
            max_paths=max_paths,
        )
        for policy in policy_list
    ]
    satisfied = sum(1 for result in results if result.satisfied)
    total = len(results)
    consistency = 1.0 if total == 0 else satisfied / total
    return PolicyConsistencyResult(
        consistency=consistency,
        total=total,
        satisfied=satisfied,
        unsatisfied=total - satisfied,
        results=results,
    )


def evaluate_policy(
    policy: Policy,
    forwarding_table: ForwardingTable,
    bgp_routes: BGPRouteTable | None = None,
    max_hops: int | None = None,
    max_paths: int = 1000,
) -> PolicyEvaluationResult:
    try:
        if isinstance(policy, ForwardPolicy):
            return _evaluate_forward_policy(policy, forwarding_table, bgp_routes)
        if isinstance(policy, ReachablePolicy):
            return _evaluate_reachable_policy(
                policy,
                forwarding_table,
                bgp_routes,
                max_hops=max_hops,
                max_paths=max_paths,
            )
        if isinstance(policy, IsolationPolicy):
            return _evaluate_isolation_policy(
                policy,
                forwarding_table,
                bgp_routes,
                max_hops=max_hops,
                max_paths=max_paths,
            )
    except ForwardingPathLimitExceeded as exc:
        return PolicyEvaluationResult(policy.policy_id, False, str(exc))
    raise TypeError(f"Unsupported policy type: {type(policy).__name__}")


def enumerate_forwarding_paths(
    forwarding_table: ForwardingTable,
    source: str,
    destination: str,
    max_hops: int | None = None,
    max_paths: int = 1000,
) -> tuple[tuple[str, ...], ...]:
    if max_paths <= 0:
        raise ValueError("max_paths must be positive")
    if source not in forwarding_table:
        return ()
    if destination not in forwarding_table[source]:
        return ()

    hop_limit = max_hops if max_hops is not None else max(1, len(forwarding_table) * 2)
    if hop_limit <= 0:
        raise ValueError("max_hops must be positive")

    paths: list[tuple[str, ...]] = []
    seen_paths: set[tuple[str, ...]] = set()

    def walk(current: str, path: tuple[str, ...]) -> None:
        if len(paths) >= max_paths:
            raise ForwardingPathLimitExceeded(
                f"Forwarding path limit exceeded for {source}->{destination}: {max_paths}"
            )
        if current == destination:
            if path not in seen_paths:
                seen_paths.add(path)
                paths.append(path)
            return
        if len(path) > hop_limit:
            return

        entry = forwarding_table.get(current, {}).get(destination)
        if entry is None or not entry.reachable:
            return
        for next_hop in sorted(entry.next_hops):
            if next_hop in path:
                continue
            walk(next_hop, (*path, next_hop))

    walk(source, (source,))
    return tuple(paths)


def _evaluate_forward_policy(
    policy: ForwardPolicy,
    forwarding_table: ForwardingTable,
    bgp_routes: BGPRouteTable | None,
) -> PolicyEvaluationResult:
    entry = _resolve_forwarding_entry(
        forwarding_table,
        bgp_routes,
        policy.source,
        policy.destination,
        policy.destination_type,
    )
    if entry is None:
        return PolicyEvaluationResult(policy.policy_id, False, "missing_forwarding_entry")
    if not entry.reachable:
        return PolicyEvaluationResult(policy.policy_id, False, "destination_unreachable")
    if policy.required_next_hop not in set(entry.next_hops):
        return PolicyEvaluationResult(policy.policy_id, False, "required_next_hop_absent")
    return PolicyEvaluationResult(policy.policy_id, True, "required_next_hop_present")


def _evaluate_reachable_policy(
    policy: ReachablePolicy,
    forwarding_table: ForwardingTable,
    bgp_routes: BGPRouteTable | None,
    max_hops: int | None,
    max_paths: int,
) -> PolicyEvaluationResult:
    resolved_destination = _resolve_path_destination(
        forwarding_table,
        bgp_routes,
        policy.source,
        policy.destination,
        policy.destination_type,
    )
    if resolved_destination is None:
        return PolicyEvaluationResult(policy.policy_id, False, "missing_forwarding_entry")
    paths = enumerate_forwarding_paths(
        forwarding_table,
        policy.source,
        resolved_destination,
        max_hops=max_hops,
        max_paths=max_paths,
    )
    if not paths:
        return PolicyEvaluationResult(policy.policy_id, False, "no_reachable_path")

    matches = [policy.must_pass in path for path in paths]
    if policy.mode == "any_path":
        if any(matches):
            return PolicyEvaluationResult(policy.policy_id, True, "must_pass_on_some_path")
        return PolicyEvaluationResult(policy.policy_id, False, "must_pass_absent")

    if all(matches):
        return PolicyEvaluationResult(policy.policy_id, True, "must_pass_on_all_paths")
    return PolicyEvaluationResult(policy.policy_id, False, "must_pass_not_on_all_paths")


def _evaluate_isolation_policy(
    policy: IsolationPolicy,
    forwarding_table: ForwardingTable,
    bgp_routes: BGPRouteTable | None,
    max_hops: int | None,
    max_paths: int,
) -> PolicyEvaluationResult:
    first_paths = _paths_for_relation(
        forwarding_table,
        bgp_routes,
        policy.first_source,
        policy.first_destination,
        policy.first_destination_type,
        max_hops=max_hops,
        max_paths=max_paths,
    )
    second_paths = _paths_for_relation(
        forwarding_table,
        bgp_routes,
        policy.second_source,
        policy.second_destination,
        policy.second_destination_type,
        max_hops=max_hops,
        max_paths=max_paths,
    )
    if not first_paths or not second_paths:
        return PolicyEvaluationResult(policy.policy_id, False, "no_reachable_path")

    if policy.mode == "forbidden_node":
        assert policy.forbidden_node is not None
        first_uses = any(policy.forbidden_node in _intermediate_nodes(path) for path in first_paths)
        second_uses = any(policy.forbidden_node in _intermediate_nodes(path) for path in second_paths)
        if first_uses and second_uses:
            return PolicyEvaluationResult(policy.policy_id, False, "forbidden_node_shared")
        return PolicyEvaluationResult(policy.policy_id, True, "forbidden_node_not_shared")

    for first_path in first_paths:
        first_intermediate = set(_intermediate_nodes(first_path))
        for second_path in second_paths:
            if first_intermediate & set(_intermediate_nodes(second_path)):
                return PolicyEvaluationResult(policy.policy_id, False, "paths_share_intermediate_node")
    return PolicyEvaluationResult(policy.policy_id, True, "paths_disjoint")


def _paths_for_relation(
    forwarding_table: ForwardingTable,
    bgp_routes: BGPRouteTable | None,
    source: str,
    destination: str,
    destination_type: DestinationType,
    max_hops: int | None,
    max_paths: int,
) -> tuple[tuple[str, ...], ...]:
    resolved_destination = _resolve_path_destination(
        forwarding_table,
        bgp_routes,
        source,
        destination,
        destination_type,
    )
    if resolved_destination is None:
        return ()
    return enumerate_forwarding_paths(
        forwarding_table,
        source,
        resolved_destination,
        max_hops=max_hops,
        max_paths=max_paths,
    )


def _resolve_forwarding_entry(
    forwarding_table: ForwardingTable,
    bgp_routes: BGPRouteTable | None,
    source: str,
    destination: str,
    destination_type: DestinationType,
) -> ForwardingEntry | None:
    resolved_destination = _resolve_path_destination(
        forwarding_table,
        bgp_routes,
        source,
        destination,
        destination_type,
    )
    if resolved_destination is None:
        return None
    return forwarding_table.get(source, {}).get(resolved_destination)


def _resolve_path_destination(
    forwarding_table: ForwardingTable,
    bgp_routes: BGPRouteTable | None,
    source: str,
    destination: str,
    destination_type: DestinationType,
) -> str | None:
    if destination_type == "node":
        return destination if destination in forwarding_table.get(source, {}) else None
    if destination_type != "prefix":
        raise ValueError("destination_type must be 'node' or 'prefix'")

    if bgp_routes is None:
        return None
    route = best_route_for_prefix(bgp_routes, source, destination)
    if route is None:
        return None
    return route.next_hop if route.next_hop in forwarding_table.get(source, {}) else None


def _intermediate_nodes(path: tuple[str, ...]) -> tuple[str, ...]:
    if len(path) <= 2:
        return ()
    return path[1:-1]
