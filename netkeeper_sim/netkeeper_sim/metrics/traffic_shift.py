from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Literal

from netkeeper_sim.routing.bgp import BGPRouteTable
from netkeeper_sim.routing.ospf import ForwardingTable
from netkeeper_sim.schemas.models import RoutingEntry
if TYPE_CHECKING:
    from netkeeper_sim.simulator.deterministic import SelectedBGPRoute

DestinationType = Literal["node", "prefix"]
TrafficShiftMode = Literal["union", "intersection"]
TrafficShiftChangeType = Literal[
    "unchanged",
    "modified",
    "added",
    "removed",
    "reachability_changed",
]
TrafficShiftMetricVersion = Literal["paper_v1", "project_v1"]


@dataclass(frozen=True, order=True)
class ForwardingKey:
    router: str
    destination: str
    destination_type: DestinationType


@dataclass(frozen=True)
class ForwardingState:
    reachable: bool
    next_hops: frozenset[str]


@dataclass(frozen=True)
class ForwardingPlaneSnapshot:
    entries: dict[ForwardingKey, ForwardingState]


@dataclass(frozen=True)
class TrafficShiftEntryDetail:
    key: ForwardingKey
    previous: ForwardingState | None
    current: ForwardingState | None
    changed: bool
    change_type: TrafficShiftChangeType


@dataclass(frozen=True)
class TrafficShiftResult:
    shift_ratio: float
    changed_entries: int
    total_entries: int
    unchanged_entries: int
    added_entries: int
    removed_entries: int
    modified_entries: int
    reachability_changed_entries: int
    changed_keys: tuple[ForwardingKey, ...]
    per_entry_details: tuple[TrafficShiftEntryDetail, ...]


@dataclass(frozen=True)
class VersionedTrafficShiftResult:
    metric_version: TrafficShiftMetricVersion
    numerator: int
    denominator: int
    shift_ratio: float
    excluded_unreachable: int
    added_entries: int
    removed_entries: int
    treatment: str


def calculate_versioned_traffic_shift(
    previous: ForwardingPlaneSnapshot,
    current: ForwardingPlaneSnapshot,
    metric_version: TrafficShiftMetricVersion,
) -> VersionedTrafficShiftResult:
    """Calculate the frozen paper/project metric definitions.

    ``paper_v1`` compares only prefix FIB entries reachable in both snapshots:
    new, removed, and unreachable entries are excluded from its denominator.
    ``project_v1`` compares all node/prefix keys in the union: added/removed or
    reachability changes are counted as shifts.
    """
    before, after = previous.entries, current.entries
    if metric_version == "paper_v1":
        keys = sorted(set(before) & set(after))
        eligible = [key for key in keys if key.destination_type == "prefix" and before[key].reachable and after[key].reachable]
        numerator = sum(before[key].next_hops != after[key].next_hops for key in eligible)
        excluded = len(keys) - len(eligible)
        return VersionedTrafficShiftResult("paper_v1", numerator, len(eligible), numerator / len(eligible) if eligible else 0.0, excluded, 0, 0, "unreachable_and_added_removed_prefixes_excluded")
    if metric_version == "project_v1":
        keys = sorted(set(before) | set(after))
        numerator = 0; added = 0; removed = 0
        for key in keys:
            left, right = before.get(key), after.get(key)
            if left is None: added += 1; numerator += 1
            elif right is None: removed += 1; numerator += 1
            elif left.reachable != right.reachable or (left.reachable and left.next_hops != right.next_hops): numerator += 1
        return VersionedTrafficShiftResult("project_v1", numerator, len(keys), numerator / len(keys) if keys else 0.0, 0, added, removed, "union_including_added_removed_and_reachability_changes")
    raise ValueError("unsupported traffic shift metric version")


def capture_forwarding_plane_snapshot(
    forwarding_table: ForwardingTable,
    bgp_routes: BGPRouteTable | None = None,
    include_self_entries: bool = False,
) -> ForwardingPlaneSnapshot:
    entries: dict[ForwardingKey, ForwardingState] = {}
    for router in sorted(forwarding_table):
        destinations = forwarding_table[router]
        for destination in sorted(destinations):
            if not include_self_entries and router == destination:
                continue
            entry = destinations[destination]
            key = ForwardingKey(router, destination, "node")
            entries[key] = ForwardingState(
                reachable=entry.reachable,
                next_hops=frozenset(entry.next_hops if entry.reachable else ()),
            )

    if bgp_routes is not None:
        for router in sorted(bgp_routes):
            for prefix in sorted(bgp_routes[router]):
                route = bgp_routes[router][prefix]
                key = ForwardingKey(router, prefix, "prefix")
                reachable = route.next_hop == router
                if not reachable:
                    entry = forwarding_table.get(router, {}).get(route.next_hop)
                    reachable = entry is not None and entry.reachable
                entries[key] = ForwardingState(
                    reachable=reachable,
                    next_hops=frozenset((route.next_hop,)) if reachable else frozenset(),
                )

    return ForwardingPlaneSnapshot(entries=entries)


def capture_schema_forwarding_plane_snapshot(
    routing: tuple[RoutingEntry, ...] | list[RoutingEntry],
    selected_bgp_routes: Iterable["SelectedBGPRoute"] = (),
) -> ForwardingPlaneSnapshot:
    entries: dict[ForwardingKey, ForwardingState] = {}
    for route in routing:
        if route.router_id == route.destination:
            continue
        entries[ForwardingKey(route.router_id, route.destination, "node")] = ForwardingState(route.reachable, frozenset(route.next_hops if route.reachable else ()))
    route_map = {(item.router_id, item.destination): item for item in routing}
    for route in selected_bgp_routes:
        igp = route_map.get((route.router_id, route.next_hop))
        reachable = route.router_id == route.next_hop or (igp is not None and igp.reachable)
        entries[ForwardingKey(route.router_id, route.prefix, "prefix")] = ForwardingState(reachable, frozenset((route.next_hop,)) if reachable else frozenset())
    return ForwardingPlaneSnapshot(entries)


def calculate_traffic_shift(
    previous: ForwardingPlaneSnapshot,
    current: ForwardingPlaneSnapshot,
    mode: TrafficShiftMode = "union",
) -> TrafficShiftResult:
    if mode == "union":
        keys = set(previous.entries) | set(current.entries)
    elif mode == "intersection":
        keys = set(previous.entries) & set(current.entries)
    else:
        raise ValueError("Traffic shift mode must be 'union' or 'intersection'")

    details: list[TrafficShiftEntryDetail] = []
    added = 0
    removed = 0
    modified = 0
    reachability_changed = 0
    unchanged = 0

    for key in sorted(keys):
        previous_state = previous.entries.get(key)
        current_state = current.entries.get(key)
        change_type = _change_type(previous_state, current_state)
        changed = change_type != "unchanged"
        if change_type == "added":
            added += 1
        elif change_type == "removed":
            removed += 1
        elif change_type == "modified":
            modified += 1
        elif change_type == "reachability_changed":
            reachability_changed += 1
        else:
            unchanged += 1
        details.append(
            TrafficShiftEntryDetail(
                key=key,
                previous=previous_state,
                current=current_state,
                changed=changed,
                change_type=change_type,
            )
        )

    changed_entries = added + removed + modified + reachability_changed
    total_entries = len(details)
    shift_ratio = 0.0 if total_entries == 0 else changed_entries / total_entries
    changed_keys = tuple(detail.key for detail in details if detail.changed)
    return TrafficShiftResult(
        shift_ratio=shift_ratio,
        changed_entries=changed_entries,
        total_entries=total_entries,
        unchanged_entries=unchanged,
        added_entries=added,
        removed_entries=removed,
        modified_entries=modified,
        reachability_changed_entries=reachability_changed,
        changed_keys=changed_keys,
        per_entry_details=tuple(details),
    )


def _change_type(
    previous: ForwardingState | None,
    current: ForwardingState | None,
) -> TrafficShiftChangeType:
    if previous is None and current is None:
        return "unchanged"
    if previous is None:
        return "added"
    if current is None:
        return "removed"
    if previous.reachable != current.reachable:
        return "reachability_changed"
    if not previous.reachable and not current.reachable:
        return "unchanged"
    if previous.next_hops != current.next_hops:
        return "modified"
    return "unchanged"
