from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from netkeeper_sim.routing.bgp import BGPRouteTable
from netkeeper_sim.routing.ospf import ForwardingTable

DestinationType = Literal["node", "prefix"]
TrafficShiftMode = Literal["union", "intersection"]
TrafficShiftChangeType = Literal[
    "unchanged",
    "modified",
    "added",
    "removed",
    "reachability_changed",
]


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
