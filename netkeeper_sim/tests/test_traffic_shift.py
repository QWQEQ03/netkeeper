from __future__ import annotations

from netkeeper_sim.metrics.traffic_shift import (
    ForwardingKey,
    ForwardingPlaneSnapshot,
    ForwardingState,
    calculate_traffic_shift,
    capture_forwarding_plane_snapshot,
)
from netkeeper_sim.routing.bgp import BGPRoute
from netkeeper_sim.routing.ospf import compute_ospf_routes
from netkeeper_sim.topology.model import build_topology_from_edges


def state(reachable: bool, *next_hops: str) -> ForwardingState:
    return ForwardingState(reachable=reachable, next_hops=frozenset(next_hops))


def snapshot(*items: tuple[ForwardingKey, ForwardingState]) -> ForwardingPlaneSnapshot:
    return ForwardingPlaneSnapshot(dict(items))


def test_identical_forwarding_planes_have_zero_shift():
    key = ForwardingKey("R1", "R4", "node")
    previous = snapshot((key, state(True, "R2", "R3")))
    current = snapshot((key, state(True, "R3", "R2")))

    result = calculate_traffic_shift(previous, current)

    assert result.shift_ratio == 0.0
    assert result.changed_entries == 0
    assert result.unchanged_entries == 1


def test_next_hop_change_counts_as_modified_entry():
    key = ForwardingKey("R1", "R4", "node")
    previous = snapshot((key, state(True, "R2")))
    current = snapshot((key, state(True, "R3")))

    result = calculate_traffic_shift(previous, current)

    assert result.shift_ratio == 1.0
    assert result.modified_entries == 1
    assert result.changed_keys == (key,)


def test_ecmp_set_reduction_counts_as_shift():
    key = ForwardingKey("R1", "R4", "node")
    previous = snapshot((key, state(True, "R2", "R3")))
    current = snapshot((key, state(True, "R2")))

    result = calculate_traffic_shift(previous, current)

    assert result.changed_entries == 1
    assert result.modified_entries == 1


def test_reachability_changes_count_as_shift():
    key = ForwardingKey("R1", "R4", "node")
    previous = snapshot((key, state(False)))
    current = snapshot((key, state(True, "R2")))

    result = calculate_traffic_shift(previous, current)

    assert result.changed_entries == 1
    assert result.reachability_changed_entries == 1


def test_added_and_removed_entries_count_only_in_union_mode():
    common = ForwardingKey("R1", "R2", "node")
    added = ForwardingKey("R1", "R3", "node")
    removed = ForwardingKey("R1", "R4", "node")
    previous = snapshot((common, state(True, "R2")), (removed, state(True, "R4")))
    current = snapshot((common, state(True, "R2")), (added, state(True, "R3")))

    union_result = calculate_traffic_shift(previous, current, mode="union")
    intersection_result = calculate_traffic_shift(previous, current, mode="intersection")

    assert union_result.total_entries == 3
    assert union_result.added_entries == 1
    assert union_result.removed_entries == 1
    assert union_result.shift_ratio == 2 / 3
    assert intersection_result.total_entries == 1
    assert intersection_result.shift_ratio == 0.0


def test_unreachable_to_unreachable_is_unchanged():
    key = ForwardingKey("R1", "R4", "node")
    previous = snapshot((key, state(False)))
    current = snapshot((key, state(False)))

    result = calculate_traffic_shift(previous, current)

    assert result.changed_entries == 0
    assert result.unchanged_entries == 1


def test_ospf_weight_update_produces_traffic_shift(diamond_topology):
    previous_table = compute_ospf_routes(diamond_topology)
    previous = capture_forwarding_plane_snapshot(previous_table)

    diamond_topology.update_ospf_weight("R1", "R3", 5)
    current_table = compute_ospf_routes(diamond_topology)
    current = capture_forwarding_plane_snapshot(current_table)

    result = calculate_traffic_shift(previous, current)

    assert result.changed_entries > 0
    assert result.shift_ratio > 0.0


def test_link_failure_and_restore_produce_shift(single_path_topology):
    original_table = compute_ospf_routes(single_path_topology)
    original = capture_forwarding_plane_snapshot(original_table)

    single_path_topology.fail_link("R2", "R3")
    failed_table = compute_ospf_routes(single_path_topology)
    failed = capture_forwarding_plane_snapshot(failed_table)
    failure_shift = calculate_traffic_shift(original, failed)

    single_path_topology.restore_link("R2", "R3")
    restored_table = compute_ospf_routes(single_path_topology)
    restored = capture_forwarding_plane_snapshot(restored_table)
    restore_shift = calculate_traffic_shift(failed, restored)

    assert failure_shift.reachability_changed_entries > 0
    assert restore_shift.reachability_changed_entries > 0


def test_bgp_best_route_change_counts_as_prefix_shift():
    topology = build_topology_from_edges(
        [
            ("R1", "R2", 1),
            ("R1", "R3", 1),
        ]
    )
    table = compute_ospf_routes(topology)
    prefix = "203.0.113.0/24"
    route_to_r2 = BGPRoute(prefix, "R2", 100, (65001,), 100, "R2", "peer-a")
    route_to_r3 = BGPRoute(prefix, "R3", 100, (65002,), 100, "R3", "peer-b")
    previous = capture_forwarding_plane_snapshot(table, {"R1": {prefix: route_to_r2}})
    current = capture_forwarding_plane_snapshot(table, {"R1": {prefix: route_to_r3}})
    key = ForwardingKey("R1", prefix, "prefix")

    result = calculate_traffic_shift(previous, current)

    assert key in result.changed_keys
    detail = next(item for item in result.per_entry_details if item.key == key)
    assert detail.change_type == "modified"
