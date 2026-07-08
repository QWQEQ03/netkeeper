from __future__ import annotations

import pytest

from netkeeper_sim.policies import (
    ForwardPolicy,
    IsolationPolicy,
    ReachablePolicy,
    evaluate_policy_consistency,
)
from netkeeper_sim.routing.ospf import compute_ospf_routes
from netkeeper_sim.topology.model import build_topology_from_edges


def test_forward_policy_is_satisfied(single_path_topology):
    table = compute_ospf_routes(single_path_topology)

    result = evaluate_policy_consistency(
        [ForwardPolicy("p1", "R1", "R3", "R2")],
        table,
    )

    assert result.consistency == 1.0
    assert result.satisfied == 1
    assert result.results[0].reason == "required_next_hop_present"


def test_forward_policy_is_unsatisfied(single_path_topology):
    table = compute_ospf_routes(single_path_topology)

    result = evaluate_policy_consistency(
        [ForwardPolicy("p1", "R1", "R3", "R9")],
        table,
    )

    assert result.consistency == 0.0
    assert result.unsatisfied == 1
    assert result.results[0].reason == "required_next_hop_absent"


def test_forward_policy_accepts_required_ecmp_next_hop(diamond_topology):
    table = compute_ospf_routes(diamond_topology)

    result = evaluate_policy_consistency(
        [
            ForwardPolicy("via-r2", "R1", "R4", "R2"),
            ForwardPolicy("via-r3", "R1", "R4", "R3"),
        ],
        table,
    )

    assert result.consistency == 1.0
    assert result.satisfied == 2


def test_reachable_policy_must_pass_on_single_path(single_path_topology):
    table = compute_ospf_routes(single_path_topology)

    result = evaluate_policy_consistency(
        [ReachablePolicy("p1", "R1", "R3", "R2")],
        table,
    )

    assert result.consistency == 1.0
    assert result.results[0].reason == "must_pass_on_some_path"


def test_reachable_policy_unsatisfied_when_node_is_absent(single_path_topology):
    table = compute_ospf_routes(single_path_topology)

    result = evaluate_policy_consistency(
        [ReachablePolicy("p1", "R1", "R3", "R9")],
        table,
    )

    assert result.consistency == 0.0
    assert result.results[0].reason == "must_pass_absent"


def test_reachable_policy_ecmp_any_path_and_all_paths(diamond_topology):
    table = compute_ospf_routes(diamond_topology)

    result = evaluate_policy_consistency(
        [
            ReachablePolicy("any", "R1", "R4", "R2", mode="any_path"),
            ReachablePolicy("all", "R1", "R4", "R2", mode="all_paths"),
        ],
        table,
    )

    assert result.consistency == 0.5
    assert result.results[0].satisfied is True
    assert result.results[1].satisfied is False
    assert result.results[1].reason == "must_pass_not_on_all_paths"


def test_reachable_policy_unsatisfied_when_destination_is_unreachable(single_path_topology):
    single_path_topology.fail_link("R2", "R3")
    table = compute_ospf_routes(single_path_topology)

    result = evaluate_policy_consistency(
        [ReachablePolicy("p1", "R1", "R3", "R2")],
        table,
    )

    assert result.consistency == 0.0
    assert result.results[0].reason == "no_reachable_path"


def test_policy_consistency_counts_partial_success(single_path_topology):
    table = compute_ospf_routes(single_path_topology)

    result = evaluate_policy_consistency(
        [
            ForwardPolicy("ok", "R1", "R3", "R2"),
            ForwardPolicy("bad", "R1", "R3", "R9"),
        ],
        table,
    )

    assert result.total == 2
    assert result.satisfied == 1
    assert result.unsatisfied == 1
    assert result.consistency == 0.5


def test_empty_policy_collection_is_explicitly_consistent(single_path_topology):
    table = compute_ospf_routes(single_path_topology)

    result = evaluate_policy_consistency([], table)

    assert result.total == 0
    assert result.consistency == 1.0


def test_unknown_node_or_prefix_is_unsatisfied(single_path_topology):
    table = compute_ospf_routes(single_path_topology)

    result = evaluate_policy_consistency(
        [
            ForwardPolicy("unknown-node", "R9", "R3", "R2"),
            ForwardPolicy(
                "unknown-prefix",
                "R1",
                "203.0.113.0/24",
                "R2",
                destination_type="prefix",
            ),
        ],
        table,
    )

    assert result.consistency == 0.0
    assert [policy_result.reason for policy_result in result.results] == [
        "missing_forwarding_entry",
        "missing_forwarding_entry",
    ]


def test_reachable_policy_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        ReachablePolicy("bad", "R1", "R2", "R3", mode="sometimes")  # type: ignore[arg-type]


def test_isolation_policy_forbidden_node_detects_shared_intermediate():
    topology = build_topology_from_edges(
        [
            ("R1", "C", 1),
            ("C", "R2", 1),
            ("R3", "C", 1),
            ("C", "R4", 1),
        ]
    )
    table = compute_ospf_routes(topology)

    result = evaluate_policy_consistency(
        [IsolationPolicy("iso", "R1", "R2", "R3", "R4", forbidden_node="C")],
        table,
    )

    assert result.consistency == 0.0
    assert result.results[0].reason == "forbidden_node_shared"


def test_isolation_policy_forbidden_node_allows_non_shared_intermediate():
    topology = build_topology_from_edges(
        [
            ("R1", "C", 1),
            ("C", "R2", 1),
            ("R3", "D", 1),
            ("D", "R4", 1),
        ]
    )
    table = compute_ospf_routes(topology)

    result = evaluate_policy_consistency(
        [IsolationPolicy("iso", "R1", "R2", "R3", "R4", forbidden_node="C")],
        table,
    )

    assert result.consistency == 1.0
    assert result.results[0].reason == "forbidden_node_not_shared"


def test_isolation_policy_path_disjoint_detects_shared_intermediate():
    topology = build_topology_from_edges(
        [
            ("R1", "C", 1),
            ("C", "R2", 1),
            ("R3", "C", 1),
            ("C", "R4", 1),
        ]
    )
    table = compute_ospf_routes(topology)

    result = evaluate_policy_consistency(
        [
            IsolationPolicy(
                "iso",
                "R1",
                "R2",
                "R3",
                "R4",
                mode="path_disjoint",
            )
        ],
        table,
    )

    assert result.consistency == 0.0
    assert result.results[0].reason == "paths_share_intermediate_node"
