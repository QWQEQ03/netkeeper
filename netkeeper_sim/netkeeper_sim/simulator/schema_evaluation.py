"""Produce an evaluated immutable snapshot from the deterministic kernel."""
from __future__ import annotations

from dataclasses import dataclass, replace

from netkeeper_sim.metrics.traffic_shift import (
    VersionedTrafficShiftResult, calculate_versioned_traffic_shift,
    capture_schema_forwarding_plane_snapshot,
)
from netkeeper_sim.policies.schema_evaluator import PolicyConsistencyReport, evaluate_schema_policies
from netkeeper_sim.schemas import Metrics, NetworkSnapshot
from netkeeper_sim.simulator.deterministic import DeterministicSimulationResult, simulate_deterministic


@dataclass(frozen=True)
class SnapshotEvaluation:
    snapshot: NetworkSnapshot
    simulation: DeterministicSimulationResult
    policy_consistency: PolicyConsistencyReport
    step_paper_v1: VersionedTrafficShiftResult | None
    total_paper_v1: VersionedTrafficShiftResult | None
    step_project_v1: VersionedTrafficShiftResult | None
    total_project_v1: VersionedTrafficShiftResult | None


def evaluate_schema_snapshot(
    snapshot: NetworkSnapshot,
    *,
    previous: NetworkSnapshot | None = None,
    initial: NetworkSnapshot | None = None,
) -> SnapshotEvaluation:
    simulation = simulate_deterministic(snapshot.topology, snapshot.configuration, snapshot.traffic)
    policies = evaluate_schema_policies(snapshot.policies, simulation.routing_table, snapshot.configuration, snapshot.topology, simulation.selected_bgp_routes)
    current_fib = capture_schema_forwarding_plane_snapshot(simulation.routing_table, simulation.selected_bgp_routes)
    def shifts(other):
        if other is None: return (None, None)
        # Recompute old selected BGP routes from its immutable configuration;
        # snapshots prior to this evaluator need not carry a duplicate FIB.
        old_simulation = simulate_deterministic(other.topology, other.configuration, other.traffic)
        old = capture_schema_forwarding_plane_snapshot(old_simulation.routing_table, old_simulation.selected_bgp_routes)
        return (calculate_versioned_traffic_shift(old, current_fib, "paper_v1"), calculate_versioned_traffic_shift(old, current_fib, "project_v1"))
    step_paper, step_project = shifts(previous)
    total_paper, total_project = shifts(initial)
    base = simulation.metrics
    metrics = replace(
        base,
        policy_consistency=policies.overall_consistency,
        policy_consistency_feasible_only=policies.feasible_only_consistency,
        policy_numerator=policies.numerator, policy_denominator=policies.denominator,
        policy_feasible_numerator=policies.feasible_numerator, policy_feasible_denominator=policies.feasible_denominator,
        policy_excluded_count=policies.excluded_count, policy_consistency_by_kind=policies.by_kind,
        traffic_shift_paper_v1=step_paper.shift_ratio if step_paper else None,
        traffic_shift_project_v1=step_project.shift_ratio if step_project else None,
        traffic_shift_step_paper_v1=step_paper.shift_ratio if step_paper else None,
        traffic_shift_total_paper_v1=total_paper.shift_ratio if total_paper else None,
        traffic_shift_step_project_v1=step_project.shift_ratio if step_project else None,
        traffic_shift_total_project_v1=total_project.shift_ratio if total_project else None,
    )
    evaluated = snapshot.next(routing_state=simulation.routing_table, directed_link_loads=simulation.directed_link_loads, metrics=metrics, policies=policies.evaluated_policies, step=snapshot.step)
    return SnapshotEvaluation(evaluated, simulation, policies, step_paper, total_paper, step_project, total_project)
