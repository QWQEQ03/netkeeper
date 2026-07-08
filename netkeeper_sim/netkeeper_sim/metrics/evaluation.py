from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from typing import Iterable

from netkeeper_sim.metrics.load import LinkLoadMetrics, calculate_link_load_metrics
from netkeeper_sim.metrics.traffic_shift import (
    ForwardingPlaneSnapshot,
    TrafficShiftMode,
    TrafficShiftResult,
    calculate_traffic_shift,
)
from netkeeper_sim.policies.evaluator import (
    PolicyConsistencyResult,
    evaluate_policy_consistency,
)
from netkeeper_sim.policies.model import Policy
from netkeeper_sim.routing.bgp import BGPRouteTable
from netkeeper_sim.routing.ospf import ForwardingTable
from netkeeper_sim.topology.model import Topology


@dataclass(frozen=True)
class EvaluationResult:
    policy_consistency: PolicyConsistencyResult
    maximum_link_utilization: float
    average_link_utilization: float
    link_utilizations: dict[str, float]
    overloaded_links: tuple[str, ...]
    traffic_shift: TrafficShiftResult | None
    load_metrics: LinkLoadMetrics


def evaluate_netkeeper_metrics(
    topology: Topology,
    link_loads: dict[str, float],
    policies: Iterable[Policy] = (),
    forwarding_table: ForwardingTable | None = None,
    bgp_routes: BGPRouteTable | None = None,
    previous_snapshot: ForwardingPlaneSnapshot | None = None,
    current_snapshot: ForwardingPlaneSnapshot | None = None,
    traffic_shift_mode: TrafficShiftMode = "union",
    overload_threshold: float = 1.0,
) -> EvaluationResult:
    load_metrics = calculate_link_load_metrics(topology, link_loads)
    policy_list = list(policies)
    if policy_list and forwarding_table is None:
        raise ValueError("forwarding_table is required when policies are provided")
    policy_consistency = evaluate_policy_consistency(
        policy_list,
        forwarding_table or {},
        bgp_routes=bgp_routes,
    )
    traffic_shift = (
        calculate_traffic_shift(previous_snapshot, current_snapshot, mode=traffic_shift_mode)
        if previous_snapshot is not None and current_snapshot is not None
        else None
    )
    utilizations = dict(load_metrics.utilization)
    average = _average_utilization(utilizations)
    overloaded = tuple(
        link_id
        for link_id, utilization in sorted(utilizations.items())
        if utilization > overload_threshold
    )
    return EvaluationResult(
        policy_consistency=policy_consistency,
        maximum_link_utilization=load_metrics.maximum_link_utilization,
        average_link_utilization=average,
        link_utilizations=utilizations,
        overloaded_links=overloaded,
        traffic_shift=traffic_shift,
        load_metrics=load_metrics,
    )


def _average_utilization(utilizations: dict[str, float]) -> float:
    if not utilizations:
        return 0.0
    return fsum(utilizations.values()) / len(utilizations)
