"""Transactional bridge from the safe API plan to UnifiedNetworkEnvironment."""
from __future__ import annotations

from dataclasses import replace
from math import isfinite
from typing import Any, Protocol

from netkeeper_sim.api.models import ApiError, ApiRequest, ApiResponse, ExecutionPlan, OptimizationRequest
from netkeeper_sim.api.registry import API_REGISTRY
from netkeeper_sim.api.validator import validate_request
from netkeeper_sim.schemas import AtomicAction, Event, JointAction, NetworkSnapshot, Policy, TrafficDemand
from netkeeper_sim.simulator import UnifiedNetworkEnvironment


class OptimizerDispatcher(Protocol):
    def dispatch(self, snapshot: NetworkSnapshot, request: OptimizationRequest) -> JointAction: ...


def execute(env: UnifiedNetworkEnvironment, snapshot: NetworkSnapshot, request: ApiRequest | dict[str, Any], *, dispatcher: OptimizerDispatcher | None = None) -> ApiResponse:
    """Validate then submit a request once; no input schema object is mutated.

    A mixed get_network_state call returns the post-commit bounded summary.
    The validator requires reads before writes, so it cannot influence writes.
    """
    request_id = request.request_id if isinstance(request, ApiRequest) else str(request.get("request_id", ""))
    before = snapshot.snapshot_id
    if env.current_snapshot is None or env.current_snapshot.snapshot_id != before:
        return _failure(request_id, before, ApiError("concurrency", "STALE_SNAPSHOT", "snapshot is not the environment current snapshot", details={"current_snapshot_id": env.current_snapshot.snapshot_id if env.current_snapshot else None}))
    checked = validate_request(request, snapshot)
    if not checked.valid:
        return ApiResponse(False, request_id, before_snapshot_id=before, after_snapshot_id=None, errors=checked.errors, plan=checked.plan)
    assert checked.plan is not None
    req = request if isinstance(request, ApiRequest) else ApiRequest.from_dict(request)
    if checked.plan.ordered_calls != tuple(range(len(req.calls))):
        return _failure(req.request_id, before, ApiError("execution", "PLAN_ORDER_INVALID", "validator plan must preserve validated user call order"), checked.plan)
    try:
        events, actions = _materialize(req, snapshot)
        optimization_status = "not_requested"
        # Dry runs expose an OptimizationRequest in the plan but never call a
        # dispatcher, which might otherwise have external side effects.
        if checked.plan.optimization_requested and not req.dry_run:
            if dispatcher is None:
                return _failure(req.request_id, before, ApiError("optimizer", "OPTIMIZER_UNAVAILABLE", "no compatible optimizer dispatcher is registered"), checked.plan, "unavailable")
            extra = dispatcher.dispatch(snapshot, _optimization_request(req, snapshot))  # type: ignore[union-attr]
            if not isinstance(extra, JointAction):
                return _failure(req.request_id, before, ApiError("optimizer", "OPTIMIZER_INVALID_RESULT", "dispatcher must return JointAction"), checked.plan, "failed")
            if extra.snapshot_id not in (None, snapshot.snapshot_id):
                return _failure(req.request_id, before, ApiError("optimizer", "STALE_SNAPSHOT", "dispatcher returned an action for another snapshot"), checked.plan, "failed")
            actions.extend(extra.actions); optimization_status = "dispatched"
        runtime_plan = replace(checked.plan, events=tuple(event.to_dict() for event in events), actions=tuple(action.to_dict() for action in actions))
    except (KeyError, TypeError, ValueError) as exc:
        return _failure(req.request_id, before, ApiError("execution", "MAPPING_FAILED", "request could not be mapped to unified operations", details={"reason": str(exc)}), checked.plan)
    if req.dry_run:
        return ApiResponse(True, req.request_id, tuple(range(len(req.calls))), before, before, optimization_status="pending" if runtime_plan.optimization_requested else "not_requested", plan=runtime_plan, state=_state_summary(snapshot))
    # A query-only request is a true read: it does not advance the step or
    # consume scheduled scenario events.  Mixed reads return post-commit state.
    if not events and not actions and not runtime_plan.optimization_requested:
        return ApiResponse(True, req.request_id, tuple(range(len(req.calls))), before, before, metrics=snapshot.metrics.to_dict(), state=_state_summary(snapshot), plan=runtime_plan)
    original_current = env.current_snapshot
    try:
        # Events deliberately advance topology_state_version before actions.
        # The executor already performed the stronger current-snapshot stale
        # check above, so the internal action ID is omitted for this one
        # combined transaction rather than becoming stale after its own events.
        result = env.step(snapshot, JointAction(tuple(actions), requested_by="api", snapshot_id=None), scheduled_events=tuple(events))
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        env.current_snapshot = original_current
        return _failure(req.request_id, before, ApiError("execution", "ENVIRONMENT_REJECTED", "unified environment rejected the planned batch", details={"reason": str(exc)}), runtime_plan, optimization_status)
    if result.errors:
        env.current_snapshot = original_current
        return _failure(req.request_id, before, ApiError("execution", "ENVIRONMENT_REJECTED", "unified environment reported planned operation errors", details={"errors": [error.to_dict() for error in result.errors]}), runtime_plan, optimization_status)
    next_snapshot = result.next_snapshot
    return ApiResponse(True, req.request_id, tuple(range(len(req.calls))), before, next_snapshot.snapshot_id, result.changed_config, result.metrics.to_dict(), optimization_status, {"api_event_ids": [event.event_id for event in events], "policy_count_before": len(snapshot.policies), "policy_count_after": len(next_snapshot.policies), "traffic_matrix_changed": snapshot.traffic != next_snapshot.traffic, "topology_state_version_before": snapshot.topology_state_version, "topology_state_version_after": next_snapshot.topology_state_version}, _state_summary(next_snapshot), (), runtime_plan)


def _materialize(request: ApiRequest, snapshot: NetworkSnapshot) -> tuple[list[Event], list[AtomicAction]]:
    events: list[Event] = []; actions: list[AtomicAction] = []; traffic = snapshot.traffic
    for index, call in enumerate(request.calls):
        definition = API_REGISTRY[call.api]; a = call.arguments
        if definition.operation_type == "event":
            event, traffic = _event_for(request, snapshot, index, call.api, a, traffic); events.append(event)
        elif definition.operation_type == "action": actions.append(_action_for(call.api, a))
    return events, actions


def _event_for(request: ApiRequest, snapshot: NetworkSnapshot, index: int, api: str, a: dict[str, Any], traffic):
    event_id = f"API:{request.request_id}:{index:04d}"
    if api.startswith("add_") and api.endswith("policy"):
        ident = a.get("policy_id") or f"P:api:{request.request_id}:{index}"
        if api == "add_reachable_policy": kind, fields = "reachable", {"source": a["src"], "destination": a["dst"]}
        elif api == "add_forward_policy": kind, fields = "forward_pass", {"source": a["src"], "destination": a["dst"], "waypoint": a["pass_node"], "path_mode": a.get("path_mode", "all_path")}
        elif api == "add_avoid_policy": kind, fields = "forward_avoid", {"source": a["src"], "destination": a["dst"], "forbidden_node": a["avoid_node"], "path_mode": a.get("path_mode", "all_path")}
        else: kind, fields = "isolation", {"first_source": a["first_src"], "first_destination": a["first_dst"], "second_source": a["second_src"], "second_destination": a["second_dst"], "resource": "node"}
        return Event(event_id, snapshot.step, "policy_add", {"policy": Policy(ident, kind, fields, priority=a.get("priority", 100)).to_dict()}, source="manual"), traffic
    if api == "remove_policy": return Event(event_id, snapshot.step, "policy_remove", target_id=a["policy_id"], source="manual"), traffic
    if api == "set_link_state": return Event(event_id, snapshot.step, "link_up" if a["state"] == "up" else "link_down", target_id=a["link_id"], source="manual"), traffic
    if api == "set_node_state": return Event(event_id, snapshot.step, "node_up" if a["state"] == "up" else "node_down", target_id=a["node_id"], source="manual"), traffic
    if api in {"set_traffic_demand", "scale_traffic_demand", "set_traffic_hotspot"}:
        demands = []
        for demand in traffic.demands:
            if (demand.source, demand.destination) != (a["src"], a["dst"]): demands.append(demand); continue
            value = float(a["demand_bps"]) if api != "scale_traffic_demand" else demand.traffic_rate_bps * float(a["factor"])
            if not isfinite(value) or value < 0: raise ValueError("traffic operation produced a non-finite demand")
            demands.append(TrafficDemand(demand.source, value, destination=demand.destination, prefix=demand.prefix, traffic_class=demand.traffic_class))
        traffic = replace(traffic, demands=tuple(demands))
        if api == "set_traffic_hotspot":
            demand = next(item for item in traffic.demands if (item.source, item.destination) == (a["src"], a["dst"]))
            return Event(event_id, snapshot.step, "hotspot_change", {"demand": demand.to_dict()}, source="manual"), traffic
        return Event(event_id, snapshot.step, "traffic_set", {"traffic": traffic.to_dict()}, source="manual"), traffic
    raise ValueError("unsupported event mapping")


def _action_for(api: str, a: dict[str, Any]) -> AtomicAction:
    if api == "set_ospf_weight": return AtomicAction("ospf", "ospf_weight", {"link_id": a["link_id"]}, "set", a["weight"])
    if api in {"set_link_bandwidth", "set_link_capacity", "set_queue_length"}:
        parameter, value_key = {"set_link_bandwidth": ("bandwidth_bps", "bandwidth_bps"), "set_link_capacity": ("capacity_bps", "capacity_bps"), "set_queue_length": ("queue_packets", "queue_packets")}[api]
        return AtomicAction("performance", parameter, {"link_id": a["link_id"]}, "set", a[value_key])
    parameter, value_key = {"set_bgp_local_pref": ("local_preference", "value"), "set_bgp_as_path_length": ("as_path_length", "length"), "set_bgp_med": ("med", "value")}[api]
    return AtomicAction("bgp", parameter, {"router_id": a["router_id"], "prefix": a["prefix"], "next_hop": a["next_hop"]}, "set", a[value_key])


def _optimization_request(request: ApiRequest, snapshot: NetworkSnapshot) -> OptimizationRequest:
    optimize = next((call.arguments for call in request.calls if call.api == "optimize_network"), {})
    return OptimizationRequest(snapshot.snapshot_id, tuple(optimize.get("objectives", ("policy_consistency", "mlu", "traffic_shift", "config_change"))), optimize.get("max_steps"), request.request_id)


def _state_summary(snapshot: NetworkSnapshot) -> dict[str, Any]:
    return {"snapshot_id": snapshot.snapshot_id, "topology_id": snapshot.topology.topology_id, "step": snapshot.step, "configuration_version": snapshot.configuration.version, "topology_state_version": snapshot.topology_state_version, "node_count": len(snapshot.topology.nodes), "link_count": len(snapshot.topology.links), "policy_count": len(snapshot.policies), "traffic_matrix_id": snapshot.traffic.matrix_id, "traffic_load_multiplier": snapshot.traffic.load_multiplier, "metrics": snapshot.metrics.to_dict()}


def _failure(request_id: str, before: str, error: ApiError, plan: ExecutionPlan | None = None, optimization_status: str = "not_requested") -> ApiResponse:
    return ApiResponse(False, request_id, before_snapshot_id=before, after_snapshot_id=None, optimization_status=optimization_status, errors=(error,), plan=plan)
