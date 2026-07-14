"""Pure three-layer validation and deterministic API planning."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import re
from typing import Any, Mapping

import networkx as nx

from netkeeper_sim.schemas import NetworkSnapshot
from netkeeper_sim.api.models import ApiCall, ApiError, ApiRequest, ExecutionPlan, ValidationResult
from netkeeper_sim.api.ranges import RANGES
from netkeeper_sim.api.registry import API_REGISTRY, API_VERSION, OBJECTIVES


_HIGH = {"policy", "traffic", "topology"}
_CONFIG = {"config"}


@dataclass(frozen=True)
class _Context:
    snapshot: NetworkSnapshot
    nodes: frozenset[str]
    links: frozenset[str]
    policies: frozenset[str]
    bgp_targets: frozenset[tuple[str, str, str]]
    demands: frozenset[tuple[str, str]]


def validate_request(request: ApiRequest | Mapping[str, Any], current_snapshot: NetworkSnapshot) -> ValidationResult:
    raw = request.to_dict() if isinstance(request, ApiRequest) else dict(request)
    errors = _top_level_errors(raw)
    if errors:
        return ValidationResult(False, tuple(_sort(errors)))
    if any(not isinstance(item, Mapping) for item in raw.get("calls", ())):
        return ValidationResult(False, (ApiError("schema", "INVALID_TYPE", "every call must be an object"),))
    request = ApiRequest.from_dict(raw)
    context = _context(current_snapshot)
    if request.expected_snapshot_id is not None and request.expected_snapshot_id != current_snapshot.snapshot_id:
        return ValidationResult(False, (ApiError("concurrency", "STALE_SNAPSHOT", "expected_snapshot_id does not match current snapshot", details={"expected_snapshot_id": request.expected_snapshot_id, "current_snapshot_id": current_snapshot.snapshot_id}),))
    errors = []
    for index, call in enumerate(request.calls):
        errors.extend(_call_structure(index, call))
        if call.api in API_REGISTRY:
            errors.extend(_entity_errors(index, call, context))
    errors.extend(_semantic_errors(request, context))
    if errors:
        return ValidationResult(False, tuple(_sort(errors)))
    return ValidationResult(True, (), _plan(request, context))


def _top_level_errors(raw: Mapping[str, Any]) -> list[ApiError]:
    allowed = {"api_version", "request_id", "calls", "expected_snapshot_id", "need_optimization", "dry_run"}; errors = []
    if not isinstance(raw, Mapping): return [ApiError("schema", "INVALID_REQUEST", "request must be an object")]
    for key in sorted(set(raw) - allowed): errors.append(ApiError("schema", "ADDITIONAL_PROPERTY", f"unexpected request field: {key}", details={"field": key}))
    if raw.get("api_version") != API_VERSION: errors.append(ApiError("schema", "UNSUPPORTED_VERSION", "api_version must be v1"))
    if not isinstance(raw.get("request_id"), str) or not raw.get("request_id"): errors.append(ApiError("schema", "INVALID_REQUEST_ID", "request_id must be a non-empty string"))
    if not isinstance(raw.get("calls"), list) or not 1 <= len(raw.get("calls", ())) <= 64: errors.append(ApiError("schema", "INVALID_CALLS", "calls must contain 1..64 entries"))
    for key in ("need_optimization", "dry_run"):
        if key in raw and type(raw[key]) is not bool: errors.append(ApiError("schema", "INVALID_TYPE", f"{key} must be boolean", details={"field": key}))
    if "expected_snapshot_id" in raw and (not isinstance(raw["expected_snapshot_id"], str) or not raw["expected_snapshot_id"]): errors.append(ApiError("schema", "INVALID_TYPE", "expected_snapshot_id must be a non-empty string"))
    return errors


def _call_structure(index: int, call: ApiCall) -> list[ApiError]:
    if call.api not in API_REGISTRY: return [_err(index, call.api, "schema", "UNSUPPORTED_API", "API is not in the static whitelist")]
    if not isinstance(call.arguments, Mapping): return [_err(index, call.api, "schema", "INVALID_TYPE", "arguments must be an object")]
    spec = API_REGISTRY[call.api].arguments_schema; props, required = set(spec["properties"]), set(spec["required"]); errors = []
    for key in sorted(set(call.arguments) - props): errors.append(_err(index, call.api, "schema", "ADDITIONAL_PROPERTY", f"unexpected argument: {key}", key))
    for key in sorted(required - set(call.arguments)): errors.append(_err(index, call.api, "schema", "MISSING_ARGUMENT", f"missing required argument: {key}", key))
    for key, value in call.arguments.items():
        schema = spec["properties"].get(key)
        if schema and not _matches(value, schema): errors.append(_err(index, call.api, "schema", "INVALID_TYPE", f"invalid value for {key}", key))
    return errors


def _matches(value: Any, schema: Mapping[str, Any]) -> bool:
    kind = schema.get("type")
    if kind == "integer" and (type(value) is not int or ("minimum" in schema and value < schema["minimum"]) or ("maximum" in schema and value > schema["maximum"])): return False
    if kind == "number" and (type(value) not in (int, float) or not isfinite(float(value)) or ("minimum" in schema and value < schema["minimum"]) or ("exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"])): return False
    if kind == "string" and (not isinstance(value, str) or not value or ("enum" in schema and value not in schema["enum"]) or ("pattern" in schema and re.match(schema["pattern"], value) is None)): return False
    if kind == "array" and (not isinstance(value, list) or ("minItems" in schema and len(value) < schema["minItems"]) or (schema.get("uniqueItems") and len(set(value)) != len(value)) or any(not _matches(v, schema.get("items", {})) for v in value)): return False
    return True


def _context(snapshot: NetworkSnapshot) -> _Context:
    return _Context(snapshot, frozenset(n.node_id for n in snapshot.topology.nodes), frozenset(l.link_id for l in snapshot.topology.links), frozenset(p.policy_id for p in snapshot.policies), frozenset((r.router_id, r.prefix, r.next_hop) for r in snapshot.configuration.bgp.routes), frozenset((d.source, d.destination) for d in snapshot.traffic.demands if d.destination is not None))


def _entity_errors(index: int, call: ApiCall, ctx: _Context) -> list[ApiError]:
    a, errors = call.arguments, []
    for field in ("src", "dst", "pass_node", "avoid_node", "node_id", "router_id", "next_hop", "first_src", "first_dst", "second_src", "second_dst"):
        if field in a and a[field] not in ctx.nodes: errors.append(_err(index, call.api, "entity", "UNKNOWN_NODE", f"unknown node: {a[field]}", field))
    if "link_id" in a and a["link_id"] not in ctx.links: errors.append(_err(index, call.api, "entity", "UNKNOWN_LINK", f"unknown link: {a['link_id']}", "link_id"))
    if call.api == "remove_policy" and a.get("policy_id") not in ctx.policies: errors.append(_err(index, call.api, "entity", "POLICY_NOT_FOUND", "policy_id does not exist", "policy_id"))
    if call.api.startswith("set_bgp_") and (a.get("router_id"), a.get("prefix"), a.get("next_hop")) not in ctx.bgp_targets: errors.append(_err(index, call.api, "entity", "UNKNOWN_BGP_ROUTE", "BGP route target does not exist"))
    if call.api in {"set_traffic_demand", "scale_traffic_demand", "set_traffic_hotspot"} and (a.get("src"), a.get("dst")) not in ctx.demands: errors.append(_err(index, call.api, "entity", "OD_NOT_FOUND", "traffic OD does not exist"))
    return errors


def _semantic_errors(request: ApiRequest, ctx: _Context) -> list[ApiError]:
    errors: list[ApiError] = []; phase = 0; config_targets: set[tuple[str, str]] = set(); traffic: set[tuple[str, str]] = set(); added: set[str] = set()
    node_state = dict(ctx.snapshot.configuration.node_states); link_state = dict(ctx.snapshot.configuration.link_states)
    graph = nx.Graph(); graph.add_nodes_from(ctx.nodes); graph.add_edges_from((l.source, l.target) for l in ctx.snapshot.topology.links)
    for i, call in enumerate(request.calls):
        definition = API_REGISTRY.get(call.api)
        if definition is None: continue
        next_phase = 0 if definition.operation_type == "read" else 1 if definition.category in _HIGH else 2 if definition.category in _CONFIG else 3
        if next_phase < phase or (phase == 3 and next_phase == 3): errors.append(_err(i, call.api, "semantic", "ORDER_VIOLATION", "calls must be read, high-level, config, then final optimizer"))
        phase = max(phase, next_phase); a = call.arguments
        if call.api == "optimize_network" and request.need_optimization: errors.append(_err(i, call.api, "semantic", "DUPLICATE_OPERATION", "need_optimization and optimize_network cannot both be requested"))
        if call.api in {"add_reachable_policy", "add_forward_policy", "add_avoid_policy", "set_traffic_demand", "scale_traffic_demand", "set_traffic_hotspot"} and a.get("src") == a.get("dst"): errors.append(_err(i, call.api, "semantic", "INVALID_ENDPOINTS", "src and dst must differ"))
        if call.api.startswith("add_") and call.api.endswith("policy"):
            required_nodes = ("src", "dst", "pass_node", "avoid_node", "first_src", "first_dst", "second_src", "second_dst")
            if any(field in a and node_state.get(a[field], "up") == "down" for field in required_nodes): errors.append(_err(i, call.api, "state", "OBJECT_DOWN", "cannot create a policy referencing a down node"))
        if call.api in {"add_forward_policy", "add_avoid_policy"}:
            middle = a.get("pass_node", a.get("avoid_node"))
            if middle in {a.get("src"), a.get("dst")}: errors.append(_err(i, call.api, "semantic", "INVALID_WAYPOINT", "waypoint may not be an endpoint"))
            if call.api == "add_forward_policy" and all(v in ctx.nodes for v in (a.get("src"), a.get("dst"), middle)):
                reduced = graph.copy(); reduced.remove_nodes_from(set(ctx.nodes) - {a["src"], a["dst"], middle}) if False else []
                if not (nx.has_path(graph, a["src"], middle) and nx.has_path(graph, middle, a["dst"])): errors.append(_err(i, call.api, "semantic", "INFEASIBLE_WAYPOINT", "no structural path through pass_node"))
        if call.api == "add_isolation_policy":
            values = (a.get("first_src"), a.get("first_dst"), a.get("second_src"), a.get("second_dst"))
            if len(set(values)) != 4: errors.append(_err(i, call.api, "semantic", "OVERLAPPING_ISOLATION", "isolation endpoints must be non-overlapping"))
        if call.api in {"set_traffic_demand", "scale_traffic_demand"}:
            key = (str(a.get("src")), str(a.get("dst")))
            if key in traffic: errors.append(_err(i, call.api, "semantic", "TRAFFIC_OPERATION_CONFLICT", "only one set/scale operation per OD is allowed"))
            traffic.add(key)
        if call.api == "set_link_state":
            link = next((v for v in ctx.snapshot.topology.links if v.link_id == a.get("link_id")), None)
            if link and a.get("state") == "up" and (node_state.get(link.source, "up") == "down" or node_state.get(link.target, "up") == "down"): errors.append(_err(i, call.api, "semantic", "INVALID_STATE_TRANSITION", "cannot bring a link up while an endpoint is down"))
            if link: link_state[link.link_id] = a.get("state")
        if call.api == "set_node_state": node_state[a.get("node_id")] = a.get("state")
        if definition.operation_type == "action":
            target = (definition.mapping, str(a.get("link_id") or (a.get("router_id"), a.get("prefix"), a.get("next_hop"))))
            if target in config_targets: errors.append(_err(i, call.api, "semantic", "DUPLICATE_OPERATION", "parameter target may be changed once"))
            config_targets.add(target)
            link_id = a.get("link_id")
            if link_id and link_state.get(link_id, "up") == "down": errors.append(_err(i, call.api, "state", "OBJECT_DOWN", "cannot configure a down link"))
            if call.api.startswith("set_bgp_") and node_state.get(a.get("router_id"), "up") == "down": errors.append(_err(i, call.api, "state", "OBJECT_DOWN", "cannot configure a BGP route on a down router"))
            errors.extend(_ranges(i, call, ctx))
        if call.api.startswith("add_") and call.api.endswith("policy"):
            ident = a.get("policy_id") or f"P:api:{request.request_id}:{i}"
            if ident in ctx.policies or ident in added: errors.append(_err(i, call.api, "semantic", "DUPLICATE_POLICY_ID", "policy_id already exists"))
            added.add(ident)
    errors.extend(_policy_conflicts(request.calls, ctx))
    return errors


def _ranges(i: int, call: ApiCall, ctx: _Context) -> list[ApiError]:
    a = call.arguments; key = {"set_ospf_weight": ("weight", "ospf_weight"), "set_bgp_local_pref": ("value", "local_preference"), "set_bgp_as_path_length": ("length", "as_path_length"), "set_bgp_med": ("value", "med"), "set_queue_length": ("queue_packets", "queue_packets")}.get(call.api)
    if key:
        value = a.get(key[0]); lo, hi = RANGES[key[1]]
        if type(value) is not int or not lo <= value <= hi: return [_err(i, call.api, "range", "OUT_OF_RANGE", f"{key[0]} must be in [{lo}, {hi}]")]
    if call.api in {"set_link_bandwidth", "set_link_capacity"}:
        link = next((v for v in ctx.snapshot.topology.links if v.link_id == a.get("link_id")), None)
        if link:
            attrs = ctx.snapshot.configuration.performance[link.link_id]; value = a.get("bandwidth_bps", a.get("capacity_bps"))
            maximum = attrs.physical_bandwidth_bps if call.api == "set_link_bandwidth" else min(attrs.bandwidth_bps, attrs.capacity_max_bps)
            minimum = attrs.capacity_bps if call.api == "set_link_bandwidth" else 1
            if type(value) is not int or not minimum <= value <= maximum: return [_err(i, call.api, "range", "OUT_OF_RANGE", "link value violates physical/configuration bounds")]
    return []


def _policy_conflicts(calls: tuple[ApiCall, ...], ctx: _Context) -> list[ApiError]:
    forward: dict[tuple[str, str, str], tuple[int, str]] = {}; isolation_pairs = set()
    for policy in ctx.snapshot.policies:
        if policy.kind == "isolation": isolation_pairs.update(((policy.fields.get("first_source"), policy.fields.get("first_destination")), (policy.fields.get("second_source"), policy.fields.get("second_destination"))))
    errors = []
    for i, call in enumerate(calls):
        if call.api == "add_isolation_policy": isolation_pairs.update(((call.arguments.get("first_src"), call.arguments.get("first_dst")), (call.arguments.get("second_src"), call.arguments.get("second_dst"))))
        if call.api not in {"add_forward_policy", "add_avoid_policy", "add_reachable_policy"}: continue
        pair = (call.arguments.get("src"), call.arguments.get("dst"))
        if pair in isolation_pairs: errors.append(_err(i, call.api, "semantic", "POLICY_CONFLICT", "reachable/forward policy conflicts with an isolation endpoint pair"))
        if call.api in {"add_forward_policy", "add_avoid_policy"}:
            node = call.arguments.get("pass_node", call.arguments.get("avoid_node")); key = (pair[0], pair[1], node); kind = call.api
            if key in forward and forward[key][1] != kind:
                errors.append(_err(i, call.api, "semantic", "POLICY_CONFLICT", "cannot both pass and avoid the same node")); errors.append(_err(forward[key][0], forward[key][1], "semantic", "POLICY_CONFLICT", "cannot both pass and avoid the same node"))
            forward[key] = (i, kind)
    return errors


def _plan(request: ApiRequest, ctx: _Context) -> ExecutionPlan:
    indexed = list(enumerate(request.calls)); ordered = sorted(indexed, key=lambda item: (0 if API_REGISTRY[item[1].api].operation_type == "read" else 1 if API_REGISTRY[item[1].api].category in _HIGH else 2 if API_REGISTRY[item[1].api].category in _CONFIG else 3, item[0]))
    events = tuple({"call_index": i, "kind": API_REGISTRY[c.api].mapping, "arguments": dict(c.arguments)} for i, c in ordered if API_REGISTRY[c.api].operation_type == "event")
    actions = tuple({"call_index": i, "parameter_type": API_REGISTRY[c.api].mapping, "arguments": dict(c.arguments)} for i, c in ordered if API_REGISTRY[c.api].operation_type == "action")
    reads = tuple(i for i, c in ordered if API_REGISTRY[c.api].operation_type == "read")
    return ExecutionPlan(ctx.snapshot.snapshot_id, tuple(i for i, _ in ordered), events, actions, reads, request.need_optimization or any(c.api == "optimize_network" for c in request.calls), {"event_count": len(events), "action_count": len(actions), "configuration_version_delta": 1 if actions else 0})


def _err(index: int, api: str, error_type: str, code: str, message: str, field: str | None = None) -> ApiError: return ApiError(error_type, code, message, index, api, {"field": field} if field else {})
def _sort(errors: list[ApiError]) -> list[ApiError]: return sorted(errors, key=lambda e: (e.call_index is None, e.call_index if e.call_index is not None else -1, e.api or "", e.code, e.message))
