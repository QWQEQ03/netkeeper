"""Static operation registry.  It is deliberately data, never reflection."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping

from netkeeper_sim.api.ranges import RANGES

API_VERSION = "v1"
OBJECTIVES = ("policy_consistency", "mlu", "traffic_shift", "config_change")


def _obj(properties: Mapping[str, Any], required: tuple[str, ...], *, extra: bool = False) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": extra, "properties": dict(properties), "required": list(required)}


def _freeze_schema(value: Any) -> Any:
    if isinstance(value, Mapping): return MappingProxyType({key: _freeze_schema(item) for key, item in value.items()})
    if isinstance(value, list): return tuple(_freeze_schema(item) for item in value)
    if isinstance(value, tuple): return tuple(_freeze_schema(item) for item in value)
    return value


def _plain_schema(value: Any) -> Any:
    if isinstance(value, Mapping): return {key: _plain_schema(item) for key, item in value.items()}
    if isinstance(value, tuple): return [_plain_schema(item) for item in value]
    return value


NODE = {"type": "string", "pattern": "^R[0-9]+$"}
LINK = {"type": "string", "minLength": 1}
POLICY = {"type": "string", "minLength": 1}
NUMBER = {"type": "number"}
INTEGER = {"type": "integer"}
PATH_MODE = {"type": "string", "enum": ["all_path", "any_path"]}
BGP_TARGET = {"router_id": NODE, "prefix": {"type": "string", "minLength": 1}, "next_hop": NODE}


@dataclass(frozen=True)
class ApiDefinition:
    name: str
    version: str
    category: Literal["policy", "traffic", "topology", "config", "control"]
    arguments_schema: Mapping[str, Any]
    mutates_state: bool
    operation_type: Literal["event", "action", "read", "optimizer"]
    mapping: str
    description: str


def _definition(name: str, category: str, props: Mapping[str, Any], required: tuple[str, ...], mutates: bool, operation: str, mapping: str, description: str) -> ApiDefinition:
    return ApiDefinition(name, API_VERSION, category, _freeze_schema(_obj(props, required)), mutates, operation, mapping, description)


API_REGISTRY: Mapping[str, ApiDefinition] = MappingProxyType({
    "add_reachable_policy": _definition("add_reachable_policy", "policy", {"src": NODE, "dst": NODE, "policy_id": POLICY, "priority": {"type": "integer", "minimum": 0}}, ("src", "dst"), True, "event", "policy_add/reachable", "Require a source to reach a destination."),
    "add_forward_policy": _definition("add_forward_policy", "policy", {"src": NODE, "dst": NODE, "pass_node": NODE, "path_mode": PATH_MODE, "policy_id": POLICY, "priority": {"type": "integer", "minimum": 0}}, ("src", "dst", "pass_node"), True, "event", "policy_add/forward_pass", "Require traffic to traverse a waypoint."),
    "add_avoid_policy": _definition("add_avoid_policy", "policy", {"src": NODE, "dst": NODE, "avoid_node": NODE, "path_mode": PATH_MODE, "policy_id": POLICY, "priority": {"type": "integer", "minimum": 0}}, ("src", "dst", "avoid_node"), True, "event", "policy_add/forward_avoid", "Require traffic to avoid a waypoint."),
    "add_isolation_policy": _definition("add_isolation_policy", "policy", {"first_src": NODE, "first_dst": NODE, "second_src": NODE, "second_dst": NODE, "policy_id": POLICY, "priority": {"type": "integer", "minimum": 0}}, ("first_src", "first_dst", "second_src", "second_dst"), True, "event", "policy_add/isolation", "Require two OD flows to have disjoint intermediate nodes."),
    "remove_policy": _definition("remove_policy", "policy", {"policy_id": POLICY}, ("policy_id",), True, "event", "policy_remove", "Remove an existing policy by stable policy ID."),
    "set_traffic_demand": _definition("set_traffic_demand", "traffic", {"src": NODE, "dst": NODE, "demand_bps": {**NUMBER, "minimum": 0}}, ("src", "dst", "demand_bps"), True, "event", "traffic_set", "Set one existing OD demand in bps."),
    "scale_traffic_demand": _definition("scale_traffic_demand", "traffic", {"src": NODE, "dst": NODE, "factor": {**NUMBER, "exclusiveMinimum": 0}}, ("src", "dst", "factor"), True, "event", "traffic_set", "Scale one existing OD demand."),
    "set_traffic_hotspot": _definition("set_traffic_hotspot", "traffic", {"src": NODE, "dst": NODE, "demand_bps": {**NUMBER, "minimum": 0}}, ("src", "dst", "demand_bps"), True, "event", "hotspot_change", "Replace an existing OD demand as a hotspot."),
    "set_link_state": _definition("set_link_state", "topology", {"link_id": LINK, "state": {"type": "string", "enum": ["up", "down"]}}, ("link_id", "state"), True, "event", "link_up/link_down", "Set an existing physical link state."),
    "set_node_state": _definition("set_node_state", "topology", {"node_id": NODE, "state": {"type": "string", "enum": ["up", "down"]}}, ("node_id", "state"), True, "event", "node_up/node_down", "Set a router node state."),
    "set_ospf_weight": _definition("set_ospf_weight", "config", {"link_id": LINK, "weight": {"type": "integer", "minimum": RANGES["ospf_weight"][0], "maximum": RANGES["ospf_weight"][1]}}, ("link_id", "weight"), True, "action", "ospf_weight", "Set OSPF link weight."),
    "set_bgp_local_pref": _definition("set_bgp_local_pref", "config", {**BGP_TARGET, "value": {"type": "integer", "minimum": RANGES["local_preference"][0], "maximum": RANGES["local_preference"][1]}}, ("router_id", "prefix", "next_hop", "value"), True, "action", "local_preference", "Set BGP local preference."),
    "set_bgp_as_path_length": _definition("set_bgp_as_path_length", "config", {**BGP_TARGET, "length": {"type": "integer", "minimum": RANGES["as_path_length"][0], "maximum": RANGES["as_path_length"][1]}}, ("router_id", "prefix", "next_hop", "length"), True, "action", "as_path_length", "Set BGP AS path length."),
    "set_bgp_med": _definition("set_bgp_med", "config", {**BGP_TARGET, "value": {"type": "integer", "minimum": RANGES["med"][0], "maximum": RANGES["med"][1]}}, ("router_id", "prefix", "next_hop", "value"), True, "action", "med", "Set BGP MED."),
    "set_link_bandwidth": _definition("set_link_bandwidth", "config", {"link_id": LINK, "bandwidth_bps": {"type": "integer", "minimum": 1}}, ("link_id", "bandwidth_bps"), True, "action", "bandwidth_bps", "Set effective link bandwidth in bps."),
    "set_link_capacity": _definition("set_link_capacity", "config", {"link_id": LINK, "capacity_bps": {"type": "integer", "minimum": 1}}, ("link_id", "capacity_bps"), True, "action", "capacity_bps", "Set link capacity in bps."),
    "set_queue_length": _definition("set_queue_length", "config", {"link_id": LINK, "queue_packets": {"type": "integer", "minimum": RANGES["queue_packets"][0], "maximum": RANGES["queue_packets"][1]}}, ("link_id", "queue_packets"), True, "action", "queue_packets", "Set link queue length in packets."),
    "get_network_state": _definition("get_network_state", "control", {"include": {"type": "array", "items": {"type": "string", "enum": ["summary", "metrics", "policies", "links"]}, "uniqueItems": True}}, (), False, "read", "snapshot_summary", "Read a bounded network state summary."),
    "optimize_network": _definition("optimize_network", "control", {"objectives": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"type": "string", "enum": list(OBJECTIVES)}}, "max_steps": {"type": "integer", "minimum": 1, "maximum": 50}}, ("objectives",), False, "optimizer", "optimizer_dispatch", "Request a compatible optimization dispatcher."),
})


def api_definitions() -> tuple[ApiDefinition, ...]: return tuple(API_REGISTRY[name] for name in sorted(API_REGISTRY))


def export_json_schema() -> dict[str, Any]:
    calls = [{"type": "object", "additionalProperties": False, "properties": {"api": {"const": item.name}, "arguments": _plain_schema(item.arguments_schema)}, "required": ["api", "arguments"]} for item in api_definitions()]
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "https://netkeeper.local/schemas/api-request-v1.json", "type": "object", "additionalProperties": False,
            "properties": {"api_version": {"const": API_VERSION}, "request_id": {"type": "string", "minLength": 1}, "expected_snapshot_id": {"type": "string", "minLength": 1}, "calls": {"type": "array", "minItems": 1, "maxItems": 64, "items": {"oneOf": calls}}, "need_optimization": {"type": "boolean"}, "dry_run": {"type": "boolean"}}, "required": ["api_version", "request_id", "calls"], "$defs": {item.name: _plain_schema(item.arguments_schema) for item in api_definitions()}}


def export_response_json_schema() -> dict[str, Any]:
    error = {"type": "object", "additionalProperties": False, "properties": {"error_type": {"type": "string"}, "code": {"type": "string"}, "message": {"type": "string"}, "call_index": {"type": ["integer", "null"]}, "api": {"type": ["string", "null"]}, "details": {"type": "object", "additionalProperties": True}}, "required": ["error_type", "code", "message", "call_index", "api", "details"]}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "https://netkeeper.local/schemas/api-response-v1.json", "type": "object", "additionalProperties": False, "properties": {"success": {"type": "boolean"}, "request_id": {"type": "string"}, "applied_calls": {"type": "array", "items": {"type": "integer"}}, "before_snapshot_id": {"type": ["string", "null"]}, "after_snapshot_id": {"type": ["string", "null"]}, "configuration_diff": {"type": "object", "additionalProperties": True}, "metrics": {"type": "object", "additionalProperties": True}, "optimization_status": {"type": "string", "enum": ["not_requested", "pending", "unavailable", "dispatched", "failed"]}, "event_diff": {"type": "object", "additionalProperties": True}, "state": {"type": "object", "additionalProperties": True}, "errors": {"type": "array", "items": error}, "plan": {"type": ["object", "null"]}}, "required": ["success", "request_id", "applied_calls", "before_snapshot_id", "after_snapshot_id", "configuration_diff", "metrics", "optimization_status", "event_diff", "state", "errors", "plan"]}
