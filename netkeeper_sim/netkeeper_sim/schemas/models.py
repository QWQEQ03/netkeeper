"""Immutable JSON-serializable schemas used at the simulation boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal, Mapping

from netkeeper_sim.schemas.ids import (
    SCHEMA_VERSION,
    directed_arc_id,
    snapshot_id,
    stable_hash,
)


def _freeze_mapping(value: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


@dataclass(frozen=True)
class ErrorRecord:
    code: str
    message: str
    severity: Literal["info", "warning", "error"] = "error"
    field_path: str | None = None
    action_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _plain(self.__dict__)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ErrorRecord":
        return cls(**dict(data))


@dataclass(frozen=True)
class LinkAttributes:
    """Per-physical-link values in explicit base units."""

    physical_bandwidth_bps: int = 100_000_000
    bandwidth_bps: int = 100_000_000
    capacity_max_bps: int = 100_000_000
    capacity_bps: int = 100_000_000
    delay_ms: float = 1.0
    queue_packets: int = 1000
    packet_size_bytes: int = 1500
    loss_rate: float = 0.0
    ospf_weight: int = 1
    state: Literal["up", "down"] = "up"
    value_source: str = "default"

    def __post_init__(self) -> None:
        if self.physical_bandwidth_bps <= 0:
            raise ValueError("physical_bandwidth_bps must be positive")
        if not 0 < self.bandwidth_bps <= self.physical_bandwidth_bps:
            raise ValueError("bandwidth_bps must be in (0, physical_bandwidth_bps]")
        if self.capacity_max_bps <= 0:
            raise ValueError("capacity_max_bps must be positive")
        if not 0 < self.capacity_bps <= min(self.bandwidth_bps, self.capacity_max_bps):
            raise ValueError("capacity_bps must be in (0, min(bandwidth_bps, capacity_max_bps)]")
        if self.delay_ms < 0:
            raise ValueError("delay_ms must be non-negative")
        if self.queue_packets < 0:
            raise ValueError("queue_packets must be non-negative")
        if not 64 <= self.packet_size_bytes <= 9_000:
            raise ValueError("packet_size_bytes must be in [64, 9000]")
        if not 0.0 <= self.loss_rate <= 1.0:
            raise ValueError("loss_rate must be in [0, 1]")
        if not 1 <= self.ospf_weight <= 65535:
            raise ValueError("ospf_weight must be in [1, 65535]")
        if self.state not in ("up", "down"):
            raise ValueError("state must be 'up' or 'down'")

    def to_dict(self) -> dict[str, Any]:
        return _plain(self.__dict__)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LinkAttributes":
        return cls(**dict(data))


@dataclass(frozen=True)
class Node:
    node_id: str
    original_id: str
    original_label: str | None = None
    node_type: Literal["router", "prefix", "external_as", "other"] = "router"
    raw_attributes: Mapping[str, Any] = field(default_factory=_freeze_mapping)

    def __post_init__(self) -> None:
        if not self.node_id.startswith("R") or not self.node_id[1:].isdigit():
            raise ValueError("node_id must use the R<number> canonical form")
        object.__setattr__(self, "raw_attributes", _freeze_mapping(self.raw_attributes))

    def to_dict(self) -> dict[str, Any]:
        return _plain(self.__dict__)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Node":
        return cls(**dict(data))


@dataclass(frozen=True)
class Link:
    link_id: str
    source: str
    target: str
    parallel_ordinal: int
    attributes: LinkAttributes
    raw_edge_id: str | None = None
    raw_name: str | None = None
    raw_attributes: Mapping[str, Any] = field(default_factory=_freeze_mapping)

    def __post_init__(self) -> None:
        if self.source == self.target:
            raise ValueError("self-loop physical links are not supported by schema v1")
        if self.parallel_ordinal < 0:
            raise ValueError("parallel_ordinal must be non-negative")
        object.__setattr__(self, "raw_attributes", _freeze_mapping(self.raw_attributes))

    def arcs(self) -> tuple["DirectedArc", "DirectedArc"]:
        return (
            DirectedArc(directed_arc_id(self.link_id, self.source, self.target), self.link_id, self.source, self.target),
            DirectedArc(directed_arc_id(self.link_id, self.target, self.source), self.link_id, self.target, self.source),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "link_id": self.link_id,
            "source": self.source,
            "target": self.target,
            "parallel_ordinal": self.parallel_ordinal,
            "attributes": self.attributes.to_dict(),
            "raw_edge_id": self.raw_edge_id,
            "raw_name": self.raw_name,
            "raw_attributes": _plain(self.raw_attributes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Link":
        values = dict(data)
        values["attributes"] = LinkAttributes.from_dict(values["attributes"])
        return cls(**values)


@dataclass(frozen=True)
class DirectedArc:
    arc_id: str
    link_id: str
    source: str
    target: str

    def to_dict(self) -> dict[str, Any]:
        return _plain(self.__dict__)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DirectedArc":
        return cls(**dict(data))


@dataclass(frozen=True)
class Topology:
    topology_id: str
    normalized_name: str
    source_format: Literal["graphml", "gml", "synthetic"]
    source_sha256: str
    nodes: tuple[Node, ...]
    links: tuple[Link, ...]
    schema_version: str = SCHEMA_VERSION
    raw_name: str | None = None
    directed: bool = False
    defaults: Mapping[str, Any] = field(default_factory=_freeze_mapping)

    def __post_init__(self) -> None:
        nodes = tuple(sorted(self.nodes, key=lambda node: int(node.node_id[1:])))
        links = tuple(sorted(self.links, key=lambda link: link.link_id))
        if len({node.node_id for node in nodes}) != len(nodes):
            raise ValueError("node IDs must be unique")
        if len({link.link_id for link in links}) != len(links):
            raise ValueError("link IDs must be unique")
        known = {node.node_id for node in nodes}
        if any(link.source not in known or link.target not in known for link in links):
            raise ValueError("every link endpoint must be a known node")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "links", links)
        object.__setattr__(self, "defaults", _freeze_mapping(self.defaults))

    @property
    def arcs(self) -> tuple[DirectedArc, ...]:
        return tuple(arc for link in self.links for arc in link.arcs())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "topology_id": self.topology_id,
            "normalized_name": self.normalized_name,
            "source_format": self.source_format,
            "source_sha256": self.source_sha256,
            "raw_name": self.raw_name,
            "directed": self.directed,
            "defaults": _plain(self.defaults),
            "nodes": [node.to_dict() for node in self.nodes],
            "links": [link.to_dict() for link in self.links],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Topology":
        values = dict(data)
        values["nodes"] = tuple(Node.from_dict(item) for item in values.pop("nodes"))
        values["links"] = tuple(Link.from_dict(item) for item in values.pop("links"))
        return cls(**values)


@dataclass(frozen=True)
class BGPRoute:
    router_id: str
    prefix: str
    next_hop: str
    local_preference: int
    as_path: tuple[int, ...]
    med: int
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.local_preference < 1 or self.med < 0 or len(self.as_path) > 255:
            raise ValueError("invalid BGP parameter range")

    def to_dict(self) -> dict[str, Any]:
        return _plain(self.__dict__)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BGPRoute":
        values = dict(data)
        values["as_path"] = tuple(values["as_path"])
        return cls(**values)


@dataclass(frozen=True)
class BGPConfiguration:
    routes: tuple[BGPRoute, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "routes", tuple(sorted(self.routes, key=lambda route: (route.router_id, route.prefix, route.next_hop))))

    def to_dict(self) -> dict[str, Any]:
        return {"routes": [route.to_dict() for route in self.routes]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BGPConfiguration":
        return cls(tuple(BGPRoute.from_dict(item) for item in data.get("routes", ())))


@dataclass(frozen=True)
class NetworkConfiguration:
    topology_id: str
    version: int
    step: int
    ospf_weights: Mapping[str, int]
    bgp: BGPConfiguration = field(default_factory=BGPConfiguration)
    performance: Mapping[str, LinkAttributes] = field(default_factory=_freeze_mapping)
    link_states: Mapping[str, Literal["up", "down"]] = field(default_factory=_freeze_mapping)
    node_states: Mapping[str, Literal["up", "down"]] = field(default_factory=_freeze_mapping)
    parent_version: int | None = None

    def __post_init__(self) -> None:
        if self.version < 0 or self.step < 0:
            raise ValueError("version and step must be non-negative")
        weights = {str(key): int(value) for key, value in self.ospf_weights.items()}
        if any(not 1 <= value <= 65535 for value in weights.values()):
            raise ValueError("ospf weights must be in [1, 65535]")
        performance = dict(self.performance)
        if any(not isinstance(value, LinkAttributes) for value in performance.values()):
            raise TypeError("performance values must be LinkAttributes")
        link_states = dict(self.link_states)
        node_states = dict(self.node_states)
        if any(value not in ("up", "down") for value in (*link_states.values(), *node_states.values())):
            raise ValueError("link and node states must be up or down")
        object.__setattr__(self, "ospf_weights", _freeze_mapping(weights))
        object.__setattr__(self, "performance", _freeze_mapping(performance))
        object.__setattr__(self, "link_states", _freeze_mapping(link_states))
        object.__setattr__(self, "node_states", _freeze_mapping(node_states))

    @classmethod
    def initial(cls, topology: Topology, step: int = 0) -> "NetworkConfiguration":
        return cls(
            topology_id=topology.topology_id,
            version=0,
            step=step,
            ospf_weights={link.link_id: link.attributes.ospf_weight for link in topology.links},
            performance={link.link_id: link.attributes for link in topology.links},
            link_states={link.link_id: link.attributes.state for link in topology.links},
            node_states={node.node_id: "up" for node in topology.nodes},
        )

    def with_updates(
        self,
        *,
        ospf_weights: Mapping[str, int] | None = None,
        performance: Mapping[str, LinkAttributes] | None = None,
        link_states: Mapping[str, Literal["up", "down"]] | None = None,
        node_states: Mapping[str, Literal["up", "down"]] | None = None,
        bgp: BGPConfiguration | None = None,
        step: int | None = None,
    ) -> "NetworkConfiguration":
        next_values = {
            "ospf_weights": dict(self.ospf_weights),
            "performance": dict(self.performance),
            "link_states": dict(self.link_states),
            "node_states": dict(self.node_states),
            "bgp": self.bgp,
        }
        for key, update in (("ospf_weights", ospf_weights), ("performance", performance), ("link_states", link_states), ("node_states", node_states)):
            if update is not None:
                next_values[key].update(update)
        if bgp is not None:
            next_values["bgp"] = bgp
        changed = next_values != {
            "ospf_weights": dict(self.ospf_weights), "performance": dict(self.performance),
            "link_states": dict(self.link_states), "node_states": dict(self.node_states), "bgp": self.bgp,
        }
        return NetworkConfiguration(
            topology_id=self.topology_id,
            version=self.version + int(changed),
            step=self.step if step is None else step,
            parent_version=self.version if changed else self.parent_version,
            **next_values,
        )

    def diff(self, other: "NetworkConfiguration") -> Mapping[str, tuple[Any, Any]]:
        if self.topology_id != other.topology_id:
            raise ValueError("cannot diff configurations for different topologies")
        result: dict[str, tuple[Any, Any]] = {}
        for section in ("ospf_weights", "performance", "link_states", "node_states"):
            before = getattr(self, section)
            after = getattr(other, section)
            for key in sorted(set(before) | set(after)):
                if before.get(key) != after.get(key):
                    result[f"{section}.{key}"] = (before.get(key), after.get(key))
        if self.bgp != other.bgp:
            result["bgp"] = (self.bgp, other.bgp)
        return _freeze_mapping(result)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topology_id": self.topology_id, "version": self.version, "step": self.step,
            "parent_version": self.parent_version, "ospf_weights": _plain(self.ospf_weights),
            "bgp": self.bgp.to_dict(),
            "performance": {key: value.to_dict() for key, value in self.performance.items()},
            "link_states": _plain(self.link_states), "node_states": _plain(self.node_states),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NetworkConfiguration":
        values = dict(data)
        values["bgp"] = BGPConfiguration.from_dict(values.get("bgp", {}))
        values["performance"] = {key: LinkAttributes.from_dict(item) for key, item in values.get("performance", {}).items()}
        return cls(**values)


@dataclass(frozen=True)
class TrafficDemand:
    source: str
    traffic_rate_bps: float
    destination: str | None = None
    prefix: str | None = None
    traffic_class: str = "default"

    def __post_init__(self) -> None:
        if (self.destination is None) == (self.prefix is None):
            raise ValueError("exactly one of destination or prefix is required")
        if self.traffic_rate_bps < 0:
            raise ValueError("traffic_rate_bps must be non-negative")

    def to_dict(self) -> dict[str, Any]: return _plain(self.__dict__)
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrafficDemand": return cls(**dict(data))


@dataclass(frozen=True)
class TrafficMatrix:
    matrix_id: str
    node_order: tuple[str, ...]
    demands: tuple[TrafficDemand, ...]
    interval_seconds: float = 1.0
    generation_mode: Literal["csv", "matrix", "random_uniform", "gravity", "diurnal", "hotspot", "burst", "trace"] = "csv"
    load_multiplier: float = 1.0
    seed: int | None = None
    unit: Literal["bps"] = "bps"

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0 or self.load_multiplier <= 0:
            raise ValueError("interval_seconds and load_multiplier must be positive")
        object.__setattr__(self, "node_order", tuple(self.node_order))
        object.__setattr__(self, "demands", tuple(self.demands))

    def to_dict(self) -> dict[str, Any]:
        return {**_plain({key: value for key, value in self.__dict__.items() if key != "demands"}), "demands": [item.to_dict() for item in self.demands]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrafficMatrix":
        values = dict(data); values["node_order"] = tuple(values["node_order"]); values["demands"] = tuple(TrafficDemand.from_dict(item) for item in values["demands"]); return cls(**values)


@dataclass(frozen=True)
class Policy:
    policy_id: str
    kind: Literal["reachable", "forward_pass", "forward_avoid", "isolation"]
    fields: Mapping[str, Any]
    enabled: bool = True
    priority: int = 100
    status: Literal["pending", "satisfied", "unsatisfied", "conflict", "infeasible_due_to_failure", "invalid", "violated", "unsatisfiable"] = "pending"
    reason_code: str | None = None
    conflict_with: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", _freeze_mapping(self.fields))
        object.__setattr__(self, "conflict_with", tuple(sorted(self.conflict_with)))

    def to_dict(self) -> dict[str, Any]: return _plain(self.__dict__)
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Policy": return cls(**dict(data))


@dataclass(frozen=True)
class Event:
    event_id: str
    step: int
    kind: Literal["policy_add", "policy_remove", "policy_enable", "policy_disable", "traffic_set", "traffic_replace", "traffic_scale", "hotspot_change", "link_down", "link_up", "node_down", "node_up", "config_patch"]
    payload: Mapping[str, Any] = field(default_factory=_freeze_mapping)
    target_id: str | None = None
    source: Literal["dataset", "llm", "simulator", "manual"] = "dataset"

    def __post_init__(self) -> None:
        if self.step < 0: raise ValueError("event step must be non-negative")
        object.__setattr__(self, "payload", _freeze_mapping(self.payload))

    def to_dict(self) -> dict[str, Any]: return _plain(self.__dict__)
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Event": return cls(**dict(data))


@dataclass(frozen=True)
class NetworkScenario:
    """Complete immutable reset input for the schema-driven environment."""
    scenario_id: str
    topology: Topology
    traffic: TrafficMatrix
    policies: tuple[Policy, ...] = ()
    configuration: NetworkConfiguration | None = None
    events: tuple[Event, ...] = ()
    max_steps: int = 20
    target_mlu: float | None = None

    def __post_init__(self) -> None:
        if self.max_steps <= 0: raise ValueError("max_steps must be positive")
        if self.configuration is not None and self.configuration.topology_id != self.topology.topology_id:
            raise ValueError("scenario configuration topology_id must match")
        object.__setattr__(self, "policies", tuple(self.policies))
        object.__setattr__(self, "events", tuple(sorted(self.events, key=lambda item: (item.step, item.event_id))))


@dataclass(frozen=True)
class AgentObservation:
    snapshot_id: str
    global_state: Mapping[str, Any]
    local_observations: Mapping[str, Mapping[str, Any]]
    action_masks: Mapping[str, Mapping[str, bool]] = field(default_factory=_freeze_mapping)
    visible_fields: Mapping[str, tuple[str, ...]] = field(default_factory=_freeze_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "global_state", _freeze_mapping(self.global_state))
        object.__setattr__(self, "local_observations", _freeze_mapping({key: _freeze_mapping(value) for key, value in self.local_observations.items()}))
        object.__setattr__(self, "action_masks", _freeze_mapping({key: _freeze_mapping(value) for key, value in self.action_masks.items()}))
        object.__setattr__(self, "visible_fields", _freeze_mapping({key: tuple(value) for key, value in self.visible_fields.items()}))

    def to_dict(self) -> dict[str, Any]: return _plain(self.__dict__)


@dataclass(frozen=True)
class AtomicAction:
    agent_id: Literal["ospf", "bgp", "performance", "baseline", "llm"]
    parameter_type: str
    target: Mapping[str, str]
    mode: Literal["set", "delta", "no_update"] = "no_update"
    value: float | int | str | None = None
    mask: bool = True
    valid: bool | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("set", "delta", "no_update"): raise ValueError("invalid action mode")
        if self.mode == "no_update" and self.value is not None: raise ValueError("no_update action cannot carry a value")
        if self.mode != "no_update" and self.value is None: raise ValueError("set/delta action requires value")
        object.__setattr__(self, "target", _freeze_mapping(self.target))

    def to_dict(self) -> dict[str, Any]: return _plain(self.__dict__)


@dataclass(frozen=True)
class JointAction:
    actions: tuple[AtomicAction, ...]
    requested_by: str = "agent"
    snapshot_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", tuple(self.actions))

    def to_dict(self) -> dict[str, Any]: return {"actions": [item.to_dict() for item in self.actions], "requested_by": self.requested_by, "snapshot_id": self.snapshot_id}


@dataclass(frozen=True)
class RewardBreakdown:
    policy_reward: float = 0.0
    mlu_reward: float = 0.0
    traffic_shift_reward: float = 0.0
    configuration_change_penalty: float = 0.0
    illegal_action_penalty: float = 0.0
    dropped_traffic_penalty: float = 0.0
    total_reward: float = 0.0
    per_agent: Mapping[str, float] = field(default_factory=_freeze_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "per_agent", _freeze_mapping(self.per_agent))

    def to_dict(self) -> dict[str, Any]: return _plain(self.__dict__)


@dataclass(frozen=True)
class RoutingEntry:
    router_id: str
    destination: str
    destination_type: Literal["node", "prefix"]
    reachable: bool
    next_hops: tuple[str, ...]
    cost: float | None = None
    def to_dict(self) -> dict[str, Any]: return _plain(self.__dict__)
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RoutingEntry":
        values = dict(data); values["next_hops"] = tuple(values["next_hops"]); return cls(**values)


@dataclass(frozen=True)
class DirectedLinkLoad:
    arc_id: str
    # ``load_bps`` is retained for wire compatibility and is the raw offered
    # rate.  The explicit fields make loss/congestion accounting unambiguous.
    load_bps: float = 0.0
    delivered_bps: float = 0.0
    dropped_bps: float = 0.0
    utilization: float = 0.0
    admitted_bps: float = 0.0
    queue_dropped_bps: float = 0.0
    loss_dropped_bps: float = 0.0
    queue_occupancy_bits: float = 0.0
    delay_ms: float = 0.0
    congested: bool = False
    @property
    def offered_bps(self) -> float:
        return self.load_bps

    def to_dict(self) -> dict[str, Any]:
        return {**_plain(self.__dict__), "offered_bps": self.offered_bps}
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DirectedLinkLoad":
        values = dict(data)
        offered = values.pop("offered_bps", None)
        if offered is not None and "load_bps" in values and float(offered) != float(values["load_bps"]):
            raise ValueError("offered_bps must equal load_bps")
        if offered is not None and "load_bps" not in values:
            values["load_bps"] = offered
        return cls(**values)


@dataclass(frozen=True)
class Metrics:
    policy_consistency: float = 1.0
    maximum_link_utilization: float = 0.0
    traffic_shift_paper_v1: float | None = None
    traffic_shift_project_v1: float | None = None
    traffic_shift_step_paper_v1: float | None = None
    traffic_shift_total_paper_v1: float | None = None
    traffic_shift_step_project_v1: float | None = None
    traffic_shift_total_project_v1: float | None = None
    total_input_bps: float = 0.0
    delivered_bps: float = 0.0
    dropped_bps: float = 0.0
    unreachable_bps: float = 0.0
    admitted_bps: float = 0.0
    congestion_arc_count: int = 0
    policy_consistency_feasible_only: float = 1.0
    policy_numerator: int = 0
    policy_denominator: int = 0
    policy_feasible_numerator: int = 0
    policy_feasible_denominator: int = 0
    policy_excluded_count: int = 0
    policy_consistency_by_kind: Mapping[str, float] = field(default_factory=_freeze_mapping)
    def __post_init__(self) -> None:
        if not 0.0 <= self.policy_consistency <= 1.0: raise ValueError("policy_consistency must be in [0, 1]")
        if not 0.0 <= self.policy_consistency_feasible_only <= 1.0: raise ValueError("policy_consistency_feasible_only must be in [0, 1]")
        object.__setattr__(self, "policy_consistency_by_kind", _freeze_mapping(self.policy_consistency_by_kind))
    def to_dict(self) -> dict[str, Any]: return _plain(self.__dict__)
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Metrics": return cls(**dict(data))


@dataclass(frozen=True)
class NetworkSnapshot:
    step: int
    topology: Topology
    configuration: NetworkConfiguration
    traffic: TrafficMatrix
    policies: tuple[Policy, ...]
    routing_state: tuple[RoutingEntry, ...] = ()
    directed_link_loads: tuple[DirectedLinkLoad, ...] = ()
    metrics: Metrics = field(default_factory=Metrics)
    topology_state_version: int = 0
    schema_version: str = SCHEMA_VERSION
    snapshot_id: str | None = None

    def __post_init__(self) -> None:
        if self.step < 0 or self.topology_state_version < 0: raise ValueError("snapshot step and topology state version must be non-negative")
        if self.configuration.topology_id != self.topology.topology_id: raise ValueError("configuration topology_id must match snapshot topology")
        object.__setattr__(self, "policies", tuple(self.policies))
        object.__setattr__(self, "routing_state", tuple(self.routing_state))
        object.__setattr__(self, "directed_link_loads", tuple(self.directed_link_loads))
        expected = snapshot_id(self.topology.topology_id, self.topology_state_version, self.configuration.version, self.step)
        if self.snapshot_id is not None and self.snapshot_id != expected: raise ValueError("snapshot_id does not match snapshot versions")
        object.__setattr__(self, "snapshot_id", expected)

    def next(self, *, configuration: NetworkConfiguration | None = None, step: int | None = None, **changes: Any) -> "NetworkSnapshot":
        return replace(self, configuration=configuration or self.configuration, step=self.step + 1 if step is None else step, snapshot_id=None, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "snapshot_id": self.snapshot_id, "step": self.step, "topology_state_version": self.topology_state_version, "topology": self.topology.to_dict(), "configuration": self.configuration.to_dict(), "traffic": self.traffic.to_dict(), "policies": [item.to_dict() for item in self.policies], "routing_state": [item.to_dict() for item in self.routing_state], "directed_link_loads": [item.to_dict() for item in self.directed_link_loads], "metrics": self.metrics.to_dict()}

    def to_json(self) -> str: return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NetworkSnapshot":
        values = dict(data)
        values["topology"] = Topology.from_dict(values["topology"]); values["configuration"] = NetworkConfiguration.from_dict(values["configuration"]); values["traffic"] = TrafficMatrix.from_dict(values["traffic"]); values["policies"] = tuple(Policy.from_dict(item) for item in values["policies"]); values["routing_state"] = tuple(RoutingEntry.from_dict(item) for item in values.get("routing_state", ())); values["directed_link_loads"] = tuple(DirectedLinkLoad.from_dict(item) for item in values.get("directed_link_loads", ())); values["metrics"] = Metrics.from_dict(values["metrics"]); return cls(**values)

    @classmethod
    def from_json(cls, value: str) -> "NetworkSnapshot": return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class StepResult:
    previous_snapshot_id: str
    next_snapshot: NetworkSnapshot
    observations: AgentObservation
    rewards: RewardBreakdown
    terminated: bool
    truncated: bool
    done_reason: str | None
    changed_config: Mapping[str, tuple[Any, Any]]
    metrics: Metrics
    errors: tuple[ErrorRecord, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "changed_config", _freeze_mapping(self.changed_config))
        object.__setattr__(self, "errors", tuple(self.errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "previous_snapshot_id": self.previous_snapshot_id, "next_snapshot": self.next_snapshot.to_dict(),
            "observations": self.observations.to_dict(), "rewards": self.rewards.to_dict(),
            "terminated": self.terminated, "truncated": self.truncated, "done_reason": self.done_reason,
            "changed_config": _plain(self.changed_config), "metrics": self.metrics.to_dict(),
            "errors": [item.to_dict() for item in self.errors],
        }


@dataclass(frozen=True)
class ExperimentResult:
    run_id: str
    method: str
    topology_id: str
    scenario_id: str
    seed: int
    episode: int
    step: int
    metrics: Metrics
    wall_time_ms: float
    configuration_version: int
    event_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        values = _plain({key: value for key, value in self.__dict__.items() if key != "metrics"}); values["metrics"] = self.metrics.to_dict(); return values

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentResult":
        values = dict(data); values["metrics"] = Metrics.from_dict(values["metrics"]); return cls(**values)
