"""Versioned, immutable interchange schemas for NetKeeper Sim.

The legacy simulation models remain available under :mod:`netkeeper_sim.topology`
and related packages.  These schemas are the stable boundary for new code.
"""

from netkeeper_sim.schemas.ids import SCHEMA_VERSION
from netkeeper_sim.schemas.adapters import legacy_topology_from_schema
from netkeeper_sim.schemas.loader import load_schema_topology
from netkeeper_sim.schemas.models import (
    BGPConfiguration,
    BGPRoute,
    AgentObservation,
    AtomicAction,
    DirectedArc,
    DirectedLinkLoad,
    ErrorRecord,
    Event,
    ExperimentResult,
    Link,
    LinkAttributes,
    Metrics,
    NetworkConfiguration,
    NetworkScenario,
    NetworkSnapshot,
    Node,
    Policy,
    RewardBreakdown,
    RoutingEntry,
    TrafficDemand,
    TrafficMatrix,
    Topology,
    JointAction,
    StepResult,
)

__all__ = [
    "SCHEMA_VERSION",
    "BGPConfiguration",
    "BGPRoute",
    "AgentObservation",
    "AtomicAction",
    "DirectedArc",
    "DirectedLinkLoad",
    "ErrorRecord",
    "Event",
    "ExperimentResult",
    "Link",
    "LinkAttributes",
    "Metrics",
    "NetworkConfiguration",
    "NetworkScenario",
    "NetworkSnapshot",
    "Node",
    "Policy",
    "RewardBreakdown",
    "RoutingEntry",
    "TrafficDemand",
    "TrafficMatrix",
    "Topology",
    "JointAction",
    "StepResult",
    "load_schema_topology",
    "legacy_topology_from_schema",
]
