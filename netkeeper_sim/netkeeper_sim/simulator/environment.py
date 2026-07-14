from __future__ import annotations

from pathlib import Path
from typing import Iterable

from netkeeper_sim.metrics.evaluation import EvaluationResult, evaluate_netkeeper_metrics
from netkeeper_sim.metrics.load import LinkLoadMetrics, calculate_link_load_metrics
from netkeeper_sim.metrics.traffic_shift import (
    ForwardingPlaneSnapshot,
    TrafficShiftMode,
    TrafficShiftResult,
    calculate_traffic_shift,
    capture_forwarding_plane_snapshot,
)
from netkeeper_sim.policies.evaluator import (
    PolicyConsistencyResult,
    evaluate_policy_consistency,
)
from netkeeper_sim.policies.model import Policy
from netkeeper_sim.routing.bgp import BGPRoute, BGPRouteTable, select_best_routes
from netkeeper_sim.routing.ospf import ForwardingTable, compute_ospf_routes
from netkeeper_sim.topology.loader import load_topology
from netkeeper_sim.topology.model import Topology, TopologyDefaults
from netkeeper_sim.traffic.matrix import TrafficMatrix
from netkeeper_sim.traffic.propagation import PropagationResult, propagate_traffic
from netkeeper_sim.schemas.models import (
    NetworkConfiguration as SchemaNetworkConfiguration,
    Topology as SchemaTopology,
    TrafficMatrix as SchemaTrafficMatrix,
)
from netkeeper_sim.simulator.deterministic import (
    DeterministicSimulationResult,
    simulate_deterministic,
)


class NetworkSimulationEnvironment:
    def __init__(self, defaults: TopologyDefaults | None = None) -> None:
        self.defaults = defaults or TopologyDefaults()
        self.topology: Topology | None = None
        self.forwarding_table: ForwardingTable | None = None
        self.bgp_routes: BGPRouteTable | None = None
        self.bgp_candidate_routes: dict[str, dict[str, list[BGPRoute]]] | None = None
        self.traffic_matrix: TrafficMatrix | None = None
        self.propagation_result: PropagationResult | None = None
        self.link_metrics: LinkLoadMetrics | None = None
        self.policies: list[Policy] = []

    def load_topology(self, path: str | Path) -> Topology:
        self.topology = load_topology(path, defaults=self.defaults)
        self.forwarding_table = None
        self.bgp_routes = None
        self.bgp_candidate_routes = None
        self.propagation_result = None
        self.link_metrics = None
        return self.topology

    def set_traffic_matrix(self, matrix: TrafficMatrix) -> None:
        self.traffic_matrix = matrix
        self.propagation_result = None
        self.link_metrics = None

    def set_policies(self, policies: Iterable[Policy]) -> None:
        self.policies = list(policies)

    def simulate_schema(
        self,
        topology: SchemaTopology,
        configuration: SchemaNetworkConfiguration,
        traffic: SchemaTrafficMatrix,
        *,
        previous: DeterministicSimulationResult | None = None,
    ) -> DeterministicSimulationResult:
        """Run the immutable schema-driven deterministic simulation kernel.

        This intentionally does not mutate the legacy environment fields.  It
        is the migration boundary for callers that need configuration version
        semantics, BGP prefix traffic, and directed performance accounting.
        """
        return simulate_deterministic(topology, configuration, traffic, previous=previous)

    def compute_ospf_routes(self) -> ForwardingTable:
        topology = self._require_topology()
        self.forwarding_table = compute_ospf_routes(topology)
        self.propagation_result = None
        self.link_metrics = None
        return self.forwarding_table

    def compute_bgp_routes(
        self,
        candidate_routes: dict[str, dict[str, list[BGPRoute]]] | None = None,
    ) -> BGPRouteTable:
        if candidate_routes is not None:
            self.bgp_candidate_routes = candidate_routes
        elif self.bgp_candidate_routes is None:
            raise RuntimeError("No BGP candidate routes have been set")
        forwarding_table = self.forwarding_table or self.compute_ospf_routes()
        self.bgp_routes = select_best_routes(self.bgp_candidate_routes, forwarding_table)
        return self.bgp_routes

    def propagate_traffic(self) -> PropagationResult:
        topology = self._require_topology()
        forwarding_table = self.forwarding_table or self.compute_ospf_routes()
        if self.traffic_matrix is None:
            raise RuntimeError("No traffic matrix has been set")
        self.propagation_result = propagate_traffic(
            topology,
            forwarding_table,
            self.traffic_matrix,
        )
        self.link_metrics = None
        return self.propagation_result

    def calculate_metrics(self) -> LinkLoadMetrics:
        topology = self._require_topology()
        propagation_result = self.propagation_result or self.propagate_traffic()
        self.link_metrics = calculate_link_load_metrics(
            topology,
            propagation_result.link_loads,
        )
        return self.link_metrics

    def evaluate_policies(self) -> PolicyConsistencyResult:
        forwarding_table = self.forwarding_table or self.compute_ospf_routes()
        return evaluate_policy_consistency(
            self.policies,
            forwarding_table,
            bgp_routes=self.bgp_routes,
        )

    def capture_forwarding_snapshot(
        self,
        include_self_entries: bool = False,
    ) -> ForwardingPlaneSnapshot:
        forwarding_table = self.forwarding_table or self.compute_ospf_routes()
        return capture_forwarding_plane_snapshot(
            forwarding_table,
            bgp_routes=self.bgp_routes,
            include_self_entries=include_self_entries,
        )

    def calculate_traffic_shift(
        self,
        previous_snapshot: ForwardingPlaneSnapshot,
        current_snapshot: ForwardingPlaneSnapshot,
        mode: TrafficShiftMode = "union",
    ) -> TrafficShiftResult:
        return calculate_traffic_shift(previous_snapshot, current_snapshot, mode=mode)

    def evaluate_metrics(
        self,
        previous_snapshot: ForwardingPlaneSnapshot | None = None,
        traffic_shift_mode: TrafficShiftMode = "union",
    ) -> EvaluationResult:
        topology = self._require_topology()
        forwarding_table = self.forwarding_table or self.compute_ospf_routes()
        if self.traffic_matrix is None and self.propagation_result is None:
            link_loads = {link_id: 0.0 for link_id in topology.links}
        else:
            propagation_result = self.propagation_result or self.propagate_traffic()
            link_loads = propagation_result.link_loads
        current_snapshot = self.capture_forwarding_snapshot()
        result = evaluate_netkeeper_metrics(
            topology,
            link_loads,
            policies=self.policies,
            forwarding_table=forwarding_table,
            bgp_routes=self.bgp_routes,
            previous_snapshot=previous_snapshot,
            current_snapshot=current_snapshot if previous_snapshot is not None else None,
            traffic_shift_mode=traffic_shift_mode,
        )
        self.link_metrics = result.load_metrics
        return result

    def fail_link(self, u: str, v: str) -> None:
        self._require_topology().fail_link(u, v)
        self.forwarding_table = None
        self.bgp_routes = None
        self.propagation_result = None
        self.link_metrics = None

    def restore_link(self, u: str, v: str) -> None:
        self._require_topology().restore_link(u, v)
        self.forwarding_table = None
        self.bgp_routes = None
        self.propagation_result = None
        self.link_metrics = None

    def update_ospf_weight(self, u: str, v: str, weight: float) -> None:
        self._require_topology().update_ospf_weight(u, v, weight)
        self.forwarding_table = None
        self.bgp_routes = None
        self.propagation_result = None
        self.link_metrics = None

    def reset(self) -> None:
        self.forwarding_table = None
        self.bgp_routes = None
        self.bgp_candidate_routes = None
        self.traffic_matrix = None
        self.propagation_result = None
        self.link_metrics = None
        self.policies = []

    def _require_topology(self) -> Topology:
        if self.topology is None:
            raise RuntimeError("No topology has been loaded")
        return self.topology
