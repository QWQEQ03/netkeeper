"""Snapshot-only graph, observation and action adapters for RL.

This module is deliberately the only conversion boundary between the immutable
schema simulator and tensor code.  It never reads scheduled future events.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from math import inf
from math import log1p
from typing import Iterable, Mapping

import torch

from netkeeper_sim.schemas import AtomicAction, JointAction, NetworkSnapshot


ACTION_VALUES = 64
NO_UPDATE = 0
PARAMETERS = {
    "ospf": ("ospf_weight",),
    "bgp": ("local_preference", "as_path_length", "med"),
    "performance": ("bandwidth_bps", "capacity_bps", "queue_packets"),
}


@dataclass(frozen=True, order=True)
class RouteTarget:
    router_id: str
    prefix: str
    next_hop: str


@dataclass(frozen=True)
class SnapshotGraph:
    """Tensor state plus stable schema target mappings.

    Node features (18): type router/prefix/external/other, active, normalized
    degree, policy endpoint, mean incident OSPF, bandwidth/physical-bandwidth,
    capacity/bandwidth, log queue, loss, incident utilization, BGP-route
    presence, mean local-pref, AS-path length, MED and global policy
    consistency.  Edge features (11):
    link/type, endpoint-up, OSPF, bandwidth/physical, capacity/bandwidth,
    log queue, loss, utilization, delay, source-up and target-up.
    """
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    node_mask: torch.Tensor
    edge_mask: torch.Tensor
    node_ids: tuple[str, ...]
    link_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    route_targets: tuple[RouteTarget, ...]
    policy_observation: torch.Tensor
    utilization_observation: torch.Tensor
    local_observations: Mapping[str, Mapping[str, torch.Tensor]]
    snapshot_id: str
    topology_id: str
    configuration_version: int


def snapshot_to_graph(snapshot: NetworkSnapshot) -> SnapshotGraph:
    nodes = tuple(node.node_id for node in snapshot.topology.nodes)
    links = tuple(link.link_id for link in snapshot.topology.links)
    node_index = {node_id: index for index, node_id in enumerate(nodes)}
    states = snapshot.configuration.link_states
    node_states = snapshot.configuration.node_states
    arc_to_link = {arc.arc_id: arc.link_id for arc in snapshot.topology.arcs}
    loads = {arc_to_link.get(item.arc_id, ""): max(loads_value, item.utilization) for item in snapshot.directed_link_loads for loads_value in (0.0,)}
    # Directed loads have two arcs per physical link; max is a fixed, current-only
    # aggregation and preserves the physical action target.
    for item in snapshot.directed_link_loads:
        link_id = arc_to_link.get(item.arc_id)
        if link_id is not None: loads[link_id] = max(loads.get(link_id, 0.0), float(item.utilization))
    routes = tuple(RouteTarget(route.router_id, route.prefix, route.next_hop) for route in snapshot.configuration.bgp.routes)
    policy_nodes = _policy_nodes(snapshot)
    degree = {node: 0 for node in nodes}
    for link in snapshot.topology.links:
        degree[link.source] += 1; degree[link.target] += 1
    route_by_node = {node: [route for route in snapshot.configuration.bgp.routes if route.router_id == node] for node in nodes}
    node_rows = []
    for node in snapshot.topology.nodes:
        routes_here = route_by_node[node.node_id]
        incident = [link for link in snapshot.topology.links if node.node_id in (link.source, link.target)]
        attrs = [snapshot.configuration.performance[link.link_id] for link in incident]
        def mean(values):
            values = tuple(values)
            return sum(values) / len(values) if values else 0.0
        node_rows.append([
            float(node.node_type == "router"), float(node.node_type == "prefix"), float(node.node_type == "external_as"), float(node.node_type == "other"),
            float(node_states.get(node.node_id, "up") == "up"), min(1.0, degree[node.node_id] / max(1, len(nodes) - 1)), float(node.node_id in policy_nodes),
            min(1.0, mean(snapshot.configuration.ospf_weights[link.link_id] / 65535.0 for link in incident)),
            mean(attr.bandwidth_bps / attr.physical_bandwidth_bps for attr in attrs),
            mean(attr.capacity_bps / max(1, attr.bandwidth_bps) for attr in attrs),
            min(1.0, mean(log1p(attr.queue_packets) / log1p(1_000_000) for attr in attrs)),
            mean(attr.loss_rate for attr in attrs), min(1.0, mean(loads.get(link.link_id, 0.0) for link in incident)),
            float(bool(routes_here)), min(1.0, mean(route.local_preference / 64.0 for route in routes_here)),
            min(1.0, mean(len(route.as_path) / 255.0 for route in routes_here)), min(1.0, mean(route.med / 65535.0 for route in routes_here)),
            float(snapshot.metrics.policy_consistency),
        ])
    edge_pairs: list[tuple[int, int]] = []; edge_rows: list[list[float]] = []; edge_ids: list[str] = []
    for link in snapshot.topology.links:
        attr = snapshot.configuration.performance[link.link_id]
        for source, target in ((link.source, link.target), (link.target, link.source)):
            edge_pairs.append((node_index[source], node_index[target])); edge_ids.append(f"{link.link_id}:{source}->{target}")
            edge_rows.append([1.0, float(states.get(link.link_id, "up") == "up"), snapshot.configuration.ospf_weights[link.link_id] / 65535.0,
                attr.bandwidth_bps / attr.physical_bandwidth_bps, attr.capacity_bps / max(1, attr.bandwidth_bps), min(1.0, log1p(attr.queue_packets) / log1p(1_000_000)),
                attr.loss_rate, min(1.0, loads.get(link.link_id, 0.0)), min(1.0, attr.delay_ms / 1000.0), float(node_states.get(source, "up") == "up"), float(node_states.get(target, "up") == "up")])
    policy = torch.tensor([snapshot.metrics.policy_consistency], dtype=torch.float32)
    utilization = torch.tensor([loads.get(link_id, 0.0) for link_id in links], dtype=torch.float32)
    local = {
        "ospf": {"policy_consistency": policy, "link_utilization": utilization, "ospf_weight": torch.tensor([snapshot.configuration.ospf_weights[x] / 65535.0 for x in links])},
        "bgp": {"policy_consistency": policy, "route_features": torch.tensor([[r.local_preference / 64.0, len(r.as_path) / 255.0, r.med / 65535.0, float(r.enabled)] for r in snapshot.configuration.bgp.routes], dtype=torch.float32).reshape(len(routes), 4)},
        "performance": {"link_utilization": utilization, "link_features": torch.tensor([[snapshot.configuration.performance[x].bandwidth_bps / snapshot.configuration.performance[x].physical_bandwidth_bps, snapshot.configuration.performance[x].capacity_bps / max(1, snapshot.configuration.performance[x].bandwidth_bps), log1p(snapshot.configuration.performance[x].queue_packets) / log1p(1_000_000)] for x in links], dtype=torch.float32).reshape(len(links), 3)},
    }
    return SnapshotGraph(torch.tensor(node_rows, dtype=torch.float32), torch.tensor(edge_pairs, dtype=torch.long).t().contiguous(), torch.tensor(edge_rows, dtype=torch.float32), torch.ones(len(nodes), dtype=torch.bool), torch.ones(len(edge_rows), dtype=torch.bool), nodes, links, tuple(edge_ids), routes, policy, utilization, local, snapshot.snapshot_id, snapshot.topology.topology_id, snapshot.configuration.version)


def to_pyg_data(graph: SnapshotGraph):
    """Create a PyG Data object without losing the schema-side mappings."""
    from torch_geometric.data import Data
    return Data(x=graph.node_features, edge_index=graph.edge_index, edge_attr=graph.edge_features,
                node_mask=graph.node_mask, edge_mask=graph.edge_mask, snapshot_id=graph.snapshot_id,
                node_ids=graph.node_ids, link_ids=graph.link_ids, route_targets=graph.route_targets)


def _policy_nodes(snapshot: NetworkSnapshot) -> set[str]:
    result: set[str] = set()
    for policy in snapshot.policies:
        for value in policy.fields.values():
            if isinstance(value, str) and value.startswith("R"):
                result.add(value)
    return result


def action_masks(snapshot: NetworkSnapshot, graph: SnapshotGraph | None = None, *, mask_equivalent: bool = True) -> dict[str, torch.Tensor]:
    """Candidate masks with first column as permanent no_update.

    Rows are agent macro-actions: `[1 + entities * parameters * 64]`; all
    invalid entities/values are false, while index zero is always true.
    """
    graph = graph or snapshot_to_graph(snapshot)
    link_ok = [snapshot.configuration.link_states.get(link, "up") == "up" and all(snapshot.configuration.node_states.get(node, "up") == "up" for node in (next(x for x in snapshot.topology.links if x.link_id == link).source, next(x for x in snapshot.topology.links if x.link_id == link).target)) for link in graph.link_ids]
    route_ok = [route.router_id in graph.node_ids and route.next_hop in graph.node_ids and snapshot.configuration.node_states.get(route.router_id, "up") == "up" and snapshot.configuration.node_states.get(route.next_hop, "up") == "up" and any((r.router_id, r.prefix, r.next_hop) == (route.router_id, route.prefix, route.next_hop) and r.enabled for r in snapshot.configuration.bgp.routes) for route in graph.route_targets]
    def make(agent: str, valid: Iterable[bool]) -> torch.Tensor:
        values = [True]
        for entity, ok in enumerate(valid):
            for parameter in PARAMETERS[agent]:
                for index in range(1, ACTION_VALUES + 1):
                    legal = ok
                    target = _candidate_target(graph, agent, entity)
                    candidate_value = _value(snapshot, parameter, target, index)
                    if agent == "performance":
                        attr = snapshot.configuration.performance[graph.link_ids[entity]]
                        current=_current_value(snapshot,parameter,target)
                        # Capacity/bandwidth reductions cannot improve routing
                        # or delivery. Queue growth is useful only after drops.
                        if parameter == "capacity_bps": legal = legal and index in {40,48,56,64} and candidate_value>current
                        elif parameter == "bandwidth_bps": legal = legal and candidate_value>current and candidate_value>=attr.capacity_bps
                        else: legal = legal and snapshot.metrics.dropped_bps>0 and candidate_value>current
                    elif agent == "bgp":
                        # Attribute edits which preserve the selected next hop
                        # only incur configuration/shift cost in this model.
                        legal = legal and _bgp_selection_changes(snapshot,target,parameter,candidate_value)
                    elif agent == "ospf":
                        legal = legal and abs(candidate_value-_current_value(snapshot,parameter,target)) in {1,2,4,8}
                    if legal and mask_equivalent:
                        # Setting a parameter to its current value is an
                        # environment-equivalent no-op.  Masking it also stops
                        # a greedy policy from repeating the same set action on
                        # every static step.
                        legal = candidate_value != _current_value(snapshot, parameter, target)
                    values.append(legal)
        return torch.tensor(values, dtype=torch.bool)
    return {"ospf": make("ospf", link_ok), "bgp": make("bgp", route_ok), "performance": make("performance", link_ok)}


def candidate_to_joint_action(snapshot: NetworkSnapshot, graph: SnapshotGraph, candidates: Mapping[str, int], *, mask_equivalent: bool = True) -> JointAction:
    masks = action_masks(snapshot, graph, mask_equivalent=mask_equivalent); actions: list[AtomicAction] = []
    for agent, raw in candidates.items():
        index = int(raw)
        if index == NO_UPDATE: continue
        if agent not in PARAMETERS or index < 0 or index >= masks[agent].numel() or not bool(masks[agent][index]):
            continue
        entity_count = len(graph.route_targets) if agent == "bgp" else len(graph.link_ids)
        offset = index - 1; value_index = offset % ACTION_VALUES + 1; group = offset // ACTION_VALUES
        parameter_index = group % len(PARAMETERS[agent]); entity_index = group // len(PARAMETERS[agent]); parameter = PARAMETERS[agent][parameter_index]
        if entity_index >= entity_count: continue
        if agent == "bgp":
            target = graph.route_targets[entity_index]; target_map = {"router_id": target.router_id, "prefix": target.prefix, "next_hop": target.next_hop}
        else: target_map = {"link_id": graph.link_ids[entity_index]}
        actions.append(AtomicAction(agent, parameter, target_map, "set", _value(snapshot, parameter, target_map, value_index)))
    return JointAction(tuple(actions), requested_by="rl", snapshot_id=snapshot.snapshot_id)


def monotonic_candidate_filter(snapshot: NetworkSnapshot, graph: SnapshotGraph, candidates: Mapping[str,int], *, reward_config=None, tolerance: float = 1e-9) -> dict[str,int]:
    """Keep only agent proposals whose joint immediate reward improves."""
    from netkeeper_sim.schemas import NetworkScenario
    from netkeeper_sim.simulator import UnifiedNetworkEnvironment
    accepted={name:0 for name in PARAMETERS}; best=0.0
    scenario=NetworkScenario(f"S:shield:{snapshot.topology.topology_id}",snapshot.topology,snapshot.traffic,snapshot.policies,configuration=snapshot.configuration,max_steps=max(snapshot.step+2,2))
    for agent in PARAMETERS:
        candidate=int(candidates.get(agent,0))
        if candidate==0: continue
        trial={**accepted,agent:candidate}; kernel=UnifiedNetworkEnvironment(reward_config); kernel.reset(scenario)
        outcome=kernel.step(snapshot,candidate_to_joint_action(snapshot,graph,trial))
        score=float(outcome.rewards.total_reward)
        if not outcome.errors and score>best+tolerance: accepted[agent]=candidate; best=score
    return accepted


def masked_policy(logits: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized probabilities and log-probabilities under one mask.

    This is the single mask semantics for rollout sampling, greedy inference,
    entropy and the later COMA counterfactual baseline.  Candidate zero makes
    an empty entity set safe; a malformed mask without it is repaired to the
    safe candidate rather than producing NaN.
    """
    logits = torch.as_tensor(logits, dtype=torch.float32)
    mask = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
    if logits.shape != mask.shape: raise ValueError("logits and candidate mask must have identical shape")
    if logits.ndim == 1: logits, mask = logits.unsqueeze(0), mask.unsqueeze(0)
    safe = mask.clone()
    empty = ~safe.any(dim=-1)
    safe[empty, 0] = True
    masked = logits.masked_fill(~safe, float("-inf"))
    log_probs = torch.log_softmax(masked, dim=-1)
    return log_probs.exp(), log_probs


def sample_candidates(logits: torch.Tensor, mask: torch.Tensor, *, greedy: bool = False) -> torch.Tensor:
    probabilities, _ = masked_policy(logits, mask)
    return probabilities.argmax(dim=-1) if greedy else torch.multinomial(probabilities, 1).squeeze(-1)


def masked_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    probabilities, log_probs = masked_policy(logits, mask)
    return -(probabilities * torch.where(torch.isfinite(log_probs), log_probs, torch.zeros_like(log_probs))).sum(dim=-1)


def _value(snapshot: NetworkSnapshot, parameter: str, target: Mapping[str, str], index: int) -> int:
    if parameter in {"ospf_weight", "local_preference", "as_path_length", "med"}: return index
    attr = snapshot.configuration.performance[target["link_id"]]
    fraction = index / ACTION_VALUES
    if parameter == "bandwidth_bps": return max(1, round(attr.physical_bandwidth_bps * fraction))
    if parameter == "capacity_bps": return max(1, round(min(attr.bandwidth_bps, attr.capacity_max_bps) * fraction))
    return round(1_000_000 * fraction)


def _candidate_target(graph: SnapshotGraph, agent: str, entity: int) -> dict[str, str]:
    if agent == "bgp":
        route = graph.route_targets[entity]
        return {"router_id": route.router_id, "prefix": route.prefix, "next_hop": route.next_hop}
    return {"link_id": graph.link_ids[entity]}


def _current_value(snapshot: NetworkSnapshot, parameter: str, target: Mapping[str, str]) -> int:
    if parameter == "ospf_weight":
        return int(snapshot.configuration.ospf_weights[target["link_id"]])
    if parameter in {"bandwidth_bps", "capacity_bps", "queue_packets"}:
        return int(getattr(snapshot.configuration.performance[target["link_id"]], parameter))
    route = next(
        item for item in snapshot.configuration.bgp.routes
        if (item.router_id, item.prefix, item.next_hop)
        == (target["router_id"], target["prefix"], target["next_hop"])
    )
    if parameter == "local_preference": return int(route.local_preference)
    if parameter == "med": return int(route.med)
    return len(route.as_path)


def _bgp_selection_changes(snapshot: NetworkSnapshot, target: Mapping[str,str], parameter: str, value: int) -> bool:
    router,prefix,next_hop=(target[key] for key in ("router_id","prefix","next_hop"))
    routes=[route for route in snapshot.configuration.bgp.routes if route.enabled and route.router_id==router and route.prefix==prefix]
    if len(routes)<2 or parameter!="local_preference": return False
    costs={(entry.router_id,entry.destination):entry.cost for entry in snapshot.routing_state if entry.destination_type=="node" and entry.reachable}
    def key(route): return (-route.local_preference,len(route.as_path),route.med,costs.get((router,route.next_hop),inf),route.next_hop)
    before=min(routes,key=key).next_hop
    def winner(candidate_value):
        changed=[replace(route,local_preference=int(candidate_value)) if route.next_hop==next_hop else route for route in routes]
        return min(changed,key=key).next_hop
    outcome=winner(value)
    if outcome==before: return False
    current=next(route.local_preference for route in routes if route.next_hop==next_hop)
    canonical=min((candidate for candidate in range(1,ACTION_VALUES+1) if winner(candidate)==outcome),key=lambda candidate:(abs(candidate-current),candidate))
    return value==canonical
