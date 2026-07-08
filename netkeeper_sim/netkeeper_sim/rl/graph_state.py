from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Mapping

from netkeeper_sim.policies.model import ForwardPolicy, IsolationPolicy, Policy, ReachablePolicy
from netkeeper_sim.rl import _tensor as T
from netkeeper_sim.rl.action_space import ActionMasks, build_action_masks
from netkeeper_sim.rl.config import RLConfig
from netkeeper_sim.routing.bgp import BGPRoute
from netkeeper_sim.topology.model import Link, Node, Topology


@dataclass(frozen=True)
class NetworkGraphState:
    node_features: T.Tensor
    edge_index: T.Tensor
    edge_features: T.Tensor
    node_mask: T.Tensor
    edge_mask: T.Tensor
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    parameter_masks: dict[str, T.Tensor]
    policy_observation: T.Tensor
    utilization_observation: T.Tensor
    link_ids: tuple[str, ...]
    bgp_route_targets: tuple[object, ...]


def build_network_graph_state(
    topology: Topology,
    config: RLConfig,
    policy_observation: T.Tensor,
    utilization_observation: T.Tensor,
    candidate_routes: Mapping[str, Mapping[str, list[BGPRoute]]] | None = None,
    link_utilizations: Mapping[str, float] | None = None,
    action_masks: ActionMasks | None = None,
) -> NetworkGraphState:
    node_ids = tuple(sorted(topology.nodes))
    link_ids = tuple(sorted(topology.links))
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    masks = action_masks or build_action_masks(topology, candidate_routes)
    utilizations = dict(link_utilizations or {})

    node_features = [
        _node_features(
            topology,
            topology.nodes[node_id],
            candidate_routes,
            config,
            _policy_endpoint_ids(tuple(topology.nodes), ()),
        )
        for node_id in node_ids
    ]

    edge_pairs: list[tuple[int, int]] = []
    edge_features: list[list[float]] = []
    edge_ids: list[str] = []
    for link_id in link_ids:
        link = topology.links[link_id]
        for source, target in ((link.source, link.target), (link.target, link.source)):
            edge_pairs.append((node_index[source], node_index[target]))
            edge_features.append(_edge_features(link, utilizations.get(link_id, 0.0), config))
            edge_ids.append(f"{link_id}:{source}->{target}")

    edge_index = [
        [source for source, _target in edge_pairs],
        [target for _source, target in edge_pairs],
    ]
    return NetworkGraphState(
        node_features=T.tensor(node_features, "float32"),
        edge_index=T.tensor(edge_index, "int64"),
        edge_features=T.tensor(edge_features, "float32"),
        node_mask=T.ones((len(node_ids),), "bool"),
        edge_mask=T.ones((len(edge_ids),), "bool"),
        node_ids=node_ids,
        edge_ids=tuple(edge_ids),
        parameter_masks=masks.as_dict(),
        policy_observation=policy_observation,
        utilization_observation=utilization_observation,
        link_ids=link_ids,
        bgp_route_targets=masks.bgp_route_targets,
    )


def build_policy_endpoint_ids(policies: tuple[Policy, ...]) -> set[str]:
    endpoints: set[str] = set()
    for policy in policies:
        if isinstance(policy, ForwardPolicy):
            endpoints.update((policy.source, policy.destination, policy.required_next_hop))
        elif isinstance(policy, ReachablePolicy):
            endpoints.update((policy.source, policy.destination, policy.must_pass))
        elif isinstance(policy, IsolationPolicy):
            endpoints.update(
                (
                    policy.first_source,
                    policy.first_destination,
                    policy.second_source,
                    policy.second_destination,
                )
            )
            if policy.forbidden_node is not None:
                endpoints.add(policy.forbidden_node)
    return endpoints


def build_network_graph_state_with_policies(
    topology: Topology,
    config: RLConfig,
    policies: tuple[Policy, ...],
    policy_observation: T.Tensor,
    utilization_observation: T.Tensor,
    candidate_routes: Mapping[str, Mapping[str, list[BGPRoute]]] | None = None,
    link_utilizations: Mapping[str, float] | None = None,
    action_masks: ActionMasks | None = None,
) -> NetworkGraphState:
    node_ids = tuple(sorted(topology.nodes))
    link_ids = tuple(sorted(topology.links))
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    masks = action_masks or build_action_masks(topology, candidate_routes)
    utilizations = dict(link_utilizations or {})
    policy_endpoints = build_policy_endpoint_ids(policies)

    node_features = [
        _node_features(
            topology,
            topology.nodes[node_id],
            candidate_routes,
            config,
            policy_endpoints,
        )
        for node_id in node_ids
    ]

    edge_pairs: list[tuple[int, int]] = []
    edge_features: list[list[float]] = []
    edge_ids: list[str] = []
    for link_id in link_ids:
        link = topology.links[link_id]
        for source, target in ((link.source, link.target), (link.target, link.source)):
            edge_pairs.append((node_index[source], node_index[target]))
            edge_features.append(_edge_features(link, utilizations.get(link_id, 0.0), config))
            edge_ids.append(f"{link_id}:{source}->{target}")

    edge_index = [
        [source for source, _target in edge_pairs],
        [target for _source, target in edge_pairs],
    ]
    return NetworkGraphState(
        node_features=T.tensor(node_features, "float32"),
        edge_index=T.tensor(edge_index, "int64"),
        edge_features=T.tensor(edge_features, "float32"),
        node_mask=T.ones((len(node_ids),), "bool"),
        edge_mask=T.ones((len(edge_ids),), "bool"),
        node_ids=node_ids,
        edge_ids=tuple(edge_ids),
        parameter_masks=masks.as_dict(),
        policy_observation=policy_observation,
        utilization_observation=utilization_observation,
        link_ids=link_ids,
        bgp_route_targets=masks.bgp_route_targets,
    )


def _node_features(
    topology: Topology,
    node: Node,
    candidate_routes: Mapping[str, Mapping[str, list[BGPRoute]]] | None,
    config: RLConfig,
    policy_endpoints: set[str],
) -> list[float]:
    adjacent_links = [
        link for link in topology.links.values() if node.node_id in (link.source, link.target)
    ]
    active_links = [link for link in adjacent_links if link.is_active]
    bgp_routes = _routes_for_router(candidate_routes, node.node_id)
    node_type = node.node_type.lower()
    type_values = {
        "router": 1.0 if node_type == "router" else 0.0,
        "prefix": 1.0 if node_type == "prefix" else 0.0,
        "external_as": 1.0 if node_type == "external_as" else 0.0,
        "policy": 1.0 if node_type == "policy" else 0.0,
    }
    type_other = 0.0 if any(type_values.values()) else 1.0
    degree_denominator = max(1, topology.node_count - 1)
    return [
        type_values["router"],
        type_values["prefix"],
        type_values["external_as"],
        type_values["policy"],
        type_other,
        _mean_normalized((link.ospf_weight for link in adjacent_links), config.ospf_weight.maximum),
        _mean_normalized((route.local_preference for route in bgp_routes), config.local_preference.maximum),
        _mean_normalized((len(route.as_path) for route in bgp_routes), config.as_path_length.maximum),
        _mean_normalized((route.med for route in bgp_routes), config.med.maximum),
        _mean_normalized((link.bandwidth for link in adjacent_links), config.bandwidth.maximum),
        _mean_normalized((link.capacity for link in adjacent_links), config.capacity.maximum),
        _mean_normalized((link.queue_length for link in adjacent_links), config.queue_length.maximum),
        _mean_normalized((link.loss_rate for link in adjacent_links), 1.0),
        0.0 if not adjacent_links else len(active_links) / len(adjacent_links),
        min(1.0, len(adjacent_links) / degree_denominator),
        1.0 if bgp_routes else 0.0,
        1.0 if node.node_id in policy_endpoints else 0.0,
    ]


def _edge_features(link: Link, utilization: float, config: RLConfig) -> list[float]:
    normalized_load = _normalized_load(utilization, config)
    return [
        1.0,
        0.0,
        0.0,
        0.0,
        _normalize(link.ospf_weight, config.ospf_weight.maximum),
        _normalize(link.bandwidth, config.bandwidth.maximum),
        _normalize(link.capacity, config.capacity.maximum),
        _normalize(link.queue_length, config.queue_length.maximum),
        _normalize(link.loss_rate, 1.0),
        1.0 if link.is_active else 0.0,
        normalized_load,
    ]


def _routes_for_router(
    candidate_routes: Mapping[str, Mapping[str, list[BGPRoute]]] | None,
    router: str,
) -> list[BGPRoute]:
    if not candidate_routes or router not in candidate_routes:
        return []
    routes: list[BGPRoute] = []
    for prefix in sorted(candidate_routes[router]):
        routes.extend(candidate_routes[router][prefix])
    return routes


def _mean_normalized(values: object, denominator: float) -> float:
    value_list = [float(value) for value in values]
    if not value_list:
        return 0.0
    return _normalize(fmean(value_list), denominator)


def _normalize(value: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, float(value) / denominator))


def _normalized_load(value: float, config: RLConfig) -> float:
    denominator = config.maximum_link_utilization_normalizer
    if denominator <= 0:
        denominator = 1.0
    return max(0.0, min(1.0, float(value) / denominator))


def _policy_endpoint_ids(_node_ids: tuple[str, ...], _policies: tuple[Policy, ...]) -> set[str]:
    return set()
