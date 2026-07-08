from __future__ import annotations

from typing import Any

import networkx as nx

from netkeeper_sim.topology.model import Link, Node, Topology, TopologyDefaults


def normalize_topology(raw_graph: nx.Graph, defaults: TopologyDefaults | None = None) -> Topology:
    defaults = defaults or TopologyDefaults()
    graph = nx.MultiGraph()
    nodes: dict[str, Node] = {}
    links: dict[str, Link] = {}

    for raw_node_id, attrs in raw_graph.nodes(data=True):
        node = _normalize_node(raw_node_id, attrs)
        nodes[node.node_id] = node
        graph.add_node(node.node_id, **_node_attrs(node))

    edge_iter = _iter_edges(raw_graph)
    for index, raw_u, raw_v, raw_key, attrs in edge_iter:
        source = str(raw_u)
        target = str(raw_v)
        link = _normalize_link(index, source, target, raw_key, attrs, defaults)
        links[link.link_id] = link
        graph.add_edge(source, target, key=link.link_id, **_link_attrs(link))

    metadata = dict(raw_graph.graph)
    return Topology(graph=graph, nodes=nodes, links=links, metadata=metadata)


def _normalize_node(raw_node_id: Any, attrs: dict[str, Any]) -> Node:
    node_id = str(raw_node_id)
    return Node(
        node_id=node_id,
        name=str(attrs.get("label") or attrs.get("name") or node_id),
        latitude=_optional_float(attrs.get("Latitude") or attrs.get("latitude")),
        longitude=_optional_float(attrs.get("Longitude") or attrs.get("longitude")),
        node_type=str(attrs.get("type") or attrs.get("node_type") or "router"),
        as_number=_optional_int(attrs.get("ASNumber") or attrs.get("as_number")),
        raw_attributes=dict(attrs),
    )


def _normalize_link(
    index: int,
    source: str,
    target: str,
    raw_key: Any,
    attrs: dict[str, Any],
    defaults: TopologyDefaults,
) -> Link:
    bandwidth = defaults.bandwidth
    capacity = defaults.capacity
    link_speed_raw = _optional_float(attrs.get("LinkSpeedRaw"))
    if defaults.use_link_speed_as_capacity and link_speed_raw is not None:
        bandwidth = link_speed_raw
        capacity = link_speed_raw

    raw_id = attrs.get("id") or attrs.get("key") or raw_key
    link_id = f"{source}-{target}-{raw_id}-{index}"
    return Link(
        link_id=link_id,
        source=source,
        target=target,
        ospf_weight=_float_or_default(attrs.get("ospf_weight"), defaults.ospf_weight),
        bandwidth=bandwidth,
        capacity=capacity,
        queue_length=int(_float_or_default(attrs.get("queue_length"), defaults.queue_length)),
        loss_rate=_float_or_default(attrs.get("loss_rate"), defaults.loss_rate),
        propagation_delay=_float_or_default(
            attrs.get("propagation_delay"), defaults.propagation_delay
        ),
        raw_attributes=dict(attrs),
    )


def _iter_edges(raw_graph: nx.Graph):
    if raw_graph.is_multigraph():
        for index, (u, v, key, attrs) in enumerate(raw_graph.edges(keys=True, data=True)):
            yield index, u, v, key, attrs
    else:
        for index, (u, v, attrs) in enumerate(raw_graph.edges(data=True)):
            yield index, u, v, index, attrs


def _node_attrs(node: Node) -> dict[str, Any]:
    return {
        "name": node.name,
        "latitude": node.latitude,
        "longitude": node.longitude,
        "node_type": node.node_type,
        "as_number": node.as_number,
        "raw_attributes": node.raw_attributes,
    }


def _link_attrs(link: Link) -> dict[str, Any]:
    return {
        "link_id": link.link_id,
        "ospf_weight": link.ospf_weight,
        "bandwidth": link.bandwidth,
        "capacity": link.capacity,
        "queue_length": link.queue_length,
        "loss_rate": link.loss_rate,
        "propagation_delay": link.propagation_delay,
        "is_active": link.is_active,
        "raw_attributes": link.raw_attributes,
    }


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_default(value: Any, default: float) -> float:
    parsed = _optional_float(value)
    return float(default if parsed is None else parsed)
