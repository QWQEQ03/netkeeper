from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from typing import Any, Iterable

import networkx as nx


@dataclass(frozen=True)
class TopologyDefaults:
    ospf_weight: float = 1.0
    bandwidth: float = 100.0
    capacity: float = 100.0
    queue_length: int = 100
    loss_rate: float = 0.0
    propagation_delay: float = 1.0
    use_link_speed_as_capacity: bool = False


@dataclass(frozen=True)
class Node:
    node_id: str
    name: str
    latitude: float | None = None
    longitude: float | None = None
    node_type: str = "router"
    as_number: int | None = None
    raw_attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Link:
    link_id: str
    source: str
    target: str
    ospf_weight: float
    bandwidth: float
    capacity: float
    queue_length: int
    loss_rate: float
    propagation_delay: float
    is_active: bool = True
    raw_attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def undirected_key(self) -> tuple[str, str]:
        return tuple(sorted((self.source, self.target)))


@dataclass
class Topology:
    graph: nx.MultiGraph
    nodes: dict[str, Node]
    links: dict[str, Link]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.links)

    def get_links_between(self, u: str, v: str) -> list[Link]:
        key = tuple(sorted((str(u), str(v))))
        return [link for link in self.links.values() if link.undirected_key == key]

    def active_routing_link_ids_between(self, u: str, v: str) -> list[str]:
        links = [
            link
            for link in self.get_links_between(u, v)
            if link.is_active
        ]
        if not links:
            return []
        min_weight = min(link.ospf_weight for link in links)
        return [
            link.link_id
            for link in links
            if link.ospf_weight == min_weight
        ]

    def fail_link(self, u: str, v: str) -> None:
        links = self.get_links_between(u, v)
        if not links:
            raise ValueError(f"No link exists between {u!r} and {v!r}")
        for link in links:
            link.is_active = False
            self.graph[link.source][link.target][link.link_id]["is_active"] = False

    def restore_link(self, u: str, v: str) -> None:
        links = self.get_links_between(u, v)
        if not links:
            raise ValueError(f"No link exists between {u!r} and {v!r}")
        for link in links:
            link.is_active = True
            self.graph[link.source][link.target][link.link_id]["is_active"] = True

    def update_ospf_weight(self, u: str, v: str, weight: float) -> None:
        if weight <= 0:
            raise ValueError("OSPF weight must be positive")
        links = self.get_links_between(u, v)
        if not links:
            raise ValueError(f"No link exists between {u!r} and {v!r}")
        for link in links:
            link.ospf_weight = float(weight)
            self.graph[link.source][link.target][link.link_id]["ospf_weight"] = float(weight)

    def to_routing_graph(self) -> nx.Graph:
        """Return a simple active graph, folding parallel links by minimum weight."""
        routing_graph = nx.Graph()
        for node_id, node in self.nodes.items():
            routing_graph.add_node(node_id, **node.raw_attributes)

        for link in self.links.values():
            if not link.is_active:
                continue
            existing = routing_graph.get_edge_data(link.source, link.target)
            if existing is None or link.ospf_weight < existing.get("ospf_weight", inf):
                routing_graph.add_edge(
                    link.source,
                    link.target,
                    ospf_weight=link.ospf_weight,
                    link_ids=[link.link_id],
                )
            elif link.ospf_weight == existing.get("ospf_weight"):
                existing["link_ids"].append(link.link_id)
        return routing_graph


def build_topology_from_edges(
    edges: Iterable[tuple[str, str, float]],
    defaults: TopologyDefaults | None = None,
) -> Topology:
    defaults = defaults or TopologyDefaults()
    graph = nx.MultiGraph()
    nodes: dict[str, Node] = {}
    links: dict[str, Link] = {}

    for raw_u, raw_v, raw_weight in edges:
        u = str(raw_u)
        v = str(raw_v)
        for node_id in (u, v):
            if node_id not in nodes:
                node = Node(node_id=node_id, name=node_id)
                nodes[node_id] = node
                graph.add_node(node_id, **node.raw_attributes)

        link_id = f"{u}-{v}-{len(links)}"
        link = Link(
            link_id=link_id,
            source=u,
            target=v,
            ospf_weight=float(raw_weight),
            bandwidth=defaults.bandwidth,
            capacity=defaults.capacity,
            queue_length=defaults.queue_length,
            loss_rate=defaults.loss_rate,
            propagation_delay=defaults.propagation_delay,
        )
        links[link_id] = link
        graph.add_edge(u, v, key=link_id, **link.__dict__)

    return Topology(graph=graph, nodes=nodes, links=links)
