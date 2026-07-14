"""Explicit compatibility adapters for the pre-schema simulation kernel.

They deliberately copy data.  A legacy ``Topology`` remains mutable, while
schema objects remain immutable and are never modified by the adapter.
"""

from __future__ import annotations

import networkx as nx

from netkeeper_sim.schemas.models import NetworkConfiguration, Topology
from netkeeper_sim.topology.model import Link as LegacyLink
from netkeeper_sim.topology.model import Node as LegacyNode
from netkeeper_sim.topology.model import Topology as LegacyTopology


def legacy_topology_from_schema(
    topology: Topology,
    configuration: NetworkConfiguration,
) -> LegacyTopology:
    """Create an isolated legacy topology view for existing routing APIs."""
    if topology.topology_id != configuration.topology_id:
        raise ValueError("configuration does not belong to topology")
    graph = nx.MultiGraph()
    nodes: dict[str, LegacyNode] = {}
    links: dict[str, LegacyLink] = {}
    for node in topology.nodes:
        legacy = LegacyNode(
            node_id=node.node_id,
            name=node.original_label or node.node_id,
            node_type=node.node_type if node.node_type != "other" else "router",
            raw_attributes=dict(node.raw_attributes),
        )
        nodes[node.node_id] = legacy
        graph.add_node(node.node_id, **legacy.raw_attributes)
    for link in topology.links:
        attrs = configuration.performance.get(link.link_id, link.attributes)
        active = (
            configuration.link_states.get(link.link_id, attrs.state) == "up"
            and configuration.node_states.get(link.source, "up") == "up"
            and configuration.node_states.get(link.target, "up") == "up"
        )
        legacy = LegacyLink(
            link_id=link.link_id,
            source=link.source,
            target=link.target,
            ospf_weight=float(configuration.ospf_weights.get(link.link_id, attrs.ospf_weight)),
            bandwidth=float(attrs.bandwidth_bps),
            capacity=float(attrs.capacity_bps),
            queue_length=attrs.queue_packets,
            loss_rate=attrs.loss_rate,
            propagation_delay=attrs.delay_ms,
            is_active=active,
            raw_attributes=dict(link.raw_attributes),
        )
        links[legacy.link_id] = legacy
        graph.add_edge(legacy.source, legacy.target, key=legacy.link_id, **legacy.__dict__)
    return LegacyTopology(graph=graph, nodes=nodes, links=links, metadata={"schema_topology_id": topology.topology_id})
