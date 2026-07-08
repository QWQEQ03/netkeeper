"""Topology loading and normalization."""

from netkeeper_sim.topology.loader import load_topology
from netkeeper_sim.topology.model import Link, Node, Topology, TopologyDefaults

__all__ = ["Link", "Node", "Topology", "TopologyDefaults", "load_topology"]
