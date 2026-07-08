from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import yaml

from netkeeper_sim.topology.model import Topology, TopologyDefaults
from netkeeper_sim.topology.normalizer import normalize_topology


def load_topology(
    path: str | Path,
    defaults: TopologyDefaults | None = None,
    config_path: str | Path | None = None,
) -> Topology:
    topology_path = Path(path)
    if not topology_path.exists():
        raise FileNotFoundError(topology_path)

    resolved_defaults = defaults or _load_defaults(config_path)
    suffix = topology_path.suffix.lower()
    if suffix == ".graphml":
        raw_graph = nx.read_graphml(topology_path)
    elif suffix == ".gml":
        raw_graph = nx.read_gml(topology_path, label="id")
    else:
        raise ValueError(f"Unsupported topology format: {suffix}")
    return normalize_topology(raw_graph, resolved_defaults)


def _load_defaults(config_path: str | Path | None) -> TopologyDefaults:
    if config_path is None:
        return TopologyDefaults()
    with Path(config_path).open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle) or {}
    defaults = data.get("topology_defaults", {})
    return TopologyDefaults(**defaults)
