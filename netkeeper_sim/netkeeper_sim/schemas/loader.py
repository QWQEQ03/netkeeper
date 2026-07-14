"""Canonical Topology Zoo loader used by the versioned schema boundary."""

from __future__ import annotations

from dataclasses import replace
from math import asin, cos, isfinite, radians, sin, sqrt
from pathlib import Path
from typing import Any

import networkx as nx

from netkeeper_sim.schemas.ids import canonical_text, link_id, natural_key, stable_hash, topology_id
from netkeeper_sim.schemas.models import Link, LinkAttributes, Node, Topology


DEFAULT_LINK_ATTRIBUTES = LinkAttributes()
_SPEED_FACTORS = {
    "bps": 1, "b": 1,
    "kbps": 1_000, "kbit/s": 1_000, "k": 1_000,
    "mbps": 1_000_000, "mbit/s": 1_000_000, "m": 1_000_000,
    "gbps": 1_000_000_000, "gbit/s": 1_000_000_000, "g": 1_000_000_000,
    "tbps": 1_000_000_000_000, "tbit/s": 1_000_000_000_000, "t": 1_000_000_000_000,
}


def load_schema_topology(path: str | Path, *, normalized_name: str | None = None, defaults: LinkAttributes | None = None) -> Topology:
    topology_path = Path(path)
    suffix = topology_path.suffix.lower()
    if suffix == ".graphml": raw = nx.read_graphml(topology_path)
    elif suffix == ".gml": raw = nx.read_gml(topology_path, label="id")
    else: raise ValueError(f"unsupported topology format: {suffix}")
    return normalize_schema_topology(raw, normalized_name or topology_path.stem, "graphml" if suffix == ".graphml" else "gml", defaults or DEFAULT_LINK_ATTRIBUTES)


def normalize_schema_topology(raw: nx.Graph, normalized_name: str, source_format: str, defaults: LinkAttributes = DEFAULT_LINK_ATTRIBUTES) -> Topology:
    raw_nodes = sorted(((canonical_text(node_id), dict(attrs)) for node_id, attrs in raw.nodes(data=True)), key=lambda item: natural_key(item[0]))
    nodes = tuple(Node(node_id=f"R{index}", original_id=raw_id, original_label=_label(attrs, raw_id), raw_attributes=attrs) for index, (raw_id, attrs) in enumerate(raw_nodes))
    map_id = {raw_id: node.node_id for (raw_id, _attrs), node in zip(raw_nodes, nodes)}
    raw_node_attrs = {raw_id: attrs for raw_id, attrs in raw_nodes}
    edge_rows: list[tuple[str, str, str, dict[str, Any]]] = []
    if raw.is_multigraph(): iterator = raw.edges(keys=True, data=True)
    else: iterator = ((u, v, "", attrs) for u, v, attrs in raw.edges(data=True))
    for u, v, key, attrs in iterator:
        source, target = sorted((map_id[canonical_text(u)], map_id[canonical_text(v)]), key=natural_key)
        edge_rows.append((source, target, canonical_text(key), dict(attrs), raw_node_attrs[canonical_text(u)], raw_node_attrs[canonical_text(v)]))
    edge_rows.sort(key=lambda row: (natural_key(row[0]), natural_key(row[1]), row[2], stable_hash(row[3])))
    ordinals: dict[tuple[str, str], int] = {}
    links: list[Link] = []
    for source, target, raw_key, attrs, source_attrs, target_attrs in edge_rows:
        pair = (source, target); ordinal = ordinals.get(pair, 0); ordinals[pair] = ordinal + 1
        attributes = _attributes(attrs, defaults)
        if not _has_explicit_delay(attrs):
            derived_delay = _geographic_delay_ms(source_attrs, target_attrs)
            if derived_delay is not None:
                attributes = replace(attributes, delay_ms=derived_delay)
        links.append(Link(link_id(source, target, ordinal), source, target, ordinal, attributes, raw_edge_id=raw_key or None, raw_name=attrs.get("LinkLabel"), raw_attributes=attrs))
    manifest = {"name": canonical_text(normalized_name), "nodes": [(node.original_id, node.original_label) for node in nodes], "links": [(link.source, link.target, link.parallel_ordinal, link.raw_edge_id, stable_hash(link.raw_attributes)) for link in links]}
    fingerprint = stable_hash(manifest, length=64)
    return Topology(topology_id=topology_id(normalized_name, manifest), normalized_name=canonical_text(normalized_name), source_format=source_format, source_sha256=fingerprint, nodes=nodes, links=tuple(links), raw_name=raw.graph.get("Network"), defaults=defaults.to_dict())


def _label(attrs: dict[str, Any], fallback: str) -> str:
    return canonical_text(attrs.get("label") or attrs.get("name") or fallback)


def _attributes(attrs: dict[str, Any], defaults: LinkAttributes) -> LinkAttributes:
    speed = _speed_bps(attrs)
    delay = _delay_ms(attrs, defaults)
    if speed is None:
        return LinkAttributes(**{**defaults.to_dict(), "delay_ms": delay, "value_source": "default_missing_unit"})
    return LinkAttributes(physical_bandwidth_bps=speed, bandwidth_bps=speed, capacity_max_bps=speed, capacity_bps=speed, delay_ms=delay, queue_packets=defaults.queue_packets, loss_rate=defaults.loss_rate, ospf_weight=defaults.ospf_weight, state=defaults.state, value_source="topology_zoo_link_speed")


def _speed_bps(attrs: dict[str, Any]) -> int | None:
    raw = attrs.get("LinkSpeedRaw")
    unit = canonical_text(attrs.get("LinkSpeedUnits") or "").lower()
    factor = _SPEED_FACTORS.get(unit)
    display = attrs.get("LinkSpeed")
    if factor is None:
        return None
    try:
        # Zoo GraphML commonly stores LinkSpeedRaw in bps while LinkSpeed is
        # the human-readable value (e.g. 1 + G).  Small synthetic fixtures
        # often store the display value in LinkSpeedRaw instead.
        raw_value = float(raw)
        display_value = float(display) if display not in (None, "") else None
        display_bps = display_value * factor if display_value is not None else None
        if display_bps is not None and _close(raw_value, display_bps):
            value = raw_value
        elif display_bps is not None:
            value = display_bps
        else:
            value = raw_value * factor
        return int(value) if isfinite(value) and value > 0 else None
    except (TypeError, ValueError):
        return None


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= max(1.0, abs(right) * 1e-6)


def _delay_ms(attrs: dict[str, Any], defaults: LinkAttributes | None = None) -> float:
    """Use an explicit edge delay when present; otherwise retain defaults.

    Geographic delay needs both endpoint coordinates and is therefore filled
    by ``normalize_schema_topology`` after edge endpoints are known.
    """
    fallback = (defaults or DEFAULT_LINK_ATTRIBUTES).delay_ms
    for key in ("delay_ms", "Delay", "delay"):
        value = attrs.get(key)
        try:
            parsed = float(value)
            if isfinite(parsed) and parsed >= 0:
                return parsed
        except (TypeError, ValueError):
            pass
    return fallback


def _has_explicit_delay(attrs: dict[str, Any]) -> bool:
    for key in ("delay_ms", "Delay", "delay"):
        try:
            if isfinite(float(attrs.get(key))):
                return True
        except (TypeError, ValueError):
            pass
    return False


def _geographic_delay_ms(source: dict[str, Any], target: dict[str, Any]) -> float | None:
    """Estimate fibre propagation delay at 200 km/ms from valid coordinates."""
    try:
        left_lat = float(source.get("Latitude", source.get("latitude")))
        left_lon = float(source.get("Longitude", source.get("longitude")))
        right_lat = float(target.get("Latitude", target.get("latitude")))
        right_lon = float(target.get("Longitude", target.get("longitude")))
    except (TypeError, ValueError):
        return None
    if not all(isfinite(value) for value in (left_lat, left_lon, right_lat, right_lon)):
        return None
    if not (-90 <= left_lat <= 90 and -90 <= right_lat <= 90 and -180 <= left_lon <= 180 and -180 <= right_lon <= 180):
        return None
    lat_delta, lon_delta = radians(right_lat - left_lat), radians(right_lon - left_lon)
    a = sin(lat_delta / 2) ** 2 + cos(radians(left_lat)) * cos(radians(right_lat)) * sin(lon_delta / 2) ** 2
    distance_km = 6_371.0 * 2 * asin(sqrt(a))
    return distance_km / 200.0
