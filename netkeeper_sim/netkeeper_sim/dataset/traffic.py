"""Reproducible traffic matrices and initial schema configurations."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import networkx as nx

from netkeeper_sim.schemas import BGPConfiguration, BGPRoute, NetworkConfiguration, Topology, TrafficDemand, TrafficMatrix
from netkeeper_sim.schemas.ids import slug
from netkeeper_sim.simulator.deterministic import simulate_deterministic


TRAFFIC_GENERATOR_VERSION = "2.0.0"
TRAFFIC_DATASET_VERSION = "netkeeper-lite.traffic.v2"
DEFAULT_TRAFFIC_SEED = 20260713
LOAD_LEVELS: Mapping[str, float] = {"Low": 0.5, "Normal": 1.0, "High": 3.0}
PATTERNS = ("gravity", "diurnal", "hotspot", "burst")


@dataclass(frozen=True)
class TrafficGenerationConfig:
    normal_target_mlu: float = 0.25
    diurnal_period_steps: int = 24
    diurnal_amplitude: float = 0.35
    hotspot_multiplier: float = 4.0
    burst_multiplier: float = 8.0
    initial_capacity_fraction: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "normal_target_mlu": self.normal_target_mlu,
            "diurnal_period_steps": self.diurnal_period_steps,
            "diurnal_amplitude": self.diurnal_amplitude,
            "hotspot_multiplier": self.hotspot_multiplier,
            "burst_multiplier": self.burst_multiplier,
            "initial_capacity_fraction": self.initial_capacity_fraction,
            "load_levels": dict(LOAD_LEVELS),
            "unit": "bps",
        }


def derive_seed(root_seed: int, namespace: Literal["topology", "scenario", "traffic", "policy", "event"], *parts: object) -> int:
    """Return a local 64-bit seed without touching global RNG state."""
    material = ":".join((str(root_seed), namespace, *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def initial_configuration(topology: Topology, *, root_seed: int = DEFAULT_TRAFFIC_SEED, capacity_fraction: float = 0.5) -> tuple[NetworkConfiguration, dict[str, Any]]:
    """Build a legal deterministic configuration and a provenance record.

    Zoo supplies no BGP semantics.  We add two deterministic candidates for
    one prefix, with a policy-distinguishable alternate path.  Initial link
    capacity deliberately retains physical headroom so Performance actions
    can improve MLU instead of only reducing capacity.
    """
    if not 0 < capacity_fraction < 1: raise ValueError("capacity_fraction must be in (0,1)")
    base = NetworkConfiguration.initial(topology)
    design = synthetic_bgp_design(topology)
    router, primary, alternate = design["router_id"], design["primary_next_hop"], design["alternate_next_hop"]
    route_seed = derive_seed(root_seed, "topology", topology.topology_id, "synthetic-bgp")
    prefix = f"198.18.{route_seed % 256}.0/24"
    asn = 64512 + int(route_seed % 1024)
    routes = (BGPRoute(router, prefix, primary, 48, (asn,), 0), BGPRoute(router, prefix, alternate, 32, (asn,), 0))
    performance={link_id:replace(attributes,capacity_bps=max(1,round(attributes.capacity_max_bps*capacity_fraction))) for link_id,attributes in base.performance.items()}
    configuration = NetworkConfiguration(
        topology_id=base.topology_id,
        version=0,
        step=0,
        ospf_weights=base.ospf_weights,
        bgp=BGPConfiguration(routes),
        performance=performance,
        link_states=base.link_states,
        node_states=base.node_states,
    )
    return configuration, {"synthetic": True, "reason": "Topology Zoo has no BGP route semantics", "seed": route_seed, "design":{**design,"prefix":prefix}, "routes": [route.to_dict() for route in routes], "initial_capacity_fraction":capacity_fraction}


def synthetic_bgp_design(topology: Topology) -> dict[str, str]:
    """Choose a stable primary/alternate pair with a distinguishing waypoint."""
    graph=nx.Graph(); graph.add_nodes_from(node.node_id for node in topology.nodes)
    for link in topology.links:
        weight=float(link.attributes.ospf_weight)
        if graph.has_edge(link.source,link.target): graph[link.source][link.target]["weight"]=min(graph[link.source][link.target]["weight"],weight)
        else: graph.add_edge(link.source,link.target,weight=weight)
    for router in sorted(graph):
        paths={target:tuple(nx.all_shortest_paths(graph,router,target,weight="weight")) for target in sorted(graph) if target!=router}
        for alternate,alternate_paths in paths.items():
            waypoints=sorted({node for path in alternate_paths for node in path[1:-1]})
            for waypoint in waypoints:
                for primary,primary_paths in paths.items():
                    if primary!=alternate and primary!=waypoint and all(waypoint not in path[1:-1] for path in primary_paths):
                        return {"router_id":router,"primary_next_hop":primary,"alternate_next_hop":alternate,"alternate_waypoint":waypoint}
    raise ValueError(f"topology {topology.topology_id} has no policy-distinguishable synthetic BGP candidates")


def generate_base_matrix(
    topology: Topology,
    configuration: NetworkConfiguration,
    pattern: Literal["gravity", "diurnal", "hotspot", "burst"],
    *,
    seed: int,
    phase: int = 0,
    time_index: int = 0,
    config: TrafficGenerationConfig = TrafficGenerationConfig(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate a finite directed N×N base matrix, calibrated once in bps."""
    if pattern not in PATTERNS:
        raise ValueError(f"unsupported traffic pattern: {pattern}")
    rng = np.random.default_rng(seed)
    n = len(topology.nodes)
    weights = _node_weights(topology)
    raw = np.outer(weights, weights)
    np.fill_diagonal(raw, 0.0)
    # A bounded OD perturbation gives distinct, reproducible instances while
    # preserving gravity's node-size relationship.
    raw *= rng.uniform(0.90, 1.10, size=(n, n))
    np.fill_diagonal(raw, 0.0)
    parameters: dict[str, Any] = {"pattern": pattern, "seed": seed}
    if pattern == "diurnal":
        offsets = rng.integers(0, config.diurnal_period_steps, size=n)
        phase_value = (phase + time_index) % config.diurnal_period_steps
        source_factor = 1.0 + config.diurnal_amplitude * np.sin(2.0 * math.pi * (phase_value + offsets) / config.diurnal_period_steps)
        raw *= source_factor[:, None]
        parameters.update({"phase": phase, "time_index": time_index, "period_steps": config.diurnal_period_steps, "amplitude": config.diurnal_amplitude, "source_phase_offsets": offsets.tolist()})
    elif pattern == "hotspot":
        count = max(1, n // 10)
        hotspots = np.sort(rng.choice(n, size=count, replace=False))
        raw[hotspots, :] *= config.hotspot_multiplier
        raw[:, hotspots] *= config.hotspot_multiplier
        np.fill_diagonal(raw, 0.0)
        parameters.update({"hotspot_indices": hotspots.tolist(), "hotspot_nodes": [topology.nodes[index].node_id for index in hotspots], "multiplier": config.hotspot_multiplier})
    elif pattern == "burst":
        count = max(1, n // 8)
        pairs = _sample_od_pairs(rng, n, count)
        for source, destination in pairs:
            raw[source, destination] *= config.burst_multiplier
        parameters.update({"burst_od_indices": [list(pair) for pair in pairs], "burst_od_pairs": [[topology.nodes[source].node_id, topology.nodes[destination].node_id] for source, destination in pairs], "multiplier": config.burst_multiplier})
    calibrated, calibration = _calibrate_to_mlu(raw, topology, configuration, pattern, seed, config.normal_target_mlu)
    parameters["calibration"] = calibration
    _validate_matrix(calibrated, topology)
    return calibrated, parameters


def traffic_matrix_from_array(topology: Topology, values: np.ndarray, *, matrix_id: str, pattern: str, seed: int, load_multiplier: float) -> TrafficMatrix:
    _validate_matrix(values, topology)
    demands = tuple(
        TrafficDemand(topology.nodes[row].node_id, float(values[row, column]), destination=topology.nodes[column].node_id)
        for row in range(values.shape[0]) for column in range(values.shape[1])
        if row != column and values[row, column] > 0.0
    )
    return TrafficMatrix(matrix_id, tuple(node.node_id for node in topology.nodes), demands, generation_mode=pattern, load_multiplier=load_multiplier, seed=seed)


def generate_traffic_dataset(dataset_root: str | Path, *, root_seed: int = DEFAULT_TRAFFIC_SEED, config: TrafficGenerationConfig = TrafficGenerationConfig()) -> dict[str, Any]:
    """Generate all 19 topology × 4 pattern base matrices and configurations."""
    root = Path(dataset_root)
    split = json.loads((root / "metadata" / "topology_split.json").read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = []
    config_entries: list[dict[str, Any]] = []
    for split_name, records in split["splits"].items():
        for topology_record in records:
            topology = Topology.from_dict(json.loads((root / topology_record["normalized_file"]).read_text(encoding="utf-8")))
            configuration, configuration_meta = initial_configuration(topology, root_seed=root_seed,capacity_fraction=config.initial_capacity_fraction)
            config_file = Path("configurations") / split_name / f"{slug(topology.normalized_name)}.json"
            encoded_config = _canonical_json(configuration.to_dict())
            _write(root / config_file, encoded_config)
            config_entries.append({"topology_id": topology.topology_id, "split": split_name, "file": config_file.as_posix(), "sha256": _sha256(encoded_config.encode()), "bgp": configuration_meta})
            for pattern in PATTERNS:
                seed = derive_seed(root_seed, "traffic", topology.topology_id, pattern)
                phase = int(derive_seed(root_seed, "traffic", topology.topology_id, pattern, "phase") % config.diurnal_period_steps)
                base, parameters = generate_base_matrix(topology, configuration, pattern, seed=seed, phase=phase, config=config)
                matrix_file = Path("traffic") / split_name / f"{slug(topology.normalized_name)}-{pattern}.npy"
                destination = root / matrix_file
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.save(destination, base, allow_pickle=False)
                digest = _file_sha256(destination)
                base_record = {"topology_id": topology.topology_id, "split": split_name, "pattern": pattern, "seed": seed, "node_order": [node.node_id for node in topology.nodes], "dtype": str(base.dtype), "shape": list(base.shape), "unit": "bps", "base_matrix_file": matrix_file.as_posix(), "sha256": digest, "parameters": parameters}
                for level, multiplier in LOAD_LEVELS.items():
                    matrix_id = f"TM:{slug(topology.normalized_name)}:{pattern}:{level.lower()}"
                    traffic = traffic_matrix_from_array(topology, base, matrix_id=matrix_id, pattern=pattern, seed=seed, load_multiplier=multiplier)
                    # The .npy base matrix plus node order, seed, pattern and
                    # multiplier is the canonical representation.  Do not
                    # duplicate every demand list three times in the manifest.
                    entry = {**base_record, "matrix_id": matrix_id, "load_level": level, "load_multiplier": multiplier}
                    entries.append(entry)
    manifest = {"dataset_version": TRAFFIC_DATASET_VERSION, "generator_version": TRAFFIC_GENERATOR_VERSION, "root_seed": root_seed, "seed_namespaces": ["topology", "scenario", "traffic", "policy", "event"], "generation_config": config.to_dict(), "matrices": entries, "configurations": config_entries}
    _write(root / "metadata" / "traffic_manifest.json", _canonical_json(manifest))
    _write(root / "metadata" / "traffic_generation_config.json", _canonical_json(config.to_dict()))
    return manifest


def load_traffic_record(dataset_root: str | Path, record: Mapping[str, Any]) -> TrafficMatrix:
    root = Path(dataset_root)
    payload = (root / str(record["base_matrix_file"])).read_bytes()
    if _sha256(payload) != record["sha256"]:
        raise ValueError("traffic matrix checksum mismatch")
    values = np.load(root / str(record["base_matrix_file"]), allow_pickle=False)
    topology_path = _topology_path_for_id(root, str(record["topology_id"]))
    topology = Topology.from_dict(json.loads(topology_path.read_text(encoding="utf-8")))
    return traffic_matrix_from_array(topology, values, matrix_id=str(record["matrix_id"]), pattern=str(record["pattern"]), seed=int(record["seed"]), load_multiplier=float(record["load_multiplier"]))


def _topology_path_for_id(root: Path, topology_id: str) -> Path:
    split = json.loads((root / "metadata" / "topology_split.json").read_text(encoding="utf-8"))
    for records in split["splits"].values():
        for record in records:
            if record["topology_id"] == topology_id:
                return root / record["normalized_file"]
    raise ValueError("unknown topology_id")


def _node_weights(topology: Topology) -> np.ndarray:
    weights = np.zeros(len(topology.nodes), dtype=np.float64)
    index = {node.node_id: offset for offset, node in enumerate(topology.nodes)}
    for link in topology.links:
        capacity_mbps = link.attributes.capacity_bps / 1_000_000.0
        weights[index[link.source]] += capacity_mbps
        weights[index[link.target]] += capacity_mbps
    return np.maximum(weights, 1.0) / max(float(weights.sum()), 1.0)


def _sample_od_pairs(rng: np.random.Generator, nodes: int, count: int) -> list[tuple[int, int]]:
    choices = [(source, destination) for source in range(nodes) for destination in range(nodes) if source != destination]
    selected = rng.choice(len(choices), size=count, replace=False)
    return [choices[int(index)] for index in np.sort(selected)]


def _calibrate_to_mlu(raw: np.ndarray, topology: Topology, configuration: NetworkConfiguration, pattern: str, seed: int, target: float) -> tuple[np.ndarray, dict[str, float]]:
    provisional = traffic_matrix_from_array(topology, raw, matrix_id="TM:calibration", pattern=pattern, seed=seed, load_multiplier=1.0)
    observed = simulate_deterministic(topology, configuration, provisional).metrics.maximum_link_utilization
    if not math.isfinite(observed) or observed <= 0.0:
        raise ValueError("cannot calibrate a zero or non-finite traffic matrix")
    factor = target / observed
    return raw * factor, {"target_normal_mlu": target, "observed_raw_mlu": observed, "scale_factor": factor}


def _validate_matrix(values: np.ndarray, topology: Topology) -> None:
    count = len(topology.nodes)
    if values.shape != (count, count):
        raise ValueError("traffic matrix shape must equal topology node count")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("traffic matrix must be finite and non-negative")
    if not np.array_equal(np.diag(values), np.zeros(count, dtype=values.dtype)):
        raise ValueError("traffic matrix diagonal must be zero")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + "\n"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())
