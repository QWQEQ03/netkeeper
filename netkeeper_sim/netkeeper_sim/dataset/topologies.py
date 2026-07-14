"""Build the reproducible Topology Zoo foundation for NetKeeper-Lite."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from netkeeper_sim.schemas import LinkAttributes, Topology, load_schema_topology
from netkeeper_sim.schemas.ids import slug, stable_hash


DATASET_VERSION = "netkeeper-lite.topologies.v1"
GENERATOR_VERSION = "1.0.0"
DEFAULT_SELECTION_SEED = 20260713
DEFAULT_LINK_ATTRIBUTES = LinkAttributes()


@dataclass(frozen=True)
class Candidate:
    source_file: str
    source_format: str
    source_sha256: str
    accepted: bool
    reasons: tuple[str, ...]
    topology: Topology | None = None
    node_count: int | None = None
    edge_count: int | None = None
    density: float | None = None
    is_multigraph: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "source_file": self.source_file,
            "source_format": self.source_format,
            "source_sha256": self.source_sha256,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "density": self.density,
            "is_multigraph": self.is_multigraph,
        }
        if self.topology is not None:
            result.update({"topology_id": self.topology.topology_id, "normalized_name": self.topology.normalized_name})
        return result


def scan_topology_zoo(root: str | Path, *, defaults: LinkAttributes = DEFAULT_LINK_ATTRIBUTES) -> tuple[Candidate, ...]:
    """Scan GraphML/GML deterministically and retain rejection evidence.

    GraphML is preferred when a Zoo topology has both formats.  GML remains
    visible in the candidate report so parse failures are reproducible.
    """
    root = Path(root)
    paths = sorted((path for path in root.rglob("*") if path.suffix.lower() in {".graphml", ".gml"}), key=lambda path: path.as_posix())
    graphml_stems = {path.stem for path in paths if path.suffix.lower() == ".graphml"}
    candidates = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        source_format = path.suffix.lower().lstrip(".")
        source_digest = _file_sha256(path)
        if source_format == "gml" and path.stem in graphml_stems:
            candidates.append(Candidate(relative, source_format, source_digest, False, ("duplicate_preferred_graphml",)))
            continue
        candidates.append(_inspect_candidate(path, relative, source_format, source_digest, defaults))
    return tuple(candidates)


def _inspect_candidate(path: Path, relative: str, source_format: str, source_digest: str, defaults: LinkAttributes) -> Candidate:
    try:
        raw = nx.read_graphml(path) if source_format == "graphml" else nx.read_gml(path, label="id")
    except Exception as exc:  # parser exception is part of the reproducible manifest
        return Candidate(relative, source_format, source_digest, False, (f"parse_error:{type(exc).__name__}",))
    reasons: list[str] = []
    nodes, edges = raw.number_of_nodes(), raw.number_of_edges()
    if raw.is_directed():
        reasons.append("directed_graph_unsupported_by_physical_link_schema")
    if not 8 <= nodes <= 80:
        reasons.append("node_count_out_of_range")
    if any(left == right for left, right in raw.edges()):
        reasons.append("self_loop_unsupported_by_schema")
    undirected = nx.Graph(raw)
    if nodes == 0 or not nx.is_connected(undirected):
        reasons.append("not_connected_under_undirected_simulation_semantics")
    # A connected tree survives no single link failure.  Require at least one
    # cycle; parallel links are counted because the schema/kernel preserves them.
    if edges < nodes:
        reasons.append("insufficient_edges_for_failure_scenarios")
    try:
        topology = load_schema_topology(path, normalized_name=path.stem, defaults=defaults)
        json.dumps(topology.to_dict(), sort_keys=True, ensure_ascii=False, allow_nan=False)
    except Exception as exc:
        reasons.append(f"schema_or_serialization_error:{type(exc).__name__}")
        topology = None
    accepted = not reasons and topology is not None
    density = _density(nodes, edges)
    return Candidate(relative, source_format, source_digest, accepted, tuple(reasons), topology, nodes, edges, density, raw.is_multigraph())


def select_topology_splits(candidates: Iterable[Candidate], *, selection_seed: int = DEFAULT_SELECTION_SEED) -> dict[str, tuple[Candidate, ...]]:
    eligible = [item for item in candidates if item.accepted and item.topology is not None]
    small = [item for item in eligible if 8 <= (item.node_count or 0) <= 50]
    large = [item for item in eligible if 51 <= (item.node_count or 0) <= 80]
    train = _choose_diverse(small, 12, selection_seed, "train")
    used = {item.topology.topology_id for item in train if item.topology}
    remaining_small = [item for item in small if item.topology and item.topology.topology_id not in used]
    validation = _choose_diverse(remaining_small, 2, selection_seed, "validation-small")
    used |= {item.topology.topology_id for item in validation if item.topology}
    validation += _choose_diverse([item for item in large if item.topology and item.topology.topology_id not in used], 1, selection_seed, "validation-large")
    used |= {item.topology.topology_id for item in validation if item.topology}
    test = _choose_diverse([item for item in remaining_small if item.topology and item.topology.topology_id not in used], 2, selection_seed, "test-small")
    used |= {item.topology.topology_id for item in test if item.topology}
    test += _choose_diverse([item for item in large if item.topology and item.topology.topology_id not in used], 2, selection_seed, "test-large")
    if tuple(map(len, (train, validation, test))) != (12, 3, 4):
        raise ValueError("insufficient eligible topologies for required 12/3/4 split")
    all_ids = [item.topology.topology_id for item in (*train, *validation, *test) if item.topology]
    if len(set(all_ids)) != 19:
        raise ValueError("topology split contains leakage")
    return {"train": tuple(train), "validation": tuple(validation), "test": tuple(test)}


def _choose_diverse(pool: list[Candidate], count: int, seed: int, label: str) -> list[Candidate]:
    groups: dict[tuple[str, str], list[Candidate]] = {}
    for candidate in pool:
        groups.setdefault((_node_band(candidate.node_count or 0), _density_band(candidate.density or 0.0)), []).append(candidate)
    for group in groups.values():
        group.sort(key=lambda item: _score(seed, label, item))
    selected: list[Candidate] = []
    # Round-robin across stable size/density strata before filling remaining
    # places by the same stable score, giving coverage without randomness.
    keys = sorted(groups, key=lambda key: _score(seed, label, "|".join(key)))
    while len(selected) < count and any(groups.values()):
        for key in keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop(0))
    return selected


def generate_topology_dataset(source_root: str | Path, output_root: str | Path, *, selection_seed: int = DEFAULT_SELECTION_SEED, defaults: LinkAttributes = DEFAULT_LINK_ATTRIBUTES) -> dict[str, Any]:
    """Write normalized schema topologies plus candidate and split manifests."""
    source_root, output_root = Path(source_root), Path(output_root)
    candidates = scan_topology_zoo(source_root, defaults=defaults)
    splits = select_topology_splits(candidates, selection_seed=selection_seed)
    split_data: dict[str, list[dict[str, Any]]] = {}
    for split, items in splits.items():
        records = []
        for candidate in items:
            assert candidate.topology is not None
            relative_path = Path("topologies") / split / f"{slug(candidate.topology.normalized_name)}-{stable_hash(candidate.topology.topology_id)}.json"
            encoded = _canonical_json(candidate.topology.to_dict())
            destination = output_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(encoded, encoding="utf-8")
            records.append({
                "topology_id": candidate.topology.topology_id,
                "source_file": candidate.source_file,
                "source_sha256": candidate.source_sha256,
                "node_count": candidate.node_count,
                "edge_count": candidate.edge_count,
                "density": candidate.density,
                "is_multigraph": candidate.is_multigraph,
                "normalized_file": relative_path.as_posix(),
                "content_sha256": _sha256_text(encoded),
            })
        split_data[split] = records
    metadata = {
        "dataset_version": DATASET_VERSION,
        "generator_version": GENERATOR_VERSION,
        "selection_seed": selection_seed,
        "source_root": source_root.as_posix(),
        "selection_policy": {"train": "12 x 8-50", "validation": "2 x 8-50 + 1 x 51-80", "test": "2 x 8-50 + 2 x 51-80"},
        "defaults": defaults.to_dict(),
        "splits": split_data,
        "selection_deviations": [],
    }
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "topology_split.json").write_text(_canonical_json(metadata), encoding="utf-8")
    candidate_data = {
        "dataset_version": DATASET_VERSION,
        "generator_version": GENERATOR_VERSION,
        "source_root": source_root.as_posix(),
        "candidates": [item.to_dict() for item in candidates],
    }
    (metadata_dir / "topology_candidates.json").write_text(_canonical_json(candidate_data), encoding="utf-8")
    return metadata


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n"


def _score(seed: int, label: str, item: Candidate | str) -> str:
    identity = item if isinstance(item, str) else item.topology.topology_id if item.topology else item.source_file
    return hashlib.sha256(f"{seed}:{label}:{identity}".encode("utf-8")).hexdigest()


def _node_band(nodes: int) -> str:
    return "08-20" if nodes <= 20 else "21-35" if nodes <= 35 else "36-50" if nodes <= 50 else "51-80"


def _density(nodes: int, edges: int) -> float:
    return 0.0 if nodes < 2 else 2.0 * edges / (nodes * (nodes - 1))


def _density_band(density: float) -> str:
    return "sparse" if density < 0.10 else "medium" if density < 0.20 else "dense"
