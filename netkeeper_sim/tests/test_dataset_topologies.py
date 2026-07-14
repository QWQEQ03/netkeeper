from __future__ import annotations

import hashlib
import json

import networkx as nx

from netkeeper_sim.dataset.topologies import generate_topology_dataset, scan_topology_zoo, select_topology_splits
from netkeeper_sim.schemas import Topology, load_schema_topology


def _write_graph(path, graph: nx.Graph) -> None:
    nx.write_graphml(graph, path)


def _cycle(nodes: int, *, parallel: bool = False) -> nx.Graph:
    graph: nx.Graph = nx.MultiGraph() if parallel else nx.Graph()
    for index in range(nodes):
        graph.add_node(str(nodes - index - 1), label=f"node-{nodes - index - 1}", Latitude=float(index), Longitude=float(index))
    for index in range(nodes):
        graph.add_edge(str(index), str((index + 1) % nodes), LinkSpeedRaw="1", LinkSpeed="1", LinkSpeedUnits="G")
    if parallel:
        graph.add_edge("0", "1", LinkSpeedRaw="100", LinkSpeed="100", LinkSpeedUnits="M")
    return graph


def _zoo_fixture(root) -> None:
    graphml = root / "graphml"
    graphml.mkdir(parents=True)
    for index in range(16):
        _write_graph(graphml / f"small-{index:02}.graphml", _cycle(8 + index % 8, parallel=index == 0))
    for index in range(3):
        _write_graph(graphml / f"large-{index:02}.graphml", _cycle(55 + index))
    disconnected = nx.Graph()
    disconnected.add_nodes_from([str(index) for index in range(8)])
    disconnected.add_edges_from((str(index), str(index + 1)) for index in range(3))
    _write_graph(graphml / "reject-disconnected.graphml", disconnected)
    self_loop = _cycle(8)
    self_loop.add_edge("0", "0")
    _write_graph(graphml / "reject-self-loop.graphml", self_loop)


def test_schema_loader_uses_stable_names_units_defaults_and_parallel_links(tmp_path):
    path = tmp_path / "parallel.graphml"
    graph = nx.MultiGraph()
    graph.add_node("10", label="ten")
    graph.add_node("2", label="two")
    graph.add_edge("10", "2", key="slow", LinkSpeedRaw="100", LinkSpeed="100", LinkSpeedUnits="M")
    graph.add_edge("10", "2", key="fast", LinkSpeedRaw="1000000000", LinkSpeed="1", LinkSpeedUnits="G")
    _write_graph(path, graph)
    first, second = load_schema_topology(path), load_schema_topology(path)
    assert [node.node_id for node in first.nodes] == ["R0", "R1"]
    assert [node.original_label for node in first.nodes] == ["two", "ten"]
    assert first == second and len(first.links) == 2 and len({link.link_id for link in first.links}) == 2
    assert {link.attributes.capacity_bps for link in first.links} == {100_000_000, 1_000_000_000}
    assert all(link.attributes.delay_ms == 1.0 for link in first.links)


def test_schema_loader_derives_delay_only_when_coordinates_are_valid(tmp_path):
    path = tmp_path / "geo.graphml"
    graph = nx.Graph()
    graph.add_node("a", Latitude=0.0, Longitude=0.0)
    graph.add_node("b", Latitude=0.0, Longitude=1.0)
    graph.add_edge("a", "b", LinkSpeedRaw="1", LinkSpeed="1", LinkSpeedUnits="G")
    _write_graph(path, graph)
    assert load_schema_topology(path).links[0].attributes.delay_ms > 0.5


def test_scan_rejects_disconnected_and_self_loops_without_dropping_parallel_edges(tmp_path):
    _zoo_fixture(tmp_path)
    candidates = scan_topology_zoo(tmp_path)
    by_file = {item.source_file: item for item in candidates}
    assert "not_connected_under_undirected_simulation_semantics" in by_file["graphml/reject-disconnected.graphml"].reasons
    assert "self_loop_unsupported_by_schema" in by_file["graphml/reject-self-loop.graphml"].reasons
    parallel = by_file["graphml/small-00.graphml"]
    assert parallel.accepted and parallel.is_multigraph and parallel.edge_count == 9
    assert parallel.topology is not None and len(parallel.topology.links) == 9


def test_generation_is_split_isolated_stable_and_schema_round_trips(tmp_path):
    source = tmp_path / "zoo"
    _zoo_fixture(source)
    first = generate_topology_dataset(source, tmp_path / "first", selection_seed=99)
    second = generate_topology_dataset(source, tmp_path / "second", selection_seed=99)
    assert [len(first["splits"][name]) for name in ("train", "validation", "test")] == [12, 3, 4]
    ids = [item["topology_id"] for group in first["splits"].values() for item in group]
    assert len(ids) == len(set(ids)) == 19
    first_split = (tmp_path / "first" / "metadata" / "topology_split.json").read_bytes()
    second_split = (tmp_path / "second" / "metadata" / "topology_split.json").read_bytes()
    assert first_split == second_split
    assert hashlib.sha256(first_split).hexdigest() == hashlib.sha256(second_split).hexdigest()
    for item in first["splits"]["test"]:
        value = json.loads((tmp_path / "first" / item["normalized_file"]).read_text(encoding="utf-8"))
        assert Topology.from_dict(value).to_dict() == value
    candidates = json.loads((tmp_path / "first" / "metadata" / "topology_candidates.json").read_text(encoding="utf-8"))
    assert any(not item["accepted"] for item in candidates["candidates"])


def test_selection_is_independent_of_input_candidate_order(tmp_path):
    _zoo_fixture(tmp_path)
    candidates = scan_topology_zoo(tmp_path)
    forward = select_topology_splits(candidates, selection_seed=3)
    backward = select_topology_splits(reversed(candidates), selection_seed=3)
    assert {
        split: [item.topology.topology_id for item in values if item.topology]
        for split, values in forward.items()
    } == {
        split: [item.topology.topology_id for item in values if item.topology]
        for split, values in backward.items()
    }
