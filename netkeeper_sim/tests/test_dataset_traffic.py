from __future__ import annotations

import json

import numpy as np
import pytest

from netkeeper_sim.dataset.traffic import (
    LOAD_LEVELS,
    TrafficGenerationConfig,
    generate_base_matrix,
    generate_traffic_dataset,
    initial_configuration,
    load_traffic_record,
    traffic_matrix_from_array,
)
from netkeeper_sim.schemas import Link, LinkAttributes, NetworkConfiguration, NetworkScenario, Node, Topology, TrafficMatrix
from netkeeper_sim.simulator import UnifiedNetworkEnvironment


def _topology() -> Topology:
    nodes = tuple(Node(f"R{index}", str(index)) for index in range(8))
    edges = [(index, (index + 1) % 8) for index in range(8)] + [(0, 4), (2, 6)]
    links = tuple(Link(f"L:R{left}--R{right}:{index}", f"R{left}", f"R{right}", 0, LinkAttributes(capacity_bps=20_000_000, bandwidth_bps=20_000_000, capacity_max_bps=20_000_000, physical_bandwidth_bps=20_000_000)) for index, (left, right) in enumerate(edges))
    return Topology("T:traffic", "traffic", "synthetic", "traffic", nodes, links)


@pytest.mark.parametrize("pattern", ("gravity", "diurnal", "hotspot", "burst"))
def test_traffic_patterns_are_directed_finite_reproducible_and_calibrated(pattern):
    topology = _topology()
    configuration, _ = initial_configuration(topology, root_seed=5)
    first, first_params = generate_base_matrix(topology, configuration, pattern, seed=17, phase=3, time_index=7)
    second, second_params = generate_base_matrix(topology, configuration, pattern, seed=17, phase=3, time_index=7)
    changed, _ = generate_base_matrix(topology, configuration, pattern, seed=18, phase=3, time_index=7)
    assert np.array_equal(first, second) and first_params == second_params
    assert not np.array_equal(first, changed)
    assert first.shape == (8, 8) and np.isfinite(first).all() and (first >= 0).all()
    assert np.array_equal(np.diag(first), np.zeros(8))
    assert first_params["calibration"]["target_normal_mlu"] == 0.25
    # OD demands deliberately remain directed; no symmetry constraint exists.
    assert not np.array_equal(first, first.T)


def test_diurnal_replays_and_hotspot_burst_modify_the_selected_ods():
    topology = _topology()
    configuration, _ = initial_configuration(topology)
    gravity, _ = generate_base_matrix(topology, configuration, "gravity", seed=31)
    diurnal, params = generate_base_matrix(topology, configuration, "diurnal", seed=31, phase=5, time_index=9)
    replay, replay_params = generate_base_matrix(topology, configuration, "diurnal", seed=31, phase=5, time_index=9)
    assert np.array_equal(diurnal, replay) and params == replay_params and params["phase"] == 5 and params["time_index"] == 9
    hotspot, hotspot_params = generate_base_matrix(topology, configuration, "hotspot", seed=31)
    hub = hotspot_params["hotspot_indices"][0]
    other = next(index for index in range(8) if index not in hotspot_params["hotspot_indices"])
    regular = next((left, right) for left in range(8) for right in range(8) if left != right and left not in hotspot_params["hotspot_indices"] and right not in hotspot_params["hotspot_indices"])
    # Calibration changes the common scale, but a hotspot OD receives a 4x
    # relative increase compared with an unaffected OD from the same gravity base.
    assert (hotspot[hub, other] / gravity[hub, other]) / (hotspot[regular] / gravity[regular]) == pytest.approx(4.0)
    burst, burst_params = generate_base_matrix(topology, configuration, "burst", seed=31)
    source, destination = burst_params["burst_od_indices"][0]
    unaffected = next((left, right) for left in range(8) for right in range(8) if left != right and [left, right] not in burst_params["burst_od_indices"])
    assert (burst[source, destination] / gravity[source, destination]) / (burst[unaffected] / gravity[unaffected]) == pytest.approx(8.0)


def test_load_multipliers_share_one_base_matrix_and_schema_round_trip():
    topology = _topology()
    configuration, _ = initial_configuration(topology)
    values, _ = generate_base_matrix(topology, configuration, "gravity", seed=3)
    matrices = {name: traffic_matrix_from_array(topology, values, matrix_id=f"TM:{name}", pattern="gravity", seed=3, load_multiplier=multiplier) for name, multiplier in LOAD_LEVELS.items()}
    totals = {name: sum(item.traffic_rate_bps for item in matrix.demands) * matrix.load_multiplier for name, matrix in matrices.items()}
    assert totals["Normal"] == pytest.approx(totals["Low"] * 2.0)
    assert totals["High"] == pytest.approx(totals["Normal"] * 3.0)
    assert TrafficMatrix.from_dict(matrices["Normal"].to_dict()) == matrices["Normal"]


def test_configuration_references_are_valid_and_environment_resets():
    topology = _topology()
    configuration, metadata = initial_configuration(topology, root_seed=4)
    link_ids, node_ids = {link.link_id for link in topology.links}, {node.node_id for node in topology.nodes}
    assert set(configuration.ospf_weights) == set(configuration.performance) == set(configuration.link_states) == link_ids
    assert set(configuration.node_states) == node_ids and metadata["synthetic"] is True
    for route in configuration.bgp.routes:
        assert route.router_id in node_ids and route.next_hop in node_ids and route.router_id != route.next_hop
    values, _ = generate_base_matrix(topology, configuration, "gravity", seed=8)
    traffic = traffic_matrix_from_array(topology, values, matrix_id="TM:reset", pattern="gravity", seed=8, load_multiplier=1.0)
    recovered = NetworkConfiguration.from_dict(configuration.to_dict())
    snapshot, _ = UnifiedNetworkEnvironment().reset(NetworkScenario("S:traffic", topology, traffic, configuration=recovered), seed=8)
    assert snapshot.topology.topology_id == topology.topology_id


def test_file_manifest_hashes_and_loaded_matrix_are_valid(tmp_path):
    topology = _topology()
    topology_file = "topologies/train/traffic.json"
    (tmp_path / "topologies" / "train").mkdir(parents=True)
    (tmp_path / topology_file).write_text(json.dumps(topology.to_dict()), encoding="utf-8")
    split = {"splits": {"train": [{"topology_id": topology.topology_id, "normalized_file": topology_file}], "validation": [], "test": []}}
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata" / "topology_split.json").write_text(json.dumps(split), encoding="utf-8")
    manifest = generate_traffic_dataset(tmp_path, root_seed=12, config=TrafficGenerationConfig(normal_target_mlu=0.2))
    assert len(manifest["matrices"]) == 12 and len(manifest["configurations"]) == 1
    record = next(item for item in manifest["matrices"] if item["pattern"] == "diurnal" and item["load_level"] == "High")
    matrix = load_traffic_record(tmp_path, record)
    assert matrix.load_multiplier == 3.0 and matrix.node_order == tuple(node.node_id for node in topology.nodes)
    assert (tmp_path / "metadata" / "traffic_manifest.json").is_file()
