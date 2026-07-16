from __future__ import annotations

import hashlib
import json

import pytest

from netkeeper_sim.dataset.scenarios import (
    DIFFICULTY_COUNTS,
    ScenarioDataset,
    ScenarioGenerationError,
    detect_policy_conflicts,
    generate_smoke_scenarios,
    largest_remainder_quota,
    _has_simple_waypoint_path,
    sample_policies,
    validate_scenarios,
)
from netkeeper_sim.dataset.traffic import generate_traffic_dataset, initial_configuration
from netkeeper_sim.schemas import AtomicAction, JointAction, Link, LinkAttributes, NetworkScenario, Node, Policy, Topology, TrafficMatrix
from netkeeper_sim.simulator import UnifiedNetworkEnvironment


def _topology(name: str) -> Topology:
    nodes = tuple(Node(f"R{index}", str(index)) for index in range(8))
    edges = [(index, (index + 1) % 8) for index in range(8)] + [(0, 4), (2, 6)]
    links = tuple(Link(f"L:R{left}--R{right}:{index}", f"R{left}", f"R{right}", 0, LinkAttributes()) for index, (left, right) in enumerate(edges))
    return Topology(f"T:{name}", name, "synthetic", name, nodes, links)


def _dataset_root(tmp_path):
    splits = {"train": [], "validation": [], "test": []}
    for split in splits:
        topology = _topology(split)
        file = f"topologies/{split}/{split}.json"
        encoded = json.dumps(topology.to_dict(), sort_keys=True, ensure_ascii=False, indent=2) + "\n"
        path = tmp_path / file
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(encoded, encoding="utf-8")
        splits[split].append({"topology_id": topology.topology_id, "normalized_file": file, "content_sha256": hashlib.sha256(encoded.encode()).hexdigest()})
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata" / "topology_split.json").write_text(json.dumps({"splits": splits}), encoding="utf-8")
    generate_traffic_dataset(tmp_path, root_seed=9)
    return tmp_path


@pytest.mark.parametrize("difficulty,per_kind", DIFFICULTY_COUNTS.items())
def test_policy_sampler_has_exact_three_kind_counts_and_nontrivial_initial_state(difficulty, per_kind):
    topology = _topology("sample")
    configuration, _ = initial_configuration(topology)
    policies, metadata = sample_policies(topology, configuration, difficulty, seed=17)
    assert len(policies) == 3 * per_kind
    assert {kind: sum(policy.kind == kind for policy in policies) for kind in ("reachable", "forward_pass", "isolation")} == {"reachable": per_kind, "forward_pass": per_kind, "isolation": per_kind}
    assert 0 < metadata["initial_policy_consistency"] < 1
    assert len({(policy.kind, tuple(sorted(policy.fields.items()))) for policy in policies}) == len(policies)
    bgp=[policy for policy in policies if policy.kind=="forward_pass" and policy.fields.get("destination_type")=="prefix"]
    assert len(bgp)==1


def test_policy_sampler_has_bounded_structured_failure():
    topology = _topology("failure")
    configuration, _ = initial_configuration(topology)
    with pytest.raises(ScenarioGenerationError) as raised:
        sample_policies(topology, configuration, "Easy", seed=1, max_attempts=0)
    assert raised.value.to_dict()["code"] == "policy_sampling_exhausted"

def test_synthetic_bgp_candidate_can_improve_prefix_policy():
    topology=_topology("bgp-action"); configuration,metadata=initial_configuration(topology)
    policies,_=sample_policies(topology,configuration,"Easy",seed=17)
    traffic=TrafficMatrix("TM:empty",tuple(node.node_id for node in topology.nodes),())
    environment=UnifiedNetworkEnvironment(); snapshot,_=environment.reset(NetworkScenario("S:bgp-action",topology,traffic,policies,configuration=configuration))
    alternate=next(route for route in configuration.bgp.routes if route.next_hop==metadata["design"]["alternate_next_hop"])
    action=AtomicAction("bgp","local_preference",{"router_id":alternate.router_id,"prefix":alternate.prefix,"next_hop":alternate.next_hop},"set",64)
    result=environment.step(snapshot,JointAction((action,),snapshot_id=snapshot.snapshot_id))
    assert result.metrics.policy_consistency > snapshot.metrics.policy_consistency
    assert result.rewards.per_agent["bgp"] > 0


def test_explicit_policy_conflict_detector_catches_duplicates_and_pass_avoid():
    duplicate = Policy("P:one", "reachable", {"source": "R0", "destination": "R1"})
    duplicate_two = Policy("P:two", "reachable", {"source": "R0", "destination": "R1"})
    passed = Policy("P:pass", "forward_pass", {"source": "R0", "destination": "R2", "waypoint": "R1"})
    avoided = Policy("P:avoid", "forward_avoid", {"source": "R0", "destination": "R2", "forbidden_node": "R1"})
    assert set(detect_policy_conflicts((duplicate, duplicate_two, passed, avoided))) == {"P:one", "P:two", "P:pass", "P:avoid"}


def test_forward_feasibility_rejects_a_repeating_walk_through_waypoint():
    import networkx as nx
    graph = nx.Graph([( "src", "center"), ("pass", "center"), ("dst", "center")])
    assert not _has_simple_waypoint_path(graph, "src", "pass", "dst")


def test_smoke_generation_quotas_lazy_round_trip_validation_and_noop(tmp_path):
    root = _dataset_root(tmp_path)
    manifest = generate_smoke_scenarios(root, root_seed=22)
    first = {split: (root / info["file"]).read_bytes() for split, info in manifest["splits"].items()}
    repeated = generate_smoke_scenarios(root, root_seed=22)
    assert first == {split: (root / info["file"]).read_bytes() for split, info in repeated["splits"].items()}
    assert largest_remainder_quota(500, {"gravity": .30, "diurnal": .20, "hotspot": .25, "burst": .25}) == {"gravity": 150, "diurnal": 100, "hotspot": 125, "burst": 125}
    report = validate_scenarios(root, check_environment=True)
    assert report["valid"] and report["scenario_counts"] == {"train": 12, "validation": 12, "test": 12}
    for split, info in manifest["splits"].items():
        scenarios = list(ScenarioDataset(root, info["file"]))
        assert len(scenarios) == 12
        assert {scenario.topology.topology_id for scenario in scenarios} == {f"T:{split}"}
        assert {scenario.traffic.generation_mode for scenario in scenarios} == {"gravity", "diurnal", "hotspot", "burst"}
        assert {scenario.traffic.load_multiplier for scenario in scenarios} == {0.5, 1.0, 3.0}
        environment = UnifiedNetworkEnvironment(); snapshot, _ = environment.reset(scenarios[0], seed=2)
        result = environment.step(snapshot, JointAction((), snapshot_id=snapshot.snapshot_id))
        assert not result.errors and result.next_snapshot.step == 1


def test_validator_rejects_split_leakage_and_hash_tampering(tmp_path):
    root = _dataset_root(tmp_path)
    manifest = generate_smoke_scenarios(root, root_seed=3)
    path = root / manifest["splits"]["train"]["file"]
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    record["topology_id"] = "T:test"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    report = validate_scenarios(root)
    assert not report["valid"] and report["errors"][0]["code"] == "invalid_scenario"
