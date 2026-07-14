from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

from netkeeper_sim.dataset.dynamic_sequences import dynamic_scenario, generate_dynamic_sequences, validate_dynamic_sequences
from netkeeper_sim.schemas import JointAction
from netkeeper_sim.simulator import UnifiedNetworkEnvironment


def _copied_dataset(tmp_path: Path) -> Path:
    source = Path(__file__).resolve().parents[2] / "data" / "netkeeper_lite"
    target = tmp_path / "netkeeper_lite"
    shutil.copytree(source, target)
    return target


def test_dynamic_smoke_is_test_only_valid_and_reproducible(tmp_path):
    root = _copied_dataset(tmp_path)
    manifest = generate_dynamic_sequences(root, count=4, root_seed=91, output_file="dynamic_sequences/smoke/test.jsonl")
    first = (root / manifest["file"]).read_bytes()
    generate_dynamic_sequences(root, count=4, root_seed=91, output_file="dynamic_sequences/smoke/test.jsonl")
    assert (root / manifest["file"]).read_bytes() == first
    report = validate_dynamic_sequences(root, manifest)
    assert report["valid"] and report["count"] == 4
    assert set(report["coverage"]) == {"policy_add", "policy_remove", "traffic_scale", "hotspot_change", "link_failure_recovery", "node_failure_recovery"}
    records = [json.loads(line) for line in (root / manifest["file"]).read_text(encoding="utf-8").splitlines()]
    assert len({record["sequence_id"] for record in records}) == 4
    assert all(record["split"] == "test" and record["expected_valid"] for record in records)
    assert all([event["step"] for event in record["events"]] == sorted(event["step"] for event in record["events"]) for record in records)


def test_dynamic_events_execute_and_restore_with_compressed_test_schedule(tmp_path):
    root = _copied_dataset(tmp_path)
    manifest = generate_dynamic_sequences(root, count=1, root_seed=92, output_file="dynamic_sequences/smoke/test.jsonl")
    record = json.loads((root / manifest["file"]).read_text(encoding="utf-8").splitlines()[0])
    scenario = dynamic_scenario(root, record)
    # Preserve all Event payloads/types but compress only this test's wall-clock
    # schedule, so full add/scale/fault/recovery coverage is inexpensive.
    scenario = replace(scenario, events=tuple(replace(event, step=index) for index, event in enumerate(scenario.events)), max_steps=20)
    environment = UnifiedNetworkEnvironment(); snapshot, _ = environment.reset(scenario, seed=4); history = [snapshot]
    observed = []
    for _ in range(8):
        result = environment.step(snapshot, JointAction((), snapshot_id=snapshot.snapshot_id))
        snapshot = result.next_snapshot; history.append(snapshot)
        observed.append((len(snapshot.policies), snapshot.traffic.load_multiplier, sum(state == "down" for state in snapshot.configuration.link_states.values()), sum(state == "down" for state in snapshot.configuration.node_states.values())))
    assert observed[0][0] == len(history[0].policies) + 1
    assert observed[1][1] == history[0].traffic.load_multiplier * 1.5
    assert observed[2][2] == 1 and observed[3][2] == 0
    assert observed[4][0] == len(history[0].policies)
    assert observed[6][3] == 1 and observed[7][3] == 0
    replay = environment.step(history[0], JointAction((), snapshot_id=history[0].snapshot_id))
    assert replay.next_snapshot.metrics == history[1].metrics
