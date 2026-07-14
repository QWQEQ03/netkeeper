"""CLI for deterministic Topology Zoo dataset generation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from netkeeper_sim.dataset.topologies import DEFAULT_SELECTION_SEED, generate_topology_dataset, scan_topology_zoo, select_topology_splits
from netkeeper_sim.dataset.traffic import generate_traffic_dataset, load_traffic_record
from netkeeper_sim.dataset.scenarios import ScenarioDataset, generate_smoke_scenarios, generate_static_scenarios, validate_scenarios
from netkeeper_sim.dataset.dynamic_sequences import SMOKE_SEQUENCE_COUNT, dynamic_scenario, generate_dynamic_sequences, validate_dynamic_sequences
from netkeeper_sim.dataset.publication import generate_release_metadata, validate_release
from netkeeper_sim.schemas import NetworkConfiguration, NetworkScenario, Topology
from netkeeper_sim.simulator import UnifiedNetworkEnvironment


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the NetKeeper-Lite topology dataset.")
    parser.add_argument("command", choices=("dry-run", "list-candidates", "generate", "generate-release", "generate-traffic", "smoke-traffic", "generate-scenarios-smoke", "generate-scenarios", "validate-scenarios", "smoke-scenarios", "generate-dynamic-smoke", "generate-dynamic", "validate-dynamic", "smoke-dynamic", "generate-metadata", "validate-release"))
    parser.add_argument("--source-root", type=Path, default=Path("../InternetTopologyZoo"))
    parser.add_argument("--output-root", type=Path, default=Path("data/netkeeper_lite"))
    parser.add_argument("--selection-seed", type=int, default=DEFAULT_SELECTION_SEED)
    args = parser.parse_args()
    if args.command == "generate-release":
        topology = generate_topology_dataset(args.source_root, args.output_root, selection_seed=args.selection_seed)
        traffic = generate_traffic_dataset(args.output_root, root_seed=args.selection_seed)
        scenarios = generate_static_scenarios(args.output_root, root_seed=args.selection_seed)
        dynamic = generate_dynamic_sequences(args.output_root, root_seed=args.selection_seed)
        metadata = generate_release_metadata(args.output_root, root_seed=args.selection_seed)
        print(json.dumps({"topologies": {split: len(items) for split, items in topology["splits"].items()}, "traffic_records": len(traffic["matrices"]), "scenarios": {split: item["count"] for split, item in scenarios["splits"].items()}, "dynamic_sequences": dynamic["count"], "manifest_files": metadata["manifest"]["file_count"]}, ensure_ascii=False, sort_keys=True))
        return
    if args.command == "generate-metadata":
        result = generate_release_metadata(args.output_root, root_seed=args.selection_seed)
        print(json.dumps({"manifest_files": result["manifest"]["file_count"]}, ensure_ascii=False, sort_keys=True))
        return
    if args.command == "validate-release":
        print(json.dumps(validate_release(args.output_root), ensure_ascii=False, sort_keys=True))
        return
    if args.command in {"generate-dynamic-smoke", "generate-dynamic"}:
        count = SMOKE_SEQUENCE_COUNT if args.command == "generate-dynamic-smoke" else 100
        output = "dynamic_sequences/smoke/test.jsonl" if args.command == "generate-dynamic-smoke" else "dynamic_sequences/test.jsonl"
        result = generate_dynamic_sequences(args.output_root, count=count, root_seed=args.selection_seed, output_file=output)
        print(json.dumps({"count": result["count"], "file": result["file"]}, ensure_ascii=False, sort_keys=True))
        return
    if args.command == "validate-dynamic":
        print(json.dumps(validate_dynamic_sequences(args.output_root), ensure_ascii=False, sort_keys=True))
        return
    if args.command == "smoke-dynamic":
        manifest = json.loads((args.output_root / "metadata" / "dynamic_smoke_manifest.json").read_text(encoding="utf-8"))
        first = json.loads((args.output_root / manifest["file"]).read_text(encoding="utf-8").splitlines()[0])
        scenario = dynamic_scenario(args.output_root, first)
        from netkeeper_sim.schemas import JointAction
        environment = UnifiedNetworkEnvironment(); snapshot, _ = environment.reset(scenario, seed=args.selection_seed)
        history = [snapshot]
        changes = []
        while snapshot.step <= max(event.step for event in scenario.events):
            result = environment.step(snapshot, JointAction((), snapshot_id=snapshot.snapshot_id))
            changes.append({"step": snapshot.step, "policy_count": len(result.next_snapshot.policies), "traffic_multiplier": result.next_snapshot.traffic.load_multiplier, "link_down": sum(state == "down" for state in result.next_snapshot.configuration.link_states.values()), "node_down": sum(state == "down" for state in result.next_snapshot.configuration.node_states.values())})
            snapshot = result.next_snapshot; history.append(snapshot)
        replay = environment.step(history[0], JointAction((), snapshot_id=history[0].snapshot_id))
        print(json.dumps({"sequence_id": first["sequence_id"], "steps": len(changes), "final_link_down": changes[-1]["link_down"], "final_node_down": changes[-1]["node_down"], "replay_matches": replay.next_snapshot.metrics == history[1].metrics, "event_changes": [item for item in changes if item["step"] in {0, 30, 60, 90, 120, 150, 180, 210}]}, ensure_ascii=False, sort_keys=True))
        return
    if args.command == "generate-scenarios-smoke":
        result = generate_smoke_scenarios(args.output_root, root_seed=args.selection_seed)
        print(json.dumps({split: info["count"] for split, info in result["splits"].items()}, ensure_ascii=False, sort_keys=True))
        return
    if args.command == "generate-scenarios":
        result = generate_static_scenarios(args.output_root, root_seed=args.selection_seed)
        print(json.dumps({split: info["count"] for split, info in result["splits"].items()}, ensure_ascii=False, sort_keys=True))
        return
    if args.command == "validate-scenarios":
        print(json.dumps(validate_scenarios(args.output_root), ensure_ascii=False, sort_keys=True))
        return
    if args.command == "smoke-scenarios":
        config = json.loads((args.output_root / "metadata" / "scenario_generation_config.json").read_text(encoding="utf-8"))
        results = []
        from netkeeper_sim.schemas import JointAction
        for split, info in config["splits"].items():
            for scenario in ScenarioDataset(args.output_root, info["file"]):
                environment = UnifiedNetworkEnvironment()
                snapshot, _ = environment.reset(scenario, seed=args.selection_seed)
                step = environment.step(snapshot, JointAction((), snapshot_id=snapshot.snapshot_id))
                results.append({"split": split, "scenario_id": scenario.scenario_id, "next_snapshot_id": step.next_snapshot.snapshot_id})
        print(json.dumps({"reset_noop_step_count": len(results), "samples": results[:3]}, ensure_ascii=False, sort_keys=True))
        return
    if args.command == "generate-traffic":
        result = generate_traffic_dataset(args.output_root, root_seed=args.selection_seed)
        print(json.dumps({"matrices": len(result["matrices"]), "configurations": len(result["configurations"])}, ensure_ascii=False, sort_keys=True))
        return
    if args.command == "smoke-traffic":
        manifest = json.loads((args.output_root / "metadata" / "traffic_manifest.json").read_text(encoding="utf-8"))
        selected = []
        seen_topologies = set()
        for record in manifest["matrices"]:
            if record["load_level"] == "Normal" and record["topology_id"] not in seen_topologies:
                selected.append(record)
                seen_topologies.add(record["topology_id"])
            if len(selected) == 2:
                break
        reset_ids = []
        for record in selected:
            traffic = load_traffic_record(args.output_root, record)
            config_record = next(item for item in manifest["configurations"] if item["topology_id"] == record["topology_id"])
            topology_record = next(item for group in json.loads((args.output_root / "metadata" / "topology_split.json").read_text(encoding="utf-8"))["splits"].values() for item in group if item["topology_id"] == record["topology_id"])
            topology = Topology.from_dict(json.loads((args.output_root / topology_record["normalized_file"]).read_text(encoding="utf-8")))
            configuration = NetworkConfiguration.from_dict(json.loads((args.output_root / config_record["file"]).read_text(encoding="utf-8")))
            snapshot, _ = UnifiedNetworkEnvironment().reset(NetworkScenario(f"S:smoke:{record['matrix_id']}", topology, traffic, configuration=configuration), seed=args.selection_seed)
            reset_ids.append(snapshot.snapshot_id)
        print(json.dumps({"reset_count": len(reset_ids), "snapshot_ids": reset_ids}, ensure_ascii=False, sort_keys=True))
        return
    candidates = scan_topology_zoo(args.source_root)
    if args.command == "list-candidates":
        print(json.dumps([item.to_dict() for item in candidates], ensure_ascii=False, indent=2, sort_keys=True))
        return
    splits = select_topology_splits(candidates, selection_seed=args.selection_seed)
    if args.command == "dry-run":
        print(json.dumps({name: [item.to_dict() for item in items] for name, items in splits.items()}, ensure_ascii=False, indent=2, sort_keys=True))
        return
    result = generate_topology_dataset(args.source_root, args.output_root, selection_seed=args.selection_seed)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
