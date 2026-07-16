"""Execute a stable disjoint shard of an already-frozen evaluation manifest.

This is an execution-only helper: it reads the existing run manifest and never
changes evaluator configuration, task identity, run keys, or result schema.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from netkeeper_sim.evaluation.batch import execute
from netkeeper_sim.evaluation.results import ResultStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--shard-count", required=True, type=int)
    args = parser.parse_args()

    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")

    root = Path(args.run_root)
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    tasks = [
        task
        for index, task in enumerate(manifest["tasks"])
        if index % args.shard_count == args.shard_index
    ]
    shard_manifest = dict(manifest)
    shard_manifest["tasks"] = tasks
    result = execute(
        args.dataset_root,
        manifest["resolved_config"],
        ResultStore(root),
        shard_manifest,
    )
    print(json.dumps({"shard_index": args.shard_index, "shard_count": args.shard_count, **result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
