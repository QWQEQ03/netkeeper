"""Minimal local CLI for the safe network operation API; no HTTP service."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from netkeeper_sim.api import API_REGISTRY, ApiRequest, execute, export_json_schema, export_response_json_schema
from netkeeper_sim.dataset.scenarios import scenario_from_record
from netkeeper_sim.simulator import UnifiedNetworkEnvironment


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NetKeeper v1 network operation API CLI")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="print the static API whitelist as JSON")
    schema = commands.add_parser("export-schema", help="write the request JSON Schema")
    schema.add_argument("--output", required=True, type=Path)
    schema.add_argument("--kind", choices=("request", "response"), default="request")
    run = commands.add_parser("execute", help="execute request.json against one dataset scenario")
    run.add_argument("--dataset-root", required=True, type=Path)
    run.add_argument("--scenario-file", default="scenarios/test.jsonl")
    selector = run.add_mutually_exclusive_group()
    selector.add_argument("--scenario-id")
    selector.add_argument("--index", type=int, default=0)
    run.add_argument("--seed", type=int)
    run.add_argument("--request", required=True, type=Path)
    run.add_argument("--response", required=True, type=Path)
    run.add_argument("--dry-run", action="store_true", help="force dry_run=true without editing request JSON")
    args = parser.parse_args(argv)
    if args.command == "list":
        print(json.dumps([{"name": item.name, "version": item.version, "category": item.category, "mutates_state": item.mutates_state, "operation_type": item.operation_type, "description": item.description} for item in API_REGISTRY.values()], sort_keys=True, ensure_ascii=False))
        return 0
    if args.command == "export-schema":
        _write_json(args.output, export_json_schema() if args.kind == "request" else export_response_json_schema()); return 0
    record = _scenario_record(args.dataset_root / args.scenario_file, args.scenario_id, args.index)
    scenario = scenario_from_record(args.dataset_root, record)
    environment = UnifiedNetworkEnvironment(); snapshot, _ = environment.reset(scenario, seed=args.seed)
    request = ApiRequest.from_dict(json.loads(args.request.read_text(encoding="utf-8")))
    if args.dry_run: request = replace(request, dry_run=True)
    response = execute(environment, snapshot, request)
    _write_json(args.response, response.to_dict())
    return 0 if response.success else 2


def _scenario_record(path: Path, scenario_id: str | None, index: int) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if scenario_id is not None:
        for row in rows:
            if row.get("scenario_id") == scenario_id: return row
        raise SystemExit(f"scenario_id not found: {scenario_id}")
    if index < 0 or index >= len(rows): raise SystemExit(f"scenario index out of range: {index}")
    return rows[index]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
