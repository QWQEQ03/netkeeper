from __future__ import annotations

import json
from pathlib import Path

from netkeeper_sim.api import ApiRequest, export_json_schema, validate_request
from netkeeper_sim.api.cli import main
from netkeeper_sim.dataset.scenarios import scenario_from_record
from netkeeper_sim.simulator import UnifiedNetworkEnvironment


ROOT = Path(__file__).resolve().parents[2] / "data" / "netkeeper_lite"


def _snapshot():
    record = json.loads((ROOT / "scenarios" / "test.jsonl").read_text(encoding="utf-8").splitlines()[0])
    env = UnifiedNetworkEnvironment(); snapshot, _ = env.reset(scenario_from_record(ROOT, record)); return snapshot


def _write(path, value): path.write_text(json.dumps(value), encoding="utf-8")


def test_cli_list_schema_and_valid_execute(tmp_path, capsys):
    assert main(["list"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert any(item["name"] == "set_ospf_weight" for item in listed)
    schema_path = tmp_path / "schema.json"
    assert main(["export-schema", "--output", str(schema_path)]) == 0
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema == export_json_schema()
    snapshot = _snapshot(); link = snapshot.topology.links[0]
    request_path, response_path = tmp_path / "request.json", tmp_path / "response.json"
    request = {"api_version": "v1", "request_id": "cli-valid", "calls": [{"api": "add_reachable_policy", "arguments": {"src": "R0", "dst": "R1"}}, {"api": "set_ospf_weight", "arguments": {"link_id": link.link_id, "weight": 64}}]}
    _write(request_path, request)
    assert main(["execute", "--dataset-root", str(ROOT), "--scenario-file", "scenarios/test.jsonl", "--index", "0", "--seed", "3", "--request", str(request_path), "--response", str(response_path)]) == 0
    response = json.loads(response_path.read_text(encoding="utf-8"))
    assert response["success"] and response["applied_calls"] == [0, 1] and response["after_snapshot_id"]
    # Re-load the exported schema and pass the same JSON through the runtime
    # validator, which performs schema plus topology-aware validation.
    assert schema["$schema"].endswith("2020-12/schema")
    assert validate_request(ApiRequest.from_dict(request), snapshot).valid


def test_cli_invalid_final_call_writes_zero_side_effect_response(tmp_path):
    request_path, response_path = tmp_path / "bad.json", tmp_path / "bad-response.json"
    _write(request_path, {"api_version": "v1", "request_id": "cli-invalid", "calls": [{"api": "add_reachable_policy", "arguments": {"src": "R0", "dst": "R1"}}, {"api": "set_ospf_weight", "arguments": {"link_id": "missing", "weight": 1}}]})
    code = main(["execute", "--dataset-root", str(ROOT), "--scenario-file", "scenarios/test.jsonl", "--index", "0", "--request", str(request_path), "--response", str(response_path)])
    response = json.loads(response_path.read_text(encoding="utf-8"))
    assert code == 2 and not response["success"] and response["applied_calls"] == [] and response["after_snapshot_id"] is None
