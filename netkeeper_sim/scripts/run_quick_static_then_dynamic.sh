#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON=".venv/bin/python"
DATASET_ROOT="../data/netkeeper_lite"
CONFIG="configs/evaluation_formal.yaml"
MANIFEST="configs/frozen_evaluation_manifest.json"
STATIC_OUTPUT="results/quick_static"
DYNAMIC_OUTPUT="results/quick_dynamic"
METHODS="no_update,ospf_default,checkpoint"

STATIC_IDS="$($PYTHON - "$MANIFEST" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    manifest = json.load(handle)
groups = manifest["analysis_groups"]["load"]["scenario_ids"]
names = ("Normal", "Hotspot", "Burst", "High-load")
selected = [scenario_id for name in names for scenario_id in groups[name][:10]]
if len(selected) != 40 or len(set(selected)) != 40:
    raise SystemExit("expected 40 unique static scenarios")
print(",".join(selected))
PY
)"

DYNAMIC_IDS="$($PYTHON - "$MANIFEST" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    manifest = json.load(handle)
selected = [item["sequence_id"] for item in manifest["dynamic"][:10]]
if len(selected) != 10 or len(set(selected)) != 10:
    raise SystemExit("expected 10 unique dynamic sequences")
print(",".join(selected))
PY
)"

validate_latest() {
    local output="$1"
    local mode="$2"
    local config_hash
    config_hash="$($PYTHON - "$output" "$mode" <<'PY'
import glob
import json
import os
import sys

matches = []
for path in glob.glob(os.path.join(sys.argv[1], "evaluation-*", "run_manifest.json")):
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("resolved_config", {}).get("mode") == sys.argv[2]:
        matches.append((os.path.getmtime(path), manifest["evaluator_config_hash"]))
if not matches:
    raise SystemExit(f"no {sys.argv[2]} run manifest found under {sys.argv[1]}")
print(max(matches)[1])
PY
)"
    "$PYTHON" -m netkeeper_sim.evaluation.cli validate-results \
        --output "$output" \
        --config-hash "$config_hash"
}

echo "[1/2] Quick static evaluation: 40 scenarios x 3 methods = 120 tasks"
"$PYTHON" -m netkeeper_sim.evaluation.cli run \
    --dataset-root "$DATASET_ROOT" \
    --config "$CONFIG" \
    --output "$STATIC_OUTPUT" \
    --mode static \
    --methods "$METHODS" \
    --scenario-ids "$STATIC_IDS" \
    --resume
validate_latest "$STATIC_OUTPUT" static

echo "[2/2] Quick dynamic evaluation: 10 sequences x 3 methods = 30 tasks"
"$PYTHON" -m netkeeper_sim.evaluation.cli run \
    --dataset-root "$DATASET_ROOT" \
    --config "$CONFIG" \
    --output "$DYNAMIC_OUTPUT" \
    --mode dynamic \
    --methods "$METHODS" \
    --sequence-ids "$DYNAMIC_IDS" \
    --resume
validate_latest "$DYNAMIC_OUTPUT" dynamic

echo "Quick static and dynamic evaluations completed successfully."
