"""Wait for formal static workers, repair missing keys, validate, and report."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
import torch


def run(command, *, cwd: Path):
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError({"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    return result.stdout.strip()


def active(unit: str, *, cwd: Path) -> bool:
    result = subprocess.run(["systemctl", "--user", "is-active", "--quiet", unit], cwd=cwd)
    return result.returncode == 0


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def terminal_count(root: Path):
    return sum(len(path.read_text(encoding="utf-8").splitlines()) for path in (root / "episodes.jsonl", root / "failures.jsonl") if path.is_file())


def atomic_json(path: Path, value):
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as handle:
        json.dump(value, handle, sort_keys=True, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--project-root",required=True)
    parser.add_argument("--run-root",required=True)
    parser.add_argument("--output-parent",required=True)
    parser.add_argument("--config",required=True)
    parser.add_argument("--dataset-root",required=True)
    parser.add_argument("--config-hash",required=True)
    parser.add_argument("--worker-prefix",required=True)
    parser.add_argument("--worker-count",type=int,required=True)
    args=parser.parse_args()
    project=Path(args.project_root).resolve(); root=Path(args.run_root).resolve()
    manifest=json.loads((root/"run_manifest.json").read_text(encoding="utf-8")); planned=len(manifest["tasks"])
    units=[f"{args.worker_prefix}-{index}.service" for index in range(args.worker_count)]
    while any(active(unit,cwd=project) for unit in units):
        time.sleep(30)

    python=str(project/".venv"/"bin"/"python")
    run_command=[python,"-m","netkeeper_sim.evaluation.cli","run","--dataset-root",args.dataset_root,"--output",args.output_parent,"--config",args.config,"--mode","static","--resume"]
    first_resume=run(run_command,cwd=project)
    first_validation=run([python,"-m","netkeeper_sim.evaluation.cli","validate-results","--output",args.output_parent,"--config-hash",args.config_hash],cwd=project)
    second_resume=run(run_command,cwd=project)
    second_validation=run([python,"-m","netkeeper_sim.evaluation.cli","validate-results","--output",args.output_parent,"--config-hash",args.config_hash],cwd=project)
    report=run([python,"scripts/report_static_evaluation.py","--run-root",str(root)],cwd=project)

    provenance={
        "dataset_manifest_sha256":sha((project/args.dataset_root/"metadata"/"manifest.json").resolve()),
        "evaluation_manifest_sha256":sha((project/"configs"/"frozen_evaluation_manifest.json").resolve()),
        "evaluator_config_file_sha256":sha((project/args.config).resolve()),
        "checkpoint_sha256":sha((project/"runs"/"rl-f27e74f349"/"best.pt").resolve()),
        "checkpoint_manifest_sha256":sha((project/"runs"/"rl-f27e74f349"/"checkpoint_manifest.json").resolve()),
        "software":{"python":platform.python_version(),"platform":platform.platform(),"torch":torch.__version__,"cuda_build":torch.version.cuda,"matplotlib":matplotlib.__version__},
        "hardware":{"cuda_available":torch.cuda.is_available(),"cuda_device_count":torch.cuda.device_count(),"cuda_name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"disk_free_bytes":shutil.disk_usage(project).free},
        "execution":{"worker_count":args.worker_count,"worker_partition":"stable task index modulo worker_count","resume":True},
    }
    manifest=json.loads((root/"run_manifest.json").read_text(encoding="utf-8")); manifest["runtime_provenance"]=provenance; atomic_json(root/"run_manifest.json",manifest)
    final={"planned":planned,"terminal":terminal_count(root),"first_resume":first_resume,"first_validation":json.loads(first_validation),"second_resume":second_resume,"second_validation":json.loads(second_validation),"report":json.loads(report),"runtime_provenance":provenance}
    atomic_json(root/"finalization.json",final)
    if final["terminal"]!=planned or not final["second_validation"].get("valid"):
        raise RuntimeError(final)
    print(json.dumps(final,sort_keys=True))


if __name__=="__main__":
    main()
