import hashlib
import json
from pathlib import Path

from netkeeper_sim.api import API_REGISTRY
from netkeeper_sim.dataset.intents import SPLIT_COUNTS, generate_intent_dataset, validate_intent_dataset


ROOT = Path(__file__).resolve().parents[2] / "data" / "netkeeper_lite"


def _rows(directory: Path, split: str):
    return [json.loads(line) for line in (directory / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]


def test_formal_intent_dataset_contract():
    directory = ROOT / "intents"
    all_rows = {split: _rows(directory, split) for split in SPLIT_COUNTS}
    assert {split: len(rows) for split, rows in all_rows.items()} == SPLIT_COUNTS
    assert sum(not row["is_valid"] for row in all_rows["test"]) == 60
    for split, rows in all_rows.items():
        assert {str(level): sum(row["level"] == int(level) for row in rows) for level in ("1", "2", "3")} == ({"1": 480, "2": 480, "3": 240} if split == "train" else {"1": 120, "2": 120, "3": 60} if split == "validation" else {"1": 200, "2": 200, "3": 100})
        assert set(API_REGISTRY) <= {call["api"] for row in rows for call in row["expected_calls"]}
        assert sum(row["rewrite_selected"] for row in rows) == len(rows) // 5
        for row in rows:
            assert not Path(row["scenario"]["file"]).is_absolute()
            assert row["content_sha256"] == hashlib.sha256(json.dumps({k: v for k, v in row.items() if k != "content_sha256"}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert not {r["family_id"] for r in all_rows["train"]} & {r["family_id"] for r in all_rows["test"]}
    few = _rows(directory, "few_shot_candidates")
    assert few and {row["split"] for row in few} == {"train"}


def test_formal_intent_dataset_validator_and_manifest():
    result = validate_intent_dataset(ROOT)
    assert result["valid"], result["errors"][:5]
    manifest = json.loads((ROOT / "intents" / "manifest.json").read_text(encoding="utf-8"))
    assert all(not Path(item["path"]).is_absolute() for item in manifest["files"])
    for item in manifest["files"]:
        path = ROOT / "intents" / item["path"]
        assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]


def test_generation_is_deterministic_on_small_real_splits(tmp_path):
    counts = {"train": 19, "validation": 19, "test": 19}
    generate_intent_dataset(ROOT, output_directory=str(tmp_path / "a"), counts=counts, seed=99)
    generate_intent_dataset(ROOT, output_directory=str(tmp_path / "b"), counts=counts, seed=99)
    for split in counts:
        assert (tmp_path / "a" / f"{split}.jsonl").read_bytes() == (tmp_path / "b" / f"{split}.jsonl").read_bytes()
