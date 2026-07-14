import json
from pathlib import Path

from netkeeper_sim.intent.evaluation import aggregate, evaluate_dataset
from netkeeper_sim.intent.translation import DeepSeekClient, DeepSeekConfig, JsonCache, PromptBuilder, Translator

ROOT = Path(__file__).resolve().parents[2] / "data" / "netkeeper_lite"

class GoldTransport:
    def __init__(self, rows): self.rows = iter(rows)
    def __call__(self, url, headers, payload, timeout):
        row = next(self.rows)
        return {"choices": [{"message": {"content": json.dumps(row["expected_translation"], ensure_ascii=False)}}], "usage": {"total_tokens": 9}}

def _rows(split, count=2): return [json.loads(x) for x in (ROOT / "intents" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()[:count]]

def test_aggregate_known_values():
    rows = [
        {"is_valid": True, "prediction": {"status": "accepted"}, "api_name_correct": 1, "api_name_total": 1, "argument_correct": 2, "argument_total": 2, "call_matches": 1, "prediction_call_count": 1, "gold_call_count": 1, "exact_match": True, "accept_reject_correct": True, "execution_success": True, "semantic_success": True, "error_group": "none", "attempts": 1, "feedback_corrected": False, "cache_hit": False, "latency_ms": 3, "token_usage": {"total_tokens": 4}, "optimization_dispatched": False},
        {"is_valid": False, "prediction": {"status": "rejected"}, "api_name_correct": 0, "api_name_total": 0, "argument_correct": 0, "argument_total": 0, "call_matches": 0, "prediction_call_count": 0, "gold_call_count": 0, "exact_match": True, "accept_reject_correct": True, "execution_success": False, "semantic_success": None, "error_group": "none", "attempts": 2, "feedback_corrected": True, "cache_hit": True, "latency_ms": 1, "token_usage": {"total_tokens": 6}, "optimization_dispatched": False},
    ]
    result = aggregate(rows)
    assert result["api_name_accuracy"] == 1.0 and result["argument_accuracy"] == 1.0
    assert result["invalid_rejection"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert result["call_f1"] == 1.0 and result["semantic_success_rate"] == 1.0
    assert result["token_usage"]["total_tokens"] == 10 and result["cache_hit_rate"] == 0.5

def test_fake_end_to_end_all_modes_and_resume(tmp_path):
    for split in ("train", "validation", "test"):
        rows = _rows(split, 2)
        for mode in ("prompt_only", "few_shot", "full"):
            client = DeepSeekClient(DeepSeekConfig(cache_directory=str(tmp_path / split / mode)), transport=GoldTransport(rows), environ={"DEEPSEEK_API_KEY": "fake"}, sleep=lambda _: None)
            translator = Translator(client, PromptBuilder(ROOT / "intents" / "few_shot_candidates.jsonl"), JsonCache(tmp_path / split / mode))
            result = evaluate_dataset(translator, ROOT, split=split, mode=mode, output_directory=tmp_path / "out", online=True, start=0, stop=2)
            assert result["sample_count"] == 2 and result["exact_match"] == 1.0
            # Same identity/mode output resumes without a new fake response.
            assert evaluate_dataset(translator, ROOT, split=split, mode=mode, output_directory=tmp_path / "out", online=True, start=0, stop=2)["sample_count"] == 2

def test_fake_invalid_rejection_is_scored(tmp_path):
    row = json.loads((ROOT / "intents" / "test.jsonl").read_text(encoding="utf-8").splitlines()[440])
    client = DeepSeekClient(DeepSeekConfig(cache_directory=str(tmp_path / "cache")), transport=GoldTransport([row]), environ={"DEEPSEEK_API_KEY": "fake"})
    translator = Translator(client, PromptBuilder(ROOT / "intents" / "few_shot_candidates.jsonl"), JsonCache(tmp_path / "cache"))
    result = evaluate_dataset(translator, ROOT, split="test", mode="full", output_directory=tmp_path / "out", online=True, start=440, stop=441)
    assert result["accept_reject_accuracy"] == 1.0
    assert result["invalid_rejection"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
