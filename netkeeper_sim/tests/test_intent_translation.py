import json
from pathlib import Path

import pytest

from netkeeper_sim.api import ApiCall
from netkeeper_sim.dataset.intents import _read_records, _snapshot
from netkeeper_sim.dataset.scenarios import scenario_from_record
from netkeeper_sim.intent.translation import (
    DeepSeekClient, DeepSeekConfig, JsonCache, PromptBuilder, RecordingDispatcher,
    TranslationFailure, Translator, parse_response, rewrite_record,
)
from netkeeper_sim.simulator import UnifiedNetworkEnvironment

ROOT = Path(__file__).resolve().parents[2] / "data" / "netkeeper_lite"

def _snapshot_and_row():
    row = json.loads((ROOT / "intents" / "test.jsonl").read_text().splitlines()[0])
    record = next(r for r in _read_records(ROOT, "test") if r["scenario_id"] == row["scenario"]["scenario_id"])
    return row, record, _snapshot(ROOT, record)

class Fake:
    def __init__(self, responses): self.responses=list(responses); self.calls=[]
    def __call__(self, url, headers, payload, timeout):
        self.calls.append((url, headers, payload, timeout)); item=self.responses.pop(0)
        if isinstance(item, Exception): raise item
        return {"choices":[{"message":{"content":json.dumps(item, ensure_ascii=False)}}], "usage":{"total_tokens":7}}

def _translator(tmp_path, responses):
    fake=Fake(responses); cfg=DeepSeekConfig(cache_directory=str(tmp_path / "cache"), max_retries=1)
    client=DeepSeekClient(cfg, transport=fake, environ={"DEEPSEEK_API_KEY":"secret"}, sleep=lambda _:None)
    return Translator(client, PromptBuilder(ROOT / "intents" / "few_shot_candidates.jsonl"), JsonCache(tmp_path / "cache")), fake

def test_prompt_modes_and_no_label_leakage(tmp_path):
    row, record, snap=_snapshot_and_row(); builder=PromptBuilder(ROOT / "intents" / "few_shot_candidates.jsonl")
    one=builder.build(row["natural_language"], snap, mode="prompt_only", scenario_id=record["scenario_id"])
    few=builder.build(row["natural_language"], snap, mode="few_shot", scenario_id=record["scenario_id"])
    assert len(one)==2 and len(few)==3
    assert row["expected_calls"] != []
    assert json.dumps(row["expected_calls"],ensure_ascii=False) not in one[-1]["content"]
    assert "Authorization" not in json.dumps(one)

def test_accepted_rejected_and_dispatcher(tmp_path):
    row, record, snap=_snapshot_and_row()
    calls=[{"api":"add_reachable_policy","arguments":{"src":row["expected_calls"][0]["arguments"].get("src", "R0"),"dst":row["expected_calls"][0]["arguments"].get("dst", "R1")}}]
    # use the row's genuine gold request so topology entities are guaranteed
    calls=row["expected_calls"]
    t,_=_translator(tmp_path,[{"status":"accepted","calls":calls,"need_optimization":row["need_optimization"]}])
    out=t.translate(row["natural_language"],snap,request_id="x",mode="prompt_only",online=True)
    assert out.status=="accepted"
    t,_=_translator(tmp_path / "reject",[{"status":"rejected","calls":[],"error":{"code":"TRANSLATION_UNKNOWN_NODE"}}])
    assert t.translate("bad",snap,request_id="y",online=True).status=="rejected"
    opt={"status":"accepted","calls":[{"api":"optimize_network","arguments":{"objectives":["mlu"]}}],"need_optimization":False}
    t,_=_translator(tmp_path / "opt",[opt]); env=UnifiedNetworkEnvironment(); env.reset(scenario_from_record(ROOT,record),seed=record["seed"]); dispatcher=RecordingDispatcher()
    out=t.translate("opt",snap,request_id="z",mode="full",online=True,env=env,run_executor=True,dispatcher=dispatcher)
    assert out.status=="accepted" and dispatcher.requests and out.execution["optimization_status"]=="dispatched"

def test_full_once_feedback_then_stop(tmp_path):
    row, record, snap=_snapshot_and_row(); bad={"status":"accepted","calls":[{"api":"nope","arguments":{}}],"need_optimization":False}; good={"status":"accepted","calls":row["expected_calls"],"need_optimization":row["need_optimization"]}
    t,fake=_translator(tmp_path,[bad,good]); out=t.translate(row["natural_language"],snap,request_id="retry",mode="full",online=True)
    assert out.status=="accepted" and out.attempts==2 and len(fake.calls)==2
    t,_=_translator(tmp_path / "bad",[bad,bad]); out=t.translate("x",snap,request_id="stop",mode="full",online=True)
    assert out.status=="failed" and out.attempts==2

@pytest.mark.parametrize("content", ["```json\n{}\n```", "{} text", "{}{}", "{", "[]"])
def test_strict_json_fallback(content):
    with pytest.raises(TranslationFailure): parse_response({"choices":[{"message":{"content":content}}]})

def test_client_key_retry_cache_and_redaction(tmp_path):
    client=DeepSeekClient(DeepSeekConfig(max_retries=1),transport=Fake([]),environ={},sleep=lambda _:None)
    with pytest.raises(TranslationFailure,match="DEEPSEEK_API_KEY"): client.complete([],online=True,request_id="none")
    fake=Fake([TranslationFailure("HTTP_429","later"),{"status":"rejected","calls":[],"error":{"code":"X"}}]); client=DeepSeekClient(DeepSeekConfig(max_retries=1),transport=fake,environ={"DEEPSEEK_API_KEY":"secret"},sleep=lambda _:None)
    response,meta=client.complete([],online=True,request_id="rate")
    assert meta["retries"]==1 and len(fake.calls)==2 and all("secret" not in json.dumps(x[2]) for x in fake.calls)
    fake=Fake([TranslationFailure("HTTP_400","bad")]); client=DeepSeekClient(DeepSeekConfig(max_retries=2),transport=fake,environ={"DEEPSEEK_API_KEY":"secret"},sleep=lambda _:None)
    with pytest.raises(TranslationFailure): client.complete([],online=True,request_id="four")
    assert len(fake.calls)==1

def test_tool_call_and_rewrite_does_not_see_labels(tmp_path):
    row,_,snap=_snapshot_and_row(); valid={"status":"rejected","calls":[],"error":{"code":"X"}}
    assert parse_response({"choices":[{"message":{"content":"","tool_calls":[{"function":{"arguments":json.dumps(valid)}}]}}]})==valid
    seen=[]
    def transport(url,headers,payload,timeout):
        seen.append(payload); return {"choices":[{"message":{"content":json.dumps({"rewrite":"请保持 R1 和 20 不变。"})}}]}
    client=DeepSeekClient(DeepSeekConfig(cache_directory=str(tmp_path)),transport=transport,environ={"DEEPSEEK_API_KEY":"secret"})
    selected={**row,"rewrite_selected":True,"original_text":"请将 R1 的权重设为 20。"}
    result=rewrite_record(selected,client,online=True,cache=JsonCache(tmp_path))
    assert result["original_text"]==selected["original_text"] and "expected_calls" not in json.dumps(seen,ensure_ascii=False)
