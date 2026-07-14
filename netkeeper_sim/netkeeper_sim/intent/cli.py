"""Offline-default CLI.  Network traffic is impossible without ``--online``."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from netkeeper_sim.dataset.intents import _read_records, _snapshot, validate_intent_dataset
from netkeeper_sim.intent.evaluation import evaluate_dataset
from netkeeper_sim.intent.translation import DeepSeekClient, PromptBuilder, TranslationFailure, Translator, load_config, rewrite_record

def _translator(root: Path) -> Translator:
    return Translator(DeepSeekClient(load_config()), PromptBuilder(root / "intents" / "few_shot_candidates.jsonl"))
def _rows(root: Path, split: str): return [json.loads(x) for x in (root / "intents" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
def _one(root: Path, split: str, row):
    record=next(x for x in _read_records(root,split) if x["scenario_id"]==row["scenario"]["scenario_id"])
    return record, _snapshot(root,record)
def _append(path: Path, value: dict, resume: bool):
    seen=set()
    if resume and path.is_file(): seen={json.loads(x)["intent_id"] for x in path.read_text(encoding="utf-8").splitlines() if x.strip()}
    if value["intent_id"] not in seen:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("a",encoding="utf-8") as h: h.write(json.dumps(value,ensure_ascii=False,default=dict)+"\n")

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("command",choices=("translate","translate-dataset","rewrite-dataset","validate-output","dry-run","evaluate")); p.add_argument("--dataset-root",type=Path,default=Path("data/netkeeper_lite")); p.add_argument("--split",default="test"); p.add_argument("--index",type=int,default=0); p.add_argument("--start",type=int,default=0); p.add_argument("--stop",type=int); p.add_argument("--text"); p.add_argument("--mode",choices=("prompt_only","few_shot","full"),default="full"); p.add_argument("--online",action="store_true"); p.add_argument("--output",type=Path); p.add_argument("--resume",action="store_true")
    a=p.parse_args(); rows=_rows(a.dataset_root,a.split); translator=_translator(a.dataset_root)
    if a.command=="validate-output": print(json.dumps(validate_intent_dataset(a.dataset_root),ensure_ascii=False)); return
    if a.command=="evaluate":
        print(json.dumps(evaluate_dataset(translator,a.dataset_root,split=a.split,mode=a.mode,output_directory=a.output or (a.dataset_root/"intent_evaluations"),online=a.online,start=a.start,stop=a.stop,resume=a.resume),ensure_ascii=False)); return
    row=rows[a.index]; record,snap=_one(a.dataset_root,a.split,row)
    if a.command=="dry-run": print(json.dumps({"messages":translator.builder.build(a.text or row["natural_language"],snap,mode=a.mode,scenario_id=record["scenario_id"]),"online":False},ensure_ascii=False)); return
    try:
        if a.command=="translate": print(json.dumps(translator.translate(a.text or row["natural_language"],snap,request_id=row["intent_id"],mode=a.mode,online=a.online,scenario_id=record["scenario_id"]).__dict__,ensure_ascii=False,default=dict)); return
        output=a.output or (a.dataset_root/"intents"/("rewrites.jsonl" if a.command=="rewrite-dataset" else "translations.jsonl"))
        for item in rows[a.start:a.stop]:
            rec,ss=_one(a.dataset_root,a.split,item)
            value=rewrite_record(item,translator.client,online=a.online) if a.command=="rewrite-dataset" else {"intent_id":item["intent_id"],"result":translator.translate(item["natural_language"],ss,request_id=item["intent_id"],mode=a.mode,online=a.online,scenario_id=rec["scenario_id"]).__dict__}
            _append(output,value,a.resume)
        print(json.dumps({"output":str(output),"count":len(rows[a.start:a.stop])},ensure_ascii=False))
    except TranslationFailure as exc:
        print(json.dumps({"status":"failed","code":exc.code,"message":exc.message},ensure_ascii=False)); raise SystemExit(2)
if __name__=="__main__": main()
