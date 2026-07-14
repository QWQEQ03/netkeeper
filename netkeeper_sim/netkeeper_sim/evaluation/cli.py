"""`python -m netkeeper_sim.evaluation.cli` batch evaluation command."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
from netkeeper_sim.evaluation.batch import METHOD_NAMES, aggregate_output, execute, plan, prepare_output, validate_output
from netkeeper_sim.evaluation.manifest import generate_evaluation_manifest

def _config(args):
    loaded=yaml.safe_load(Path(args.config).read_text()) if args.config else {}
    random_seeds=loaded.get("random_seeds",loaded.get("seeds",[20260714]))
    value={"methods":loaded.get("methods",["no_update"]),"deterministic_seed":int(loaded.get("deterministic_seed",random_seeds[0])),"random_seeds":list(random_seeds),"mode":loaded.get("mode","static"),"max_steps":loaded.get("max_steps"),"hold_steps":loaded.get("hold_steps",3),"local_search_budget":loaded.get("local_search_budget",64),"local_search_deltas":loaded.get("local_search_deltas",[1,2,4,8]),"traffic_shift_primary":loaded.get("traffic_shift_primary","paper_v1"),"traffic_shift_secondary":loaded.get("traffic_shift_secondary","project_v1"),"evaluation_manifest":loaded.get("evaluation_manifest"),"checkpoint":loaded.get("checkpoint",None),"checkpoint_manifest":loaded.get("checkpoint_manifest",None),"checkpoint_status":loaded.get("checkpoint_status","debug_unconverged"),"require_formal_checkpoint":bool(loaded.get("require_formal_checkpoint",False)),"scenario_ids":loaded.get("scenario_ids",[]),"sequence_ids":loaded.get("sequence_ids",[]),"device":loaded.get("device","cpu"),"recovery_budget":loaded.get("recovery_budget",30)}
    for key in ("methods","seeds","scenario_ids","sequence_ids"):
        override=getattr(args,key,None)
        if override: value[key]=override.split(",") if key!="seeds" else [int(x) for x in override.split(",")]
    if getattr(args,"seeds",None): value["random_seeds"]=[int(x) for x in args.seeds.split(",")]
    for key in ("mode","checkpoint","max_steps","local_search_budget"):
        override=getattr(args,key,None)
        if override is not None: value[key]=int(override) if key in {"max_steps","local_search_budget"} else override
    return value

def main(argv=None):
    parser=argparse.ArgumentParser(description="NetKeeper schema-only evaluation runner")
    sub=parser.add_subparsers(dest="command",required=True)
    freeze=sub.add_parser("freeze-manifest"); freeze.add_argument("--dataset-root",required=True); freeze.add_argument("--output",required=True); freeze.add_argument("--deterministic-seed",type=int,default=20260714); freeze.add_argument("--random-seeds",default="20260714,20260715,20260716")
    run=sub.add_parser("run"); run.add_argument("--dataset-root",required=True); run.add_argument("--output",required=True); run.add_argument("--config"); run.add_argument("--methods"); run.add_argument("--seeds"); run.add_argument("--mode",choices=("static","dynamic","both")); run.add_argument("--scenario-ids"); run.add_argument("--sequence-ids"); run.add_argument("--max-steps",type=int); run.add_argument("--local-search-budget",type=int); run.add_argument("--checkpoint"); run.add_argument("--evaluation-manifest"); run.add_argument("--resume",action="store_true"); run.add_argument("--overwrite",action="store_true"); run.add_argument("--dry-run",action="store_true")
    ag=sub.add_parser("aggregate"); ag.add_argument("--output",required=True); ag.add_argument("--config-hash",required=True); ag.add_argument("--group-by",default="method_name")
    va=sub.add_parser("validate-results"); va.add_argument("--output",required=True); va.add_argument("--config-hash",required=True)
    args=parser.parse_args(argv)
    if args.command=="freeze-manifest":
        value=generate_evaluation_manifest(args.dataset_root,seeds=tuple(int(x) for x in args.random_seeds.split(",")),deterministic_seed=args.deterministic_seed,hold_steps=3)
        target=Path(args.output); target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps({"path":str(target),"manifest_hash":value["manifest_hash"],"static":len(value["static"]),"dynamic":len(value["dynamic"])},ensure_ascii=False)); return 0
    if args.command=="run":
        config=_config(args); config["evaluation_manifest"]=args.evaluation_manifest or config.get("evaluation_manifest"); manifest,tasks=plan(args.dataset_root,config)
        if args.dry_run:
            print(json.dumps({"planned":len(tasks),"methods":config["methods"],"static":sum(x["kind"]=="static" for x in tasks),"dynamic":sum(x["kind"]=="dynamic" for x in tasks),"tasks":[{"method":x["method"],"scenario_id":x["scenario_id"],"sequence_id":x.get("sequence_id"),"seed":x["seed"]} for x in tasks]},ensure_ascii=False)); return 0
        store=prepare_output(args.output,manifest,resume=args.resume,overwrite=args.overwrite); print(json.dumps(execute(args.dataset_root,config,store,manifest),ensure_ascii=False)); aggregate_output(store); return 0
    root=Path(args.output)/f"evaluation-{args.config_hash[:12]}"; from netkeeper_sim.evaluation.results import ResultStore; store=ResultStore(root)
    if args.command=="aggregate": print(json.dumps(aggregate_output(store,group_by=tuple(args.group_by.split(","))),ensure_ascii=False)); return 0
    result=validate_output(store); print(json.dumps(result,ensure_ascii=False)); return 0 if result["valid"] else 1
if __name__=="__main__": raise SystemExit(main())
