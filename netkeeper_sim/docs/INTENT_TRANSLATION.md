# Intent translation (Block 4)

## Flow

`IntentRecord.natural_language` and a reconstructed Block-2 snapshot enter
`PromptBuilder`, then `Translator`, `TranslationResult`, Block-3 validation,
and optionally the transactional executor.  The executor is only invoked for
accepted output.  A recording dispatcher proves that an optimization request
was dispatched; it never claims that an optimizer improved the network.

`PromptBuilder` dynamically enumerates `API_REGISTRY` and
`export_json_schema()`.  It supplies snapshot-derived nodes, link IDs and
endpoints, policy IDs, BGP targets, OD pairs and state.  Test gold labels are
never included in prompts, feedback, retrieval or cache keys.

## Data and schemas

`data/netkeeper_lite/intents/{train,validation,test}.jsonl` contains
1,200/300/500 records.  Their immutable `IntentRecord` has scenario reference,
template/family provenance, original text, expected translation/calls,
validity/error information, rewrite selection and content hash.  Train/test
template families are disjoint; `few_shot_candidates.jsonl` contains fixed
train-only IDs.

`TranslationResult` is a strict tagged envelope:

```json
{"status":"accepted","calls":[{"api":"...","arguments":{}}],"need_optimization":false}
```

or

```json
{"status":"rejected","calls":[],"error":{"code":"TRANSLATION_UNKNOWN_ENTITY"}}
```

## Modes

All modes use `PROMPT_VERSION=netkeeper-translation-v1`, identical model and
generation configuration.

| Mode | fixed train few-shot | structured request | validator feedback |
|---|---:|---:|---:|
| `prompt_only` | no | no | no |
| `few_shot` | yes | no | no |
| `full` | yes | yes | one maximum |

Feedback contains only code, call index, API and short message.  JSON parsing
accepts exactly one object or one tool call; code, fences, extra text and
multiple/truncated JSON are rejected.

## Configuration and safety

Only `DEEPSEEK_API_KEY` supplies the credential.  `.env` is ignored; the
repository provides `.env.example`.  Other supported environment variables are
`DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`, `DEEPSEEK_TIMEOUT`,
`DEEPSEEK_TEMPERATURE`, `DEEPSEEK_MAX_TOKENS` and
`DEEPSEEK_STRUCTURED_OUTPUT_MODE`; non-sensitive defaults are in
`configs/deepseek.example.yaml`.

All online calls require `--online`.  The client has finite retry with jitter,
timeout, rate limiting and injectable transport.  Cache keys hash provider,
model, mode, prompt version, input, context and schema.  Cache directories are
ignored and default cache entries contain validated structured output and usage
metadata, not Authorization headers or raw credentials.  Rewrite is a separate
derived record: it sees only original text, preserves it, records hashes and
marks failed entity/number preservation for review.

## CLI

Run from `netkeeper_sim/` with `PYTHONPATH=.`:

```bash
python -m netkeeper_sim.intent.cli dry-run --split test --index 0 --mode full
python -m netkeeper_sim.intent.cli translate --split test --index 0 --mode full --online
python -m netkeeper_sim.intent.cli translate-dataset --split test --start 0 --stop 5 --output /tmp/nk-smoke.jsonl --resume --online
python -m netkeeper_sim.intent.cli evaluate --split test --start 0 --stop 5 --mode full --output /tmp/nk-eval --resume --online
python -m netkeeper_sim.intent.cli rewrite-dataset --split train --start 0 --stop 5 --output /tmp/nk-rewrites.jsonl --resume --online
```

The 3–5 sample commands are the only recommended online smoke.  Outputs belong
under `/tmp` or another ignored directory.  Do not run the formal 500-sample
experiment in Block 4.  Block 8 reuses `evaluate` with `--split test --start 0
--stop 500` for each mode, separate output directories, cache and resume.

## Evaluation

`netkeeper_sim.intent.evaluation.evaluate_dataset` emits one JSONL item per
`intent_id+mode`, summary JSON and compact CSV.  It reports strict API-name and
argument accuracy, exact match, accept/reject accuracy, invalid rejection
precision/recall/F1, execution success, direct semantic success, ordered
one-to-one call precision/recall/F1, failure groups, feedback recovery, calls,
latency, tokens, cache hits and optimization dispatches.  Values, enums and all
network identifiers compare exactly.  Semantic success excludes optimization
performance claims; optimization-only requests are not applicable.

Known limitation: direct semantic checks cover policy, traffic, failure/state
and configuration changes.  Performance improvements requiring Block-5 agents
remain intentionally unscored here.
