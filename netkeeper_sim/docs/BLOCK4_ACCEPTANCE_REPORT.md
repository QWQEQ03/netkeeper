# Block 4 acceptance report

## Verdict

**Code and offline workflow: passed.**  **Real DeepSeek connectivity: pending
an explicit user-run `--online` smoke command.**  No online request was made
during this acceptance.

## Evidence

- Intent data validation checks all 2,000 records, scenario/topology split,
  family isolation, hashes, real entities and Block-3 validator/dry-run calls.
- Formal data quotas are train 1,200, validation 300 and test 500; levels are
  40/40/20 per split and test has exactly 60 rejected records.
- Train/test template-family intersection is empty; few-shot IDs are train-only.
- All 19 registered APIs occur in every split.  The test rejects cover unknown
  entity/policy/BGP targets, range, endpoint/waypoint/isolation, conflict,
  missing/ambiguous/unsupported and ordering cases.
- Fake-client tests cover accepted/rejected, single/multi call, tool output,
  strict JSON fallback, retry classes, cache, no-key, rewrite label isolation,
  one feedback correction, dispatcher and all three evaluation modes.
- Final offline test suite: `184 passed, 8 skipped`.
- Evaluator outputs per-sample JSONL, aggregate JSON and CSV; it is resumable
  by `intent_id+mode` and does not feed gold into the translator.
- Credential/cache scan found no committed `.env`, cache directory, literal
  DeepSeek credential or Authorization value.  Source contains the required
  Authorization header construction only; logs/results do not serialize it.

## Online smoke (not run)

After exporting `DEEPSEEK_API_KEY`, the user may explicitly run from
`netkeeper_sim/`:

```bash
PYTHONPATH=. python -m netkeeper_sim.intent.cli evaluate --dataset-root ../data/netkeeper_lite --split test --start 0 --stop 5 --mode full --output /tmp/netkeeper-deepseek-smoke --resume --online
```

Expected files are `test.full.jsonl`, `test.full.json` and `test.full.csv` under
`/tmp/netkeeper-deepseek-smoke`.  This has a five-call upper bound plus at most
one correction per item; cache/resume prevents repeat spending.

Block 8 reuses the same evaluator for each mode with `--stop 500`.  It is not
executed by this block.
