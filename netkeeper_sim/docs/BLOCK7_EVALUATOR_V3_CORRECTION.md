# Block 7 evaluator-v3 correctness correction

## Scope and verdict

Evaluator v2 formal output is invalidated.  Overlapping or interrupted resume
workers could interleave file-level read/modify/replace operations.  Since
steps, event recovery rows, and the terminal record were committed separately,
the output could contain a terminal record with missing or mismatched trajectory
rows.  Static validation observed missing terminals, non-contiguous steps, and
snapshot-chain mismatches.

## Versioned correction

- evaluator version: `netkeeper-evaluation-v2` -> `netkeeper-evaluation-v3`;
- checkpoint adapter: `rl-coma-v2-adapter.1` -> `.2`;
- the evaluator version is now included in evaluator config and run-manifest
  hashes, so v3 cannot reuse a v2 result directory or run key;
- a run is committed under an inter-process lock; orphan pre-terminal step and
  event rows are replaced, and the terminal record is written last;
- concurrent workers recheck terminal existence while holding the lock;
- validator checks orphan rows, step-key/index continuity, snapshot-to-terminal
  consistency, logical event count/projection, and cross-method event schedule;
- the checkpoint method config hash now covers only its greedy inference
  protocol, not static/dynamic plan fields, so the same method has one identity
  in both experiment modes.

No checkpoint, model weights, dataset/manifest, seed, threshold, recovery
budget, Local Search neighbourhood/candidate budget, or baseline behavior was
changed.

## Invalidated artifacts

The v2 directories were retained and renamed with
`-evaluator-v2-invalid`.  They are audit-only and must never be aggregated,
plotted, migrated, or combined with v3 output.

## Verification

- focused evaluation regression: 12 passed;
- final complete regression after both corrections: 206 passed, 33 skipped;
- v3 checkpoint validation smoke: strict formal bundle accepted, 1/1 terminal,
  validator valid;
- v3 dynamic contract smoke: 240 steps, six paired logical event results,
  recovery budget 30, validator valid;
- second resume: one skipped, zero completed, zero failed.

The formal v3 plan must be regenerated and all affected static and dynamic
cells rerun from the beginning.  No v2 terminal or trajectory row is eligible
for reuse.

Regenerated formal identities:

- static evaluator config hash:
  `6868c853b55b704a8d13f1008339e5c1c1b075c562ac70ad14d9857bfbc1ad4e`;
- dynamic evaluator config hash:
  `ad26515846abad8dee0b14f9c94ad63c046017da36b6f547fd3435a75810b5f9`;
- checkpoint adapter config hash:
  `f5c198c346e63c884da1545f93bbf05201c038d19128c61105f76e066b4d7a5e`.

The v3 seven-cell static pilot passed validation and a second resume skipped
all seven terminals.  The clean 3,500-cell static rerun was then started under
`runs/block7-static-formal-v3`; dynamic formal execution remains gated on that
static run completing and validating.
