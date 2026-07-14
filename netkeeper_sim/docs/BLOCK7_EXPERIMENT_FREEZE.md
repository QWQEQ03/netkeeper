# Block 7 experiment freeze

Evaluator protocol: `netkeeper-evaluation-v2`. Frozen evaluation manifest:
`configs/frozen_evaluation_manifest.json`. Formal configuration:
`configs/evaluation_formal.yaml`.

The manifest contains all 500 static test scenarios and all 100 dynamic test
sequences. Static max steps are 50; dynamic max steps are 240; success and
recovery require three consecutive qualifying states; every logical recovery
budget is 30 steps. Paper-v1 Traffic Shift is primary and project-v1 is the
project-wide robustness definition.

Load groups are mutually exclusive and complete for the published schema:
Normal is Normal gravity/diurnal (166); Hotspot is Low hotspot (125); Burst is
Low burst (42); High-load is High/3.0 (167). This avoids inventing Normal
hotspot/burst records that do not exist in the frozen dataset.

Deterministic methods use seed 20260714. Random uses the predeclared seeds
20260714, 20260715 and 20260716. With one formal checkpoint, the matrix is 3500
static plus 700 dynamic tasks. Random repetitions are not independent
topologies.

The formal checkpoint gate requires train-only fitting, validation-total-reward
selection, resolved Experiment config, dataset manifest and seed provenance,
model/schema versions, strict actor loading, accepted validation greedy
dispatch, `formal_validation_selected` status, and matching SHA-256 in
`checkpoint_manifest.json`. Test data is never used for checkpoint selection.

The frozen checkpoint is `runs/rl-f27e74f349/best.pt`, selected at episode 59
from a 100-episode Experiment run with validation total reward
0.02668009693884514. Its formal SHA-256 is
`6ec7a3fbba370dab7656916a1d6b7737b8a29ce3f9925be94b5c71038afeb728`.
The resolved config SHA-256 is
`6fd819c324df15ee94cf71078c10e61283aaaa0d2cb09fd41202fef7b3069195`.

Any correctness fix after the first formal task increments the evaluator or
method version, invalidates affected run keys, and requires a complete rerun of
every affected method/scenario/seed cell. Failed runs remain in raw output.
Invalidated results must not be used to modify the frozen checkpoint, seeds,
groups, thresholds, budgets or method hyperparameters.
