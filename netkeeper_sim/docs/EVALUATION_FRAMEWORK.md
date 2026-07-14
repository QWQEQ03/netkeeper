# Evaluation framework (Block 6, evaluator v2)

`netkeeper_sim.evaluation` is a schema-only evaluator: every formal state
transition is `UnifiedNetworkEnvironment.step(JointAction)`.  Methods implement
`reset(context)` and `act(snapshot, observation, context)`, declare immutable
metadata (version/config/checkpoint hashes, permissions, determinism and
lookahead), and cannot mutate topology, traffic or configuration directly.

## Methods and fairness

`no_update` sends no action. `random` uniformly samples the exact Block-5
candidate masks, including no-op, one macro action per OSPF/BGP/Performance
agent. `ospf_default` sets one sorted, legal non-default OSPF link per step to
weight 1 and never adapts. `local_search_ospf` is OSPF-only: ±1/2/4/8, values
1..64, 64 candidates plus a no-op reference, strict lexicographic objective
`max PC, min MLU, min project-v1 shift, min field diff`; each candidate is a
fresh same-snapshot sandbox replay. Its extra simulator calls are logged.

Checkpoint inference is greedy/eval/no-grad through the Block-5 dispatcher.
Formal plans require a `formal_validation_selected` checkpoint bundle whose
SHA-256, dataset manifest, train/validation provenance and validation smoke are
verified before any task is planned. Missing/incompatible debug checkpoints
remain failure records only in non-formal smoke configurations.

## Metrics and results

PC is schema `satisfied/enabled`; MLU is schema maximum directed utilization.
Both paper-v1 and project-v1 Traffic Shift are retained. Configuration ratio is
final differing action-addressable scalar fields / initial field universe,
including link OSPF/performance/state, node state and BGP route parameters;
action count is separate. Success requires three consecutive qualifying states.
Environment termination does not bypass this hold window. Dynamic runs always
consume all 240 steps and all scheduled events; paired link/node recovery is
measured after the up event, with worst values covering the failure interval.
Unreached convergence/recovery is null and censored.

The formal manifest freezes one deterministic seed and three Random seeds.
Random repetitions are paired within scenario; aggregation separately reports
between-scenario variation and within-scenario Random-seed variation.

Output directory is `evaluation-<config-hash>/` and contains resolved config,
run manifest, steps/episodes/failures JSONL, aggregate JSON/CSV and mean±std.
Writes are atomic and run keys include method metadata, scenario/sequence,
seed and evaluator config hash. Resume skips only terminal keys.

## CLI

```bash
python -m netkeeper_sim.evaluation.cli run --dataset-root ../data/netkeeper_lite \
  --output /tmp/eval --config configs/evaluation_smoke.yaml --dry-run
python -m netkeeper_sim.evaluation.cli run ... --resume
python -m netkeeper_sim.evaluation.cli validate-results --output /tmp/eval --config-hash HASH
python -m netkeeper_sim.evaluation.cli aggregate --output /tmp/eval --config-hash HASH --group-by method_name,difficulty
```

The frozen input is `configs/frozen_evaluation_manifest.json`; the formal plan
is `configs/evaluation_formal.yaml`. The plan is intentionally blocked until
its checkpoint and checkpoint manifest identify a qualified bundle.
