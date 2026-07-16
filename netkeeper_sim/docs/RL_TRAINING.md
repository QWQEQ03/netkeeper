# Unified COMA training

The current model/training/checkpoint tuple is
`rl-coma-v3` / `coma-counterfactual-v4` / `training-state-v2`. Checkpoints made
before this tuple are inference-only and cannot be resumed. The actor and
critic consume current link/route state, per-agent TD targets use role-aligned
rewards, and model-based counterfactual replay supplies simulator-derived
advantages. Semantically equivalent actions are masked and the monotonic
shield rejects joint edits that reduce immediate total reward.

`python -m netkeeper_sim.rl.cli --config configs/rl_debug_train.yaml --dataset-root ../data/netkeeper_lite --output-root runs`
creates a unique run directory with resolved YAML, JSONL/CSV metrics, latest
and validation-selected best checkpoints.  Resume uses `--resume RUN/latest.pt`
with the same resolved configuration.  Checkpoints are strict: model version,
configuration, state dictionaries, optimizers, scaler, replay and RNG state
must match. `latest.pt` is the transaction commit point; metrics are rebuilt
from its embedded records after interruption, so resume cannot mix stale log
rows with an older checkpoint.

Train reads only `scenarios/train.jsonl`; validation reads only validation JSONL
through a fixed seeded sampler.  Test and dynamic test have no runner path.
Validation is deterministically stratified and selects checkpoints by paired
reward improvement over `no_update`, not by raw reward.  Static episodes stop
after the configured number of consecutive steps without policy-consistency or
MLU improvement.  JSONL records include per-agent action/no-op rates, unique
action counts, and role-aligned diagnostic rewards.
The dispatcher is `TrainedPolicyDispatcher(checkpoint)` and returns one greedy
`JointAction` for exactly the supplied Snapshot; API execution performs the
single environment step.

A candidate may be frozen for formal evaluation only when its paired validation
delta is strictly positive, the validation summary is complete, and each of
OSPF, BGP, and Performance has at least one accepted non-noop action. The freeze
command enforces these conditions and strict-loads the checkpoint against the
current dataset manifest:

`python -m netkeeper_sim.rl.cli --freeze-checkpoint RUN/best.pt --dataset-root ../data/netkeeper_lite`

Debug defaults are 32 hidden, 2 GINE + 1 Transformer, 2 heads, batch 2,
20 steps and 20 episodes.  Experiment defaults are 64 hidden, 2+2 layers,
4 heads, batch 4, 50 steps and 500 episodes.  Suggested RTX 4060 commands:

`python -m netkeeper_sim.rl.cli --config configs/rl_experiment.yaml --episodes 100 ...`

then repeat with `--episodes 200` and `--episodes 500 --resume RUN/latest.pt`.
Do not freeze or evaluate the short readiness runs as formal experiments.
