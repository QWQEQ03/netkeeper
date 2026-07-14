# Unified COMA training

`python -m netkeeper_sim.rl.cli --config configs/rl_debug_train.yaml --dataset-root ../data/netkeeper_lite --output-root runs`
creates a unique run directory with resolved YAML, JSONL/CSV metrics, latest
and validation-selected best checkpoints.  Resume uses `--resume RUN/latest.pt`
with the same resolved configuration.  Checkpoints are strict: model version,
configuration, state dictionaries, optimizers, scaler, replay and RNG state
must match.

Train reads only `scenarios/train.jsonl`; validation reads only validation JSONL
through a fixed seeded sampler.  Test and dynamic test have no runner path.
The dispatcher is `TrainedPolicyDispatcher(checkpoint)` and returns one greedy
`JointAction` for exactly the supplied Snapshot; API execution performs the
single environment step.

Debug defaults are 32 hidden, 2 GINE + 1 Transformer, 2 heads, batch 2,
20 steps and 20 episodes.  Experiment defaults are 64 hidden, 2+2 layers,
4 heads, batch 4, 50 steps and 500 episodes.  Suggested RTX 4060 commands:

`python -m netkeeper_sim.rl.cli --config configs/rl_experiment.yaml --episodes 100 ...`

then repeat with `--episodes 200` and `--episodes 500 --resume RUN/latest.pt`.
Do not treat the short validation runs as formal experiments.
