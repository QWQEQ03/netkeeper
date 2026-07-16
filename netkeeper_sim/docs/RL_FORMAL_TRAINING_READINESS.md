# RL formal training readiness

Status: **ready for formal training** (2026-07-15).

The fixed-seed, multi-topology readiness run used the current dataset/model
semantics and reached the formal candidate gate at episode 59:

- paired validation reward delta versus no-update: `+0.158004003`
- positive validation scenarios: `60 / 60`
- accepted non-noop actions: OSPF `17`, BGP `60`, Performance `1`
- focused regression tests before the run: `55 passed`
- release validation: valid, with `3000 / 400 / 500` static scenarios and
  `100` dynamic sequences
- intent dataset validation after release regeneration: valid

The readiness checkpoint under `/tmp/netkeeper_training_readiness_v2` is only
evidence that the training design meets the gate. It is not a formal artifact
and must not be used by `configs/evaluation_formal.yaml`.

Start each formal run from the committed v2 release and the experiment config:

```bash
python -m netkeeper_sim.rl.cli \
  --config configs/rl_experiment.yaml \
  --episodes 500 \
  --dataset-root ../data/netkeeper_lite \
  --output-root runs \
  --seed SEED
```

After training, freeze only the validation-selected candidate. The CLI rejects
non-positive validation delta, incomplete summaries, inactive agents, stale
model/training/checkpoint versions, or a mismatched dataset manifest.
