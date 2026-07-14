# Block 5 acceptance report

## Verdict

**Conditionally passed.** Unified dataflow, lightweight model, COMA toy tests,
checkpoint/resume and dispatcher are implemented.  A fixed 50-episode scenario
run and 24-episode 12-topology Debug run completed; neither establishes useful
learning, so no formal result is claimed.

## Actual bounded runs

| run | budget | result |
|---|---:|---|
| fixed Medium train scenario `S:train:00300` / `zoo:arpanet19706:381c167f` | 50x20 steps, seed 20260714 | initial policy=.75; first-10 reward=-2.4175, policy=.6917, MLU=1.4420, shift=.0260; last-10 reward=-2.1699, policy=.7000, MLU=1.0636, shift=.0356; best train reward=-1.0663 at episode 44; best validation=-.01333 |
| balanced train debug | 24x20 steps, seed 20260714 | all 12 train topologies each sampled exactly twice; first-3 reward=-2.6404/policy=.6389/MLU=1.8380; last-3=-3.1729/.5833/1.7495; validation reward=-.02307 |
| experiment startup | 2x3 CPU steps | AMP false, peak GPU=0; finite loss/entropy/grad |

Validation uses three fixed topology-distinct records (policy/MLU/shift):
`(.5,.25,0)`, `(.5417,.1295,0)`, `(.6667,.75,0)`. The fixed run is a
diagnostic, not convergence evidence. Artifacts were written under `/tmp`, not
committed experiment results.

## Tests

COMA toy math, mask/no-op, joint-context, gradient boundary, target interval,
CPU update and unified/API regression pass. CPU exact-resume now matches
reward=-.3574009377, two steps, global_step=6 and every Actor parameter bitwise
after saving replay RNG and sampler state. CUDA is unavailable in this
environment, so AMP/RTX validation is explicitly pending user hardware.

Completed pytest groups: 16 intent tests and 60 unified-RL/schema/API tests
passed. A single `pytest -q` was started after migration; it was interrupted by
the execution window before final aggregation, so a full-suite number is not
claimed in this report.

## Final reward-accounting audit

Both training and validation now call the same `aggregate_trajectory()` helper.
`reward` is the sum of `RewardBreakdown.total_reward` over environment steps;
`mean_reward` is that sum divided by `steps`; `breakdown_total` and
`breakdown_mean` use the same rule for every component.  Best checkpoint still
compares validation **total** reward, always on the fixed validation trajectory
budget, rather than mixing total and mean.  A regression test passes the exact
same two-step trajectory through both paths and obtains identical total, mean
and complete breakdown.

The large train/validation difference is therefore not a unit mismatch: train
uses exploratory actions and accumulates 20 modified steps, while validation is
greedy/no-grad and generally selects no-update on different validation
scenarios.  In the fixed 50-episode run, per-step breakdown means for the
first/last ten episodes were respectively: policy `0.0000/-0.0083`, MLU
`.0255/.0102`, Traffic Shift `-.0130/-.0178`, configuration `-.0290/-.0300`,
total `-.0247/-.0460`.  Policy is not numerically overwhelmed by MLU; rather,
the learned policy did not consistently select policy-improving candidates.

Mask response audit found 641 valid OSPF candidates and 193 valid BGP
candidates in the fixed Medium scenario.  OSPF candidate 131
(`L:R1--R3:0`, weight 3) changes policy consistency from `.75` to `.8333` in
one accepted unified-environment step.  Thus the action mapping/mask does not
make policy improvement unreachable.

## CUDA manual verification

The user manually verified: `torch.cuda.is_available()=True`, one GPU,
`NVIDIA GeForce RTX 4060 Laptop GPU`, 8.0 GB, PyTorch `2.6.0+cu124`, CUDA build
12.4.  No manual CUDA forward/update, tensor-device, GradScaler state or peak
memory number was supplied, so this report deliberately does not claim them.

## Deferred work

Formal 500 episode training, baselines, test evaluation, dynamic adaptation
and nine ablations are reserved for Blocks 6-8.
