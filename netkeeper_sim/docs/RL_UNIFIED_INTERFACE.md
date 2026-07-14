# Unified RL data interface

RL consumes only `NetworkSnapshot` values emitted by `UnifiedNetworkEnvironment`.
The graph adapter has 17 node features and 11 directed-edge features, in the
documented order in `rl/schema_adapter.py`.  It uses only fixed schema limits:
OSPF `/65535`, bandwidth `/physical_bandwidth`, capacity `/bandwidth`, queue
`log1p(x)/log1p(1_000_000)`, delay `/1000`, and current utilization clipped to
`[0,1]`.  It never computes dataset-wide statistics or reads future events.

Each step permits at most one action per agent.  Candidate zero is always
`no_update`; all other candidates identify an allowed target, parameter and one
of 64 values.  OSPF controls only `ospf_weight`; BGP controls only
`local_preference`, `as_path_length`, `med`; Performance controls only
`bandwidth_bps`, `capacity_bps`, `queue_packets`.  Down links/nodes, disabled or
unknown BGP routes and infeasible targets are masked.  An all-invalid target set
therefore leaves only no-update.

The shared reward is produced solely by `StepResult.rewards`.  Its YAML weights
multiply policy-consistency improvement, MLU improvement, step project Traffic
Shift penalty, changed-configuration count penalty, rejected-action count and
dropped-traffic ratio.  `include_traffic_shift: false` is the Block-8 ablation
switch.  Defaults are deliberately O(1), except each configuration change is
0.01, so no term is silently amplified by topology size.

## Lightweight COMA contract

The Actor uses one 2xGINEConv + 2xTransformerConv encoder (`hidden=64`,
`heads=4`, `dropout=0.1`) and three independent two-layer candidate heads.
Each head returns `[1 + entities * parameters * 64]`; candidate zero is
no-update.  Link candidates use the mean of the two endpoint embeddings and
route candidates use router/next-hop embeddings, so they are not copies of one
pooled graph vector.  The Critic has an independent encoder and returns the
same candidate shape for `Q_i(s,u_-i,a_i)`.

For a macro action per agent, other two rollout actions remain fixed and:

`b_i = sum_a pi_i(a|tau_i) Q_i(s,u_-i,a)` and
`A_i = detach(Q_i(s,u)-b_i)`.  The actor loss is
`-mean(A_i * log pi_i(chosen)) - entropy_coef * entropy`.  All terms use the
same candidate mask; padded/invalid candidates have zero probability.  The
target is one-step `r + .85*(1-terminated)*target_Q`; a time-limit truncation
continues to bootstrap.  AMP is CUDA-only and unscales before clipping to 1.0.
