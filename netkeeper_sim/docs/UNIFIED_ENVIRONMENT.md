# Unified schema environment

`UnifiedNetworkEnvironment` is the canonical deterministic environment.  Its
only runtime state boundary is an immutable `NetworkSnapshot`:

```text
NetworkScenario
  -> reset -> NetworkSnapshot + AgentObservation
  -> step(JointAction, Event*) -> StepResult + next NetworkSnapshot
```

## Core schema

- `NetworkScenario`: topology, immutable initial configuration, traffic,
  policies, scheduled events, `max_steps`, optional `target_mlu`.
- `NetworkSnapshot`: topology/configuration/traffic/policies plus evaluated
  routing table, directed loads and versioned metrics.
- `AtomicAction`: `agent_id`, `parameter_type`, target mapping,
  `set|delta|no_update`, value and mask/validity.
- `JointAction`: ordered atomic actions for one snapshot.
- `StepResult`: previous/next IDs, observation, `RewardBreakdown`, terminal
  flags, changed configuration diff, metrics and `ErrorRecord`s.

`NetworkConfiguration` is the only configuration state.  `Topology` is
physical and immutable.  A link target always uses `link_id`, including
parallel links.

## Example

```python
from netkeeper_sim.schemas import AtomicAction, JointAction, NetworkScenario
from netkeeper_sim.simulator import UnifiedNetworkEnvironment

scenario = NetworkScenario("S:demo", topology, traffic, policies)
env = UnifiedNetworkEnvironment()
snapshot, observation = env.reset(scenario, seed=7)

action = JointAction((
    AtomicAction("ospf", "ospf_weight", {"link_id": "L:R0--R1:0"}, "set", 10),
), snapshot_id=snapshot.snapshot_id)
result = env.step(snapshot, action)
```

The fixed step order is:

```text
scheduled events -> post-event observation -> action validation
-> configuration update -> OSPF/BGP -> traffic/performance
-> policy/metrics -> reward -> next snapshot
```

Invalid events are isolated by event; invalid actions reject the entire action
batch.  Neither may partially change configuration.  `no_update` advances time
but does not change configuration version.  `terminated` requires an explicit
scenario `target_mlu` and feasible policy consistency; policy consistency of 1
alone is not terminal.  `truncated` means `max_steps` was reached.

## Compatibility

`NetworkSimulationEnvironment` and `MultiAgentNetworkEnvironment` remain
available for old CLI/RL tests.  `NetworkSimulationEnvironment.simulate_schema`
already exposes the schema deterministic kernel.  Legacy actor dictionaries can
be converted without changing actor heads with:

```python
from netkeeper_sim.rl.joint_action_adapter import legacy_action_to_joint_action
```

Remaining migration callers are `rl/multi_agent_env.py`, `rl/trainer.py`, and
the old CLI/evaluation façade.  They still use legacy mutable topology and old
`StepResult`; migrate them only with the fifth-block actor/critic/COMA work.

## First-block acceptance

The tests cover deterministic reset, immutable history, OSPF/BGP/performance
actions, no-op, link/node/traffic/policy events, policy consistency, MLU,
paper/project Traffic Shift, invalid input atomicity, and replay.

Not included here: Zoo/scenario dataset generation (block 2), network API
validation/execution (block 3), natural-language/DeepSeek (block 4), Actor,
Critic, COMA and reward redesign (block 5), or baseline/formal experiments
(blocks 6--8).
