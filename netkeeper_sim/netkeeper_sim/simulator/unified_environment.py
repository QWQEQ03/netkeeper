"""Canonical immutable reset/step interface for NetKeeper Sim."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from netkeeper_sim.schemas import (
    AgentObservation, AtomicAction, BGPConfiguration, ErrorRecord, Event,
    JointAction, LinkAttributes, NetworkConfiguration, NetworkScenario,
    NetworkSnapshot, RewardBreakdown, StepResult,
)
from netkeeper_sim.simulator.events import apply_scheduled_events
from netkeeper_sim.simulator.schema_evaluation import evaluate_schema_snapshot


@dataclass(frozen=True)
class RewardConfig:
    """Bounded shared-team reward weights; no RL code recomputes metrics."""
    policy_improvement: float = 1.0
    mlu_improvement: float = 1.0
    traffic_shift_penalty: float = 0.5
    configuration_change_penalty: float = 0.01
    illegal_action_penalty: float = 1.0
    dropped_traffic_penalty: float = 1.0
    include_traffic_shift: bool = True
    component_clip: float = 1.0


class UnifiedNetworkEnvironment:
    """State-light façade; snapshots, not mutable topology objects, are state."""

    def __init__(self, reward_config: RewardConfig | None = None) -> None:
        self.scenario: NetworkScenario | None = None
        self.initial_snapshot: NetworkSnapshot | None = None
        self.current_snapshot: NetworkSnapshot | None = None
        self.seed: int | None = None
        self.reward_config = reward_config or RewardConfig()

    def reset(self, scenario: NetworkScenario, *, seed: int | None = None) -> tuple[NetworkSnapshot, AgentObservation]:
        self.scenario, self.seed = scenario, seed
        configuration = scenario.configuration or NetworkConfiguration.initial(scenario.topology, step=0)
        raw = NetworkSnapshot(0, scenario.topology, configuration, scenario.traffic, scenario.policies)
        evaluated = evaluate_schema_snapshot(raw).snapshot
        self.initial_snapshot = evaluated
        self.current_snapshot = evaluated
        return evaluated, _observation(evaluated)

    def step(
        self,
        snapshot: NetworkSnapshot,
        action: JointAction,
        *,
        scheduled_events: Sequence[Event] = (),
    ) -> StepResult:
        if self.scenario is None or self.initial_snapshot is None:
            raise RuntimeError("reset(scenario) must be called before step")
        # (1) scheduled events; each bad event is isolated by the scheduler.
        event_result = apply_scheduled_events(snapshot, (*self.scenario.events, *scheduled_events))
        event_snapshot = event_result.snapshot
        # (2) observation is constructed after events, before action validation.
        _event_observation = _observation(event_snapshot)
        # (3-4) validate all actions before applying any configuration mutation.
        # The caller selected its action for the supplied pre-event snapshot.
        # Scheduled events may legitimately change the intermediate snapshot
        # identity before validation, so freshness is checked against the
        # caller-visible snapshot while legality uses the post-event state.
        config, action_errors = _apply_joint_action(event_snapshot.configuration, event_snapshot, action, expected_snapshot_id=snapshot.snapshot_id)
        next_step = snapshot.step + 1
        config = config.with_updates(step=next_step)
        raw = event_snapshot.next(configuration=config, step=next_step)
        # (5-7) recomputation, traffic, policy, metrics; prior/initial give shifts.
        evaluated = evaluate_schema_snapshot(raw, previous=snapshot, initial=self.initial_snapshot).snapshot
        changed = snapshot.configuration.diff(evaluated.configuration)
        rewards = _reward(snapshot, evaluated, changed, len(action_errors), self.reward_config)
        truncated = next_step >= self.scenario.max_steps
        # Policy consistency alone is intentionally not terminal: MLU objectives
        # and future recovery events may still matter.
        terminated = bool(
            self.scenario.target_mlu is not None
            and evaluated.metrics.policy_consistency_feasible_only == 1.0
            and evaluated.metrics.maximum_link_utilization <= self.scenario.target_mlu
            and not any(event.step >= next_step for event in self.scenario.events)
        )
        reason = "target_policy_and_mlu_reached" if terminated else ("max_steps" if truncated else None)
        self.current_snapshot = evaluated
        return StepResult(snapshot.snapshot_id, evaluated, _observation(evaluated), rewards, terminated, truncated, reason, changed, evaluated.metrics, (*event_result.errors, *action_errors))


def _apply_joint_action(config: NetworkConfiguration, snapshot: NetworkSnapshot, joint: JointAction, *, expected_snapshot_id: str | None = None):
    errors: list[ErrorRecord] = []
    if joint.snapshot_id is not None and joint.snapshot_id != (expected_snapshot_id or snapshot.snapshot_id):
        return config, [ErrorRecord("stale_action", "action snapshot_id does not match step snapshot")]
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    for index, action in enumerate(joint.actions):
        key = (action.parameter_type, tuple(sorted(action.target.items())))
        if key in seen: errors.append(ErrorRecord("duplicate_action_target", "target+parameter may be changed once", field_path=f"actions[{index}]"))
        seen.add(key)
        if not action.mask or action.valid is False: errors.append(ErrorRecord("masked_action", "action target is masked or invalid", field_path=f"actions[{index}]"))
        else:
            try: _validate_action(action, config, snapshot)
            except (ValueError, TypeError) as exc: errors.append(ErrorRecord("invalid_action", str(exc), field_path=f"actions[{index}]"))
    if errors:
        return config, errors
    candidate = config
    for action in joint.actions:
        if action.mode != "no_update": candidate = _apply_action(candidate, action)
    return candidate, []


def _validate_action(action: AtomicAction, config, snapshot):
    if action.mode == "no_update": return
    if action.parameter_type in {"ospf_weight", "bandwidth_bps", "capacity_bps", "queue_packets", "loss_rate", "link_state"}:
        link_id = action.target.get("link_id")
        if link_id not in {link.link_id for link in snapshot.topology.links}: raise ValueError("unknown_link_id")
    elif action.parameter_type == "node_state":
        if action.target.get("node_id") not in {node.node_id for node in snapshot.topology.nodes}: raise ValueError("unknown_node_id")
    elif action.parameter_type in {"local_preference", "as_path_length", "med"}:
        required = {"router_id", "prefix", "next_hop"}
        if not required <= set(action.target): raise ValueError("BGP action requires router_id/prefix/next_hop")
        if not any((route.router_id, route.prefix, route.next_hop) == tuple(action.target[key] for key in ("router_id", "prefix", "next_hop")) for route in config.bgp.routes): raise ValueError("unknown_bgp_route")
    else: raise ValueError("unsupported_parameter_type")
    if action.parameter_type in {"link_state", "node_state"} and action.mode == "delta": raise ValueError("state action does not support delta")


def _apply_action(config, action):
    value = action.value
    if action.parameter_type == "ospf_weight":
        link_id = action.target["link_id"]; current = config.ospf_weights[link_id]
        return config.with_updates(ospf_weights={link_id: int(value) if action.mode == "set" else current + int(value)})
    if action.parameter_type in {"link_state", "node_state"}:
        state = str(value)
        if state not in {"up", "down"}: raise ValueError("state must be up/down")
        return config.with_updates(**({"link_states": {action.target["link_id"]: state}} if action.parameter_type == "link_state" else {"node_states": {action.target["node_id"]: state}}))
    if action.parameter_type in {"bandwidth_bps", "capacity_bps", "queue_packets", "loss_rate"}:
        link_id = action.target["link_id"]; attrs = config.performance[link_id]
        field = action.parameter_type
        current = getattr(attrs, field)
        candidate = value if action.mode == "set" else current + value
        if field in {"bandwidth_bps", "capacity_bps", "queue_packets"}: candidate = int(candidate)
        attrs = replace(attrs, **{field: candidate})
        return config.with_updates(performance={link_id: attrs})
    target = (action.target["router_id"], action.target["prefix"], action.target["next_hop"])
    routes = []
    for route in config.bgp.routes:
        if (route.router_id, route.prefix, route.next_hop) != target: routes.append(route); continue
        if action.parameter_type == "local_preference": route = replace(route, local_preference=int(value) if action.mode == "set" else route.local_preference + int(value))
        elif action.parameter_type == "med": route = replace(route, med=int(value) if action.mode == "set" else route.med + int(value))
        else:
            length = int(value) if action.mode == "set" else len(route.as_path) + int(value)
            if not 0 <= length <= 255: raise ValueError("as_path_length must be in [0,255]")
            asn = route.as_path[0] if route.as_path else 64512
            route = replace(route, as_path=tuple(asn for _ in range(length)))
        routes.append(route)
    return config.with_updates(bgp=BGPConfiguration(tuple(routes)))


def _observation(snapshot):
    loads = {item.arc_id: item.utilization for item in snapshot.directed_link_loads}
    global_state = {"snapshot_id": snapshot.snapshot_id, "step": snapshot.step, "policy_consistency": snapshot.metrics.policy_consistency, "mlu": snapshot.metrics.maximum_link_utilization, "link_utilization": loads}
    link_mask = {link.link_id: snapshot.configuration.link_states.get(link.link_id, "up") == "up" for link in snapshot.topology.links}
    return AgentObservation(snapshot.snapshot_id, global_state, {
        "ospf": {"ospf_weights": dict(snapshot.configuration.ospf_weights), "policy_consistency": snapshot.metrics.policy_consistency},
        "bgp": {"routes": [item.to_dict() for item in snapshot.configuration.bgp.routes], "policy_consistency": snapshot.metrics.policy_consistency},
        "performance": {"link_utilization": loads, "delivered_bps": snapshot.metrics.delivered_bps, "dropped_bps": snapshot.metrics.dropped_bps},
    }, {"ospf": link_mask, "performance": link_mask}, {"ospf": ("ospf_weights", "policy_consistency"), "bgp": ("routes", "policy_consistency"), "performance": ("link_utilization", "delivered_bps", "dropped_bps")})


def _reward(previous, current, changed, error_count, weights: RewardConfig):
    clip = lambda value: max(-weights.component_clip, min(weights.component_clip, value))
    policy = weights.policy_improvement * clip(current.metrics.policy_consistency - previous.metrics.policy_consistency)
    # MLU deltas can be orders of magnitude larger than the bounded policy
    # signal.  Clipping prevents congestion outliers from erasing that goal.
    mlu = weights.mlu_improvement * clip(previous.metrics.maximum_link_utilization - current.metrics.maximum_link_utilization)
    shift = -weights.traffic_shift_penalty * (current.metrics.traffic_shift_step_project_v1 or 0.0) if weights.include_traffic_shift else 0.0
    config_penalty = -weights.configuration_change_penalty * len(changed)
    illegal = -weights.illegal_action_penalty * error_count
    dropped = -weights.dropped_traffic_penalty * (current.metrics.dropped_bps / current.metrics.total_input_bps) if current.metrics.total_input_bps else 0.0
    total = policy + mlu + shift + config_penalty + illegal + dropped
    changed_names=tuple(str(name) for name in changed)
    own_changes={
        "ospf":sum(name.startswith("ospf_weight") for name in changed_names),
        "bgp":sum(name.startswith("bgp") for name in changed_names),
        "performance":sum(name.startswith("performance") for name in changed_names),
    }
    local={
        "ospf":policy+mlu+shift-weights.configuration_change_penalty*own_changes["ospf"]+illegal/3,
        "bgp":policy-weights.configuration_change_penalty*own_changes["bgp"]+illegal/3,
        "performance":mlu+shift+dropped-weights.configuration_change_penalty*own_changes["performance"]+illegal/3,
    }
    return RewardBreakdown(policy, mlu, shift, config_penalty, illegal, dropped, total, local)
