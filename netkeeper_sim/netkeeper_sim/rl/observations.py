from __future__ import annotations

from dataclasses import dataclass

from netkeeper_sim.metrics.evaluation import EvaluationResult
from netkeeper_sim.policies.evaluator import PolicyConsistencyResult
from netkeeper_sim.rl import _tensor as T


@dataclass(frozen=True)
class AgentObservations:
    ospf: T.Tensor
    bgp: T.Tensor
    performance: T.Tensor
    policy_satisfaction: T.Tensor
    link_utilization: T.Tensor

    def as_dict(self) -> dict[str, T.Tensor]:
        return {
            "ospf": self.ospf,
            "bgp": self.bgp,
            "performance": self.performance,
        }


def policy_satisfaction_vector(result: PolicyConsistencyResult) -> T.Tensor:
    values = [1.0 if item.satisfied else 0.0 for item in result.results]
    return T.tensor(values, "float32")


def link_utilization_vector(
    evaluation: EvaluationResult,
    link_ids: tuple[str, ...],
) -> T.Tensor:
    values = [float(evaluation.link_utilizations.get(link_id, 0.0)) for link_id in link_ids]
    return T.tensor(values, "float32")


def build_agent_observations(
    policy_observation: T.Tensor,
    utilization_observation: T.Tensor,
) -> AgentObservations:
    ospf = T.tensor(
        [*T.to_numpy(policy_observation).reshape(-1), *T.to_numpy(utilization_observation).reshape(-1)],
        "float32",
    )
    bgp = T.tensor(T.to_numpy(policy_observation).reshape(-1), "float32")
    performance = T.tensor(T.to_numpy(utilization_observation).reshape(-1), "float32")
    return AgentObservations(
        ospf=ospf,
        bgp=bgp,
        performance=performance,
        policy_satisfaction=policy_observation,
        link_utilization=utilization_observation,
    )
