"""Multi-agent reinforcement-learning adapter for the deterministic simulator."""

from netkeeper_sim.rl.config import COMATrainingConfig, GraphNetworkConfig, RLConfig

__all__ = [
    "COMATrainingConfig",
    "GraphNetworkConfig",
    "RLConfig",
    "MultiAgentNetworkEnvironment",
    "UnifiedRLEnvironment",
]

def __getattr__(name: str):
    if name in {"MultiAgentNetworkEnvironment", "UnifiedRLEnvironment"}:
        from netkeeper_sim.rl.multi_agent_env import MultiAgentNetworkEnvironment, UnifiedRLEnvironment
        return {"MultiAgentNetworkEnvironment": MultiAgentNetworkEnvironment, "UnifiedRLEnvironment": UnifiedRLEnvironment}[name]
    raise AttributeError(name)
