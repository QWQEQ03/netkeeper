from __future__ import annotations

import argparse
from pathlib import Path

from netkeeper_sim.policies import ForwardPolicy
from netkeeper_sim.rl.checkpoints import save_checkpoint
from netkeeper_sim.rl.config import COMATrainingConfig, RLConfig
from netkeeper_sim.rl.multi_agent_env import MultiAgentNetworkEnvironment
from netkeeper_sim.rl.trainer import COMATrainer
from netkeeper_sim.topology.model import build_topology_from_edges
from netkeeper_sim.traffic.matrix import TrafficDemand, TrafficMatrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small COMA debug training loop.")
    parser.add_argument("--config", default="configs/rl_debug.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/rl_debug.pt")
    args = parser.parse_args()

    training_config = COMATrainingConfig.from_yaml(args.config)
    rl_config = RLConfig(max_steps=training_config.max_steps)
    topology = build_topology_from_edges(
        [
            ("R1", "R2", 1),
            ("R2", "R4", 1),
            ("R1", "R3", 1),
            ("R3", "R4", 1),
        ]
    )
    env = MultiAgentNetworkEnvironment(
        topology=topology,
        traffic_matrix=TrafficMatrix((TrafficDemand("R1", "R4", 100.0),)),
        policies=[ForwardPolicy("must-use-r3", "R1", "R4", "R3")],
        config=rl_config,
    )
    trainer = COMATrainer(env, training_config)
    logs = trainer.train()
    for entry in logs:
        print(
            "episode={episode} step={step} epsilon={epsilon:.4f} "
            "policy={policy_consistency:.3f} mlu={maximum_link_utilization:.3f} "
            "shift={traffic_shift:.3f} rewards=({ospf_reward:.3f},"
            "{bgp_reward:.3f},{performance_reward:.3f}) "
            "loss=({actor_loss},{critic_loss}) grad={gradient_norm} "
            "terminated={terminated} truncated={truncated}".format(
                **entry.__dict__,
            )
        )
    save_checkpoint(
        Path(args.checkpoint),
        trainer.actor,
        trainer.critic,
        trainer.target_critic,
        trainer.actor_optimizer,
        trainer.critic_optimizer,
        metadata={"config": args.config, "steps": len(logs)},
    )


if __name__ == "__main__":
    main()
