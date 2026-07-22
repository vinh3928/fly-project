from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HomeostaticRewardConfig:
    curiosity_weight: float = 0.1
    energy_weight: float = 1.0
    hydration_weight: float = 1.0
    damage_weight: float = 2.0
    effort_weight: float = 0.05
    instability_weight: float = 0.5


def compute_total_reward(
    curiosity: float,
    energy_improvement: float,
    hydration_improvement: float,
    damage: float,
    motor_effort: float,
    instability: float,
    config: HomeostaticRewardConfig | None = None,
) -> float:
    """Combine curiosity with survival objectives.

    This is self-supervised representation learning + intrinsic RL + homeostatic
    objectives — not purely unsupervised learning.
    """
    cfg = config or HomeostaticRewardConfig()
    curiosity = max(0.0, min(1.0, curiosity))
    return (
        cfg.curiosity_weight * curiosity
        + cfg.energy_weight * energy_improvement
        + cfg.hydration_weight * hydration_improvement
        - cfg.damage_weight * damage
        - cfg.effort_weight * motor_effort
        - cfg.instability_weight * instability
    )
