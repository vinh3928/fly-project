from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HomeostaticRewardConfig:
    curiosity_weight: float = 0.35
    energy_weight: float = 1.0
    hydration_weight: float = 1.0
    damage_weight: float = 2.0
    effort_weight: float = 0.05
    instability_weight: float = 0.5
    sugar_consume_weight: float = 2.0
    bitter_penalty_weight: float = 1.5
    hunger_penalty_weight: float = 0.3


def compute_total_reward(
    curiosity: float,
    energy_improvement: float,
    hydration_improvement: float,
    damage: float,
    motor_effort: float,
    instability: float,
    config: HomeostaticRewardConfig | None = None,
    *,
    sugar_consume: float = 0.0,
    bitter_contact: float = 0.0,
    hunger: float = 0.0,
) -> float:
    """Combine curiosity with survival / feeding objectives.

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
        + cfg.sugar_consume_weight * max(0.0, sugar_consume)
        - cfg.bitter_penalty_weight * max(0.0, bitter_contact)
        - cfg.hunger_penalty_weight * max(0.0, min(1.0, hunger))
    )
