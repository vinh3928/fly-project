"""Locomotion / goal-tracking reward (Phase 9)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flybrain.phase9 import (
    ACTION_EFFORT_WEIGHT,
    ALIVE_BONUS,
    FALL_PENALTY,
    FORWARD_PROGRESS_WEIGHT,
    GOAL_TRACKING_WEIGHT,
    UPRIGHT_WEIGHT,
)


@dataclass
class LocomotionRewardConfig:
    forward_progress_weight: float = FORWARD_PROGRESS_WEIGHT
    goal_tracking_weight: float = GOAL_TRACKING_WEIGHT
    upright_weight: float = UPRIGHT_WEIGHT
    fall_penalty: float = FALL_PENALTY
    effort_weight: float = ACTION_EFFORT_WEIGHT
    alive_bonus: float = ALIVE_BONUS


def compute_locomotion_reward(
    *,
    goal: np.ndarray,
    action: np.ndarray,
    forward_progress: float,
    yaw_progress: float = 0.0,
    upright: float = 1.0,
    fell: bool = False,
    config: LocomotionRewardConfig | None = None,
) -> float:
    """Reward motor for tracking mind goals and making requested progress."""
    cfg = config or LocomotionRewardConfig()
    g = np.asarray(goal, dtype=np.float32).reshape(-1)
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    n = min(g.size, a.size)
    tracking_err = float(np.mean(np.abs(g[:n] - a[:n]))) if n else 0.0
    # Requested motion: weight progress by |goal_forward| / |goal_turn|.
    g_fwd = float(g[0]) if g.size else 0.0
    g_turn = float(g[1]) if g.size > 1 else 0.0
    progress = float(forward_progress) * abs(g_fwd) + float(yaw_progress) * abs(g_turn)
    upright = float(np.clip(upright, 0.0, 1.0))
    effort = float(np.mean(np.abs(a))) if a.size else 0.0
    reward = (
        cfg.forward_progress_weight * progress
        - cfg.goal_tracking_weight * tracking_err
        + cfg.upright_weight * upright
        - cfg.effort_weight * effort
        + cfg.alive_bonus
    )
    if fell:
        reward -= cfg.fall_penalty
    return float(reward)
