"""Motor layer package (Phase 9 architecture B)."""

from .layer import (
    HeuristicMotorTracker,
    LearnedMotorTracker,
    MotorLayer,
    MotorLayerConfig,
    MotorPolicyNet,
    clip_goal,
    ensure_locomotion_floor,
    goal_from_intent,
)
from .rewards import compute_locomotion_reward

__all__ = [
    "HeuristicMotorTracker",
    "LearnedMotorTracker",
    "MotorLayer",
    "MotorLayerConfig",
    "MotorPolicyNet",
    "clip_goal",
    "compute_locomotion_reward",
    "ensure_locomotion_floor",
    "goal_from_intent",
]
