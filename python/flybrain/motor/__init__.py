"""Motor layer package (Phase 9 architecture B)."""

from .act_state import (
    ActMode,
    MotorActController,
    MotorDirective,
    make_directive,
    mode_for_plan,
)
from .layer import (
    HeuristicMotorTracker,
    LearnedMotorTracker,
    MotorLayer,
    MotorLayerConfig,
    MotorPolicyNet,
    clip_goal,
    ensure_locomotion_floor,
    goal_from_intent,
    orient_goal,
)
from .rewards import compute_locomotion_reward

__all__ = [
    "ActMode",
    "HeuristicMotorTracker",
    "LearnedMotorTracker",
    "MotorActController",
    "MotorDirective",
    "MotorLayer",
    "MotorLayerConfig",
    "MotorPolicyNet",
    "clip_goal",
    "compute_locomotion_reward",
    "ensure_locomotion_floor",
    "goal_from_intent",
    "make_directive",
    "mode_for_plan",
    "orient_goal",
]
