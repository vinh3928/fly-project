"""Mind-pushed motor act states: act_anything / still / idle.

Architecture B: the mind pushes a :class:`MotorDirective` when a think result
completes (or when stimulus interrupt requests deep-thought idle). Between
pushes the motor only tracks the held goal — no per-tick replan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from flybrain.motor.layer import clip_goal
from flybrain.thinking.stimulus_interrupt import STOP_GOAL

if TYPE_CHECKING:
    from flybrain.motor.layer import MotorLayer

ActMode = Literal["act_anything", "still", "idle"]

STILL_PLAN_NAMES = frozenset({"", "stop", "still"})


def mode_for_plan(plan_name: str | None) -> ActMode:
    """Map an L1 plan name to an act mode (idle is mind/SI only, not an L1 plan)."""
    name = str(plan_name or "").strip().lower()
    if name == "idle":
        return "idle"
    if name in STILL_PLAN_NAMES:
        return "still"
    return "act_anything"


@dataclass
class MotorDirective:
    """Immutable-ish snapshot the mind pushes into the motor controller."""

    mode: ActMode
    plan_name: str
    goal: np.ndarray
    reason: str = ""
    step: int = 0
    # Skip the motor's forward floor (turn in place / back up out of a jam).
    allow_pivot: bool = False

    def __post_init__(self) -> None:
        g = clip_goal(self.goal).copy()
        if self.mode in ("still", "idle"):
            g = STOP_GOAL.copy()
        self.goal = g
        self.plan_name = str(self.plan_name or self.mode)


def make_directive(
    *,
    mode: ActMode | None = None,
    plan_name: str | None = None,
    goal: np.ndarray | None = None,
    reason: str = "",
    step: int = 0,
    allow_pivot: bool = False,
) -> MotorDirective:
    """Build a directive from mode and/or plan name + goal."""
    resolved_mode = mode if mode is not None else mode_for_plan(plan_name)
    name = str(plan_name or resolved_mode)
    if resolved_mode in ("still", "idle"):
        g = STOP_GOAL.copy()
    elif goal is not None:
        g = clip_goal(goal)
    else:
        g = STOP_GOAL.copy()
    return MotorDirective(
        mode=resolved_mode,
        plan_name=name,
        goal=g,
        reason=str(reason or ""),
        step=int(step),
        allow_pivot=bool(allow_pivot),
    )


@dataclass
class MotorActController:
    """Act-state machine above :class:`MotorLayer`; intake is push-only."""

    motor: MotorLayer
    _directive: MotorDirective = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._directive = make_directive(
            mode="still",
            plan_name="stop",
            reason="init",
            step=0,
        )
        self.motor.set_goal(self._directive.goal.copy())
        self.motor.reset(self._directive.goal.copy())

    @property
    def current(self) -> MotorDirective:
        return self._directive

    @property
    def mode(self) -> ActMode:
        return self._directive.mode

    def push(self, directive: MotorDirective) -> MotorDirective:
        """Mind interrupt intake: replace held mode/goal."""
        self._directive = directive
        self.motor.set_goal(directive.goal.copy(), allow_pivot=directive.allow_pivot)
        return self._directive

    def apply_tick(self, observation: np.ndarray | None = None) -> np.ndarray:
        """Emit one motor action for the held directive (no replan)."""
        if self._directive.mode in ("still", "idle"):
            self.motor.set_goal(STOP_GOAL.copy())
        return self.motor.act(observation)
