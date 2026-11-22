"""Playback sequencer for skill-cortex motor programs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from flybrain.memory.skill_cortex import SkillProgram, SkillStep
from flybrain.motor.layer import clip_goal


@dataclass
class SkillPlaybackFrame:
    """One tick of skill playback ready for ``_push_motor_directive``."""

    active: bool
    finished: bool
    plan_name: str
    goal: np.ndarray
    step_index: int
    skill_id: str
    skill_name: str
    reason: str = "skill_playback"


@dataclass
class SkillExecutor:
    """Advance timed skill steps; preempt clears the active program."""

    program: SkillProgram | None = None
    step_index: int = 0
    ticks_in_step: int = 0
    preempted: bool = False
    _motion_samples: list[float] = field(default_factory=list)
    _novelty_samples: list[float] = field(default_factory=list)
    _damage_spike: bool = False

    @property
    def active(self) -> bool:
        return self.program is not None and not self.preempted and not self.finished

    @property
    def finished(self) -> bool:
        if self.program is None:
            return True
        if self.preempted:
            return True
        return self.step_index >= len(self.program.steps)

    def start(self, program: SkillProgram) -> None:
        self.program = program
        self.step_index = 0
        self.ticks_in_step = 0
        self.preempted = False
        self._motion_samples = []
        self._novelty_samples = []
        self._damage_spike = False

    def preempt(self, reason: str = "preempt") -> None:
        _ = reason
        if self.program is not None:
            self.preempted = True

    def clear(self) -> None:
        self.program = None
        self.step_index = 0
        self.ticks_in_step = 0
        self.preempted = False
        self._motion_samples = []
        self._novelty_samples = []
        self._damage_spike = False

    def _current_step(self) -> SkillStep | None:
        if self.program is None:
            return None
        if self.step_index < 0 or self.step_index >= len(self.program.steps):
            return None
        return self.program.steps[self.step_index]

    def observe(self, *, motion: float = 0.0, novelty: float = 0.0, damage: float = 0.0) -> None:
        if not self.active:
            return
        self._motion_samples.append(float(motion))
        self._novelty_samples.append(float(novelty))
        if float(damage) >= 0.35:
            self._damage_spike = True

    def episode_stats(self) -> dict[str, Any]:
        motion = np.asarray(self._motion_samples, dtype=np.float32)
        novelty = np.asarray(self._novelty_samples, dtype=np.float32)
        return {
            "motion_variance": float(np.var(motion)) if motion.size else 0.0,
            "novelty_during": float(np.mean(novelty)) if novelty.size else 0.0,
            "damage_spike": bool(self._damage_spike),
            "completed": bool(self.finished and not self.preempted and self.program is not None),
            "skill_id": self.program.skill_id if self.program else "",
            "skill_name": self.program.name if self.program else "",
        }

    def tick(self) -> SkillPlaybackFrame:
        """Advance one motor/mind tick; return directive fields."""
        if self.program is None or self.preempted:
            return SkillPlaybackFrame(
                active=False,
                finished=True,
                plan_name="idle",
                goal=clip_goal(np.zeros(5, dtype=np.float32)),
                step_index=self.step_index,
                skill_id="",
                skill_name="",
                reason="skill_idle",
            )
        step = self._current_step()
        if step is None:
            return SkillPlaybackFrame(
                active=False,
                finished=True,
                plan_name=f"skill_{self.program.name}",
                goal=clip_goal(np.zeros(5, dtype=np.float32)),
                step_index=self.step_index,
                skill_id=self.program.skill_id,
                skill_name=self.program.name,
                reason="skill_done",
            )
        goal = np.asarray(step.goal, dtype=np.float32).copy()
        plan = f"skill_{self.program.name}"
        frame = SkillPlaybackFrame(
            active=True,
            finished=False,
            plan_name=plan,
            goal=goal,
            step_index=self.step_index,
            skill_id=self.program.skill_id,
            skill_name=self.program.name,
            reason="skill_playback",
        )
        self.ticks_in_step += 1
        if self.ticks_in_step >= int(step.duration_steps):
            self.step_index += 1
            self.ticks_in_step = 0
            if self.step_index >= len(self.program.steps):
                frame = SkillPlaybackFrame(
                    active=False,
                    finished=True,
                    plan_name=plan,
                    goal=goal,
                    step_index=self.step_index,
                    skill_id=self.program.skill_id,
                    skill_name=self.program.name,
                    reason="skill_done",
                )
        return frame
