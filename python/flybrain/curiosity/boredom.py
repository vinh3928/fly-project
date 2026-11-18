"""Stimulus-agnostic boredom: stuck thoughts and/or stuck body → force locomotion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def thought_fingerprint(plan_name: str | None, refined_thought: str | None) -> str:
    """Stable id for 'same thought' without referencing world objects."""
    plan = str(plan_name or "").strip().lower()
    text = " ".join(str(refined_thought or "").split()).lower()
    # Keep enough text to detect template sameness; drop tiny numeric jitter noise by
    # rounding embedded floats lightly via stripping digits after the decimal in long blobs.
    return f"{plan}|{text[:240]}"


@dataclass
class BoredomConfig:
    enabled: bool = True
    thought_weight: float = 0.55
    physical_weight: float = 0.55
    thought_horizon_ticks: int = 4
    physical_horizon_steps: int = 12
    motion_eps: float = 0.002
    action_eps: float = 0.08
    force_threshold: float = 0.55
    stop_penalty: float = 0.35
    locomotor_bonus: float = 0.22
    escape_forward: float = 0.45
    escape_turn: float = 0.55
    # Blocked: commanding forward into a wall while the body does not move.
    blocked_forward_min: float = 0.3
    blocked_speed_max: float = 0.01  # m/s, when the env reports body speed
    blocked_wall_min: float = 0.25  # wall_hazard must clear this or escape never arms
    blocked_horizon_steps: int = 200  # 1 s at 200 Hz physics
    escape_hold_steps: int = 160
    escape_backup: float = 0.5

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> BoredomConfig:
        data = dict(raw or {})
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class BoredomState:
    boredom: float
    thought_stuck: float
    physical_stuck: float
    thought_streak: int
    physical_streak: int
    force_locomotion: bool
    blocked_streak: int = 0
    escaping: bool = False

    @property
    def pushing_blocked(self) -> bool:
        """Commanding forward without moving right now, or backing out of a jam."""
        return self.blocked_streak > 0 or self.escaping

    def to_dict(self) -> dict[str, Any]:
        return {
            "boredom": float(self.boredom),
            "thought_stuck": float(self.thought_stuck),
            "physical_stuck": float(self.physical_stuck),
            "thought_streak": int(self.thought_streak),
            "physical_streak": int(self.physical_streak),
            "force_locomotion": bool(self.force_locomotion),
            "blocked_streak": int(self.blocked_streak),
            "escaping": bool(self.escaping),
        }


class BoredomTracker:
    """Accumulates thought-sameness and physical immobility into a boredom score."""

    def __init__(self, config: BoredomConfig | None = None) -> None:
        self.config = config or BoredomConfig()
        self._prev_thought: str | None = None
        self._thought_streak = 0
        self.reset()

    def reset(self) -> None:
        self._prev_thought = None
        self._thought_streak = 0
        self._prev_position = None
        self._physical_streak = 0
        self._blocked_streak = 0
        self._escape_remaining = 0
        self._escape_sign = 1.0
        self._last = BoredomState(
            boredom=0.0,
            thought_stuck=0.0,
            physical_stuck=0.0,
            thought_streak=0,
            physical_streak=0,
            force_locomotion=False,
        )

    @property
    def escape_sign(self) -> float:
        return self._escape_sign

    def state(self) -> BoredomState:
        return self._last

    def _recompute(self) -> BoredomState:
        cfg = self.config
        if not cfg.enabled:
            self._last = BoredomState(0.0, 0.0, 0.0, 0, 0, False)
            return self._last
        thought_stuck = _clamp01(
            float(self._thought_streak) / max(1, int(cfg.thought_horizon_ticks))
        )
        physical_stuck = _clamp01(
            max(
                float(self._physical_streak) / max(1, int(cfg.physical_horizon_steps)),
                float(self._blocked_streak) / max(1, int(cfg.blocked_horizon_steps)),
            )
        )
        # "Or" semantics: either channel can drive boredom high.
        boredom = _clamp01(
            max(
                float(cfg.thought_weight) * thought_stuck,
                float(cfg.physical_weight) * physical_stuck,
            )
            + 0.25 * min(thought_stuck, physical_stuck)
        )
        force = boredom >= float(cfg.force_threshold)
        self._last = BoredomState(
            boredom=boredom,
            thought_stuck=thought_stuck,
            physical_stuck=physical_stuck,
            thought_streak=int(self._thought_streak),
            physical_streak=int(self._physical_streak),
            force_locomotion=force,
            blocked_streak=int(self._blocked_streak),
            escaping=self._escape_remaining > 0,
        )
        return self._last

    def observe_thought(self, *, plan_name: str | None, refined_thought: str | None) -> BoredomState:
        """Call after a mind tick with the new thought/plan fingerprint."""
        if not self.config.enabled:
            return self._recompute()
        fp = thought_fingerprint(plan_name, refined_thought)
        if self._prev_thought is None:
            self._prev_thought = fp
            self._thought_streak = 0
        elif fp == self._prev_thought:
            self._thought_streak += 1
        else:
            self._prev_thought = fp
            self._thought_streak = 0
            self._escape_sign *= -1.0
        return self._recompute()

    def observe_physical(
        self,
        *,
        position: np.ndarray | None,
        action: np.ndarray | None,
        speed: float | None = None,
        wall_hazard: float | None = None,
    ) -> BoredomState:
        """Call every step.

        Idle-stuck: little motion and near-zero action. Blocked: commanding forward
        into a wall (``wall_hazard >= blocked_wall_min``) while the body stays put
        (position, else ``speed`` in m/s) for ``blocked_horizon_steps`` — then hold
        a back-up-and-turn escape. Open space (low/no wall) never arms escape.
        """
        if not self.config.enabled:
            return self._recompute()
        cfg = self.config
        act = np.asarray(action if action is not None else [0.0, 0.0], dtype=np.float32).reshape(-1)
        fwd = float(act[0]) if act.size else 0.0
        turn = float(act[1]) if act.size > 1 else 0.0
        action_mag = max(abs(fwd), abs(turn))
        wall = float(wall_hazard) if wall_hazard is not None else 0.0

        motion = 0.0
        # None = this env cannot tell whether the body moved.
        still: bool | None = None
        if position is not None:
            pos = np.asarray(position, dtype=np.float32).reshape(-1)
            if self._prev_position is not None and pos.size >= 2:
                prev = self._prev_position
                n = min(pos.size, prev.size, 3)
                motion = float(np.linalg.norm(pos[:n] - prev[:n]))
                still = motion < float(cfg.motion_eps)
            self._prev_position = pos.copy()
        else:
            # No pose: treat near-zero commanded action as physical stuck.
            motion = 1.0 if action_mag > float(cfg.action_eps) else 0.0
            if speed is not None:
                still = float(speed) < float(cfg.blocked_speed_max)

        if motion < float(cfg.motion_eps) and action_mag <= float(cfg.action_eps):
            self._physical_streak += 1
        else:
            self._physical_streak = 0

        if self._escape_remaining > 0:
            self._escape_remaining -= 1
        elif (
            still
            and fwd >= float(cfg.blocked_forward_min)
            and wall >= float(cfg.blocked_wall_min)
        ):
            self._blocked_streak += 1
            if self._blocked_streak >= int(cfg.blocked_horizon_steps):
                self._blocked_streak = 0
                self._escape_remaining = int(cfg.escape_hold_steps)
                # Alternate sides so a repeat jam tries the other way out.
                self._escape_sign *= -1.0
        else:
            self._blocked_streak = 0
        return self._recompute()


def apply_boredom_to_candidate_value(
    *,
    name: str,
    value: float,
    boredom: float,
    config: BoredomConfig | None = None,
) -> float:
    """Bias L1 scores: high boredom punishes stop and rewards locomotion."""
    cfg = config or BoredomConfig()
    if not cfg.enabled or boredom <= 1e-6:
        return float(value)
    b = _clamp01(boredom)
    v = float(value)
    if name == "stop":
        v -= float(cfg.stop_penalty) * b
    elif name in ("turn_left", "turn_right", "explore", "forward"):
        v += float(cfg.locomotor_bonus) * b
        if name in ("turn_left", "turn_right") and b >= float(cfg.force_threshold):
            v += 0.08 * b
    return float(v)


def apply_blocked_escape_goal(
    goal: np.ndarray,
    state: BoredomState,
    *,
    config: BoredomConfig | None = None,
    escape_sign: float = 1.0,
) -> np.ndarray:
    """While escaping a jam, override the goal with back up + turn (last word on the goal)."""
    cfg = config or BoredomConfig()
    g = np.asarray(goal, dtype=np.float32).reshape(-1).copy()
    if not cfg.enabled or not state.escaping:
        return g
    sign = 1.0 if escape_sign >= 0.0 else -1.0
    g[0] = -float(cfg.escape_backup)
    if g.size > 1:
        g[1] = float(sign * cfg.escape_turn)
    return g


def apply_boredom_escape_goal(
    goal: np.ndarray,
    state: BoredomState,
    *,
    config: BoredomConfig | None = None,
    escape_sign: float = 1.0,
) -> np.ndarray:
    """If boredom forces locomotion and goal is a near-stop, inject turn+crawl."""
    cfg = config or BoredomConfig()
    g = np.asarray(goal, dtype=np.float32).reshape(-1).copy()
    if g.size < 5:
        padded = np.zeros(5, dtype=np.float32)
        padded[: g.size] = g
        g = padded
    if not cfg.enabled or not state.force_locomotion:
        return g
    if max(abs(float(g[0])), abs(float(g[1]))) > float(cfg.action_eps):
        # Already moving; still ensure a bit of forward when force is on.
        if abs(float(g[0])) < float(cfg.escape_forward) * 0.5:
            g[0] = float(np.clip(max(float(g[0]), float(cfg.escape_forward) * 0.5), -1.0, 1.0))
        return g
    sign = 1.0 if escape_sign >= 0.0 else -1.0
    g[0] = float(cfg.escape_forward)
    g[1] = float(sign * cfg.escape_turn)
    g[3] = max(float(g[3]), 0.55)
    return g
