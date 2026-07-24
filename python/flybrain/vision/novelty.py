"""Eye-driven visual novelty: new stimuli look interesting and pull explore motor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


@dataclass
class VisualNoveltyConfig:
    enabled: bool = True
    n_columns: int = 8
    signature_bins: int = 6
    first_visit_bonus: float = 1.0
    repeat_decay: float = 0.7
    change_weight: float = 0.45
    novelty_weight: float = 0.55
    # Motor pull when visual novelty is high.
    turn_gain: float = 0.85
    forward_gain: float = 0.55
    min_novelty_for_pull: float = 0.18
    # Soft L1 explore boost.
    explore_candidate_bonus: float = 0.28
    # Habituation: staring at the same view (e.g. a wall) slowly loses novelty;
    # a genuinely changing scene recovers it (dishabituation).
    habituation_enabled: bool = True
    static_change_threshold: float = 0.08  # scene_change below this ⇒ "same view"
    habituation_rate: float = 0.02  # slow rise toward full habituation when static
    dishabituation_rate: float = 0.5  # fast recovery when the scene changes
    habituation_strength: float = 0.9  # max fraction of novelty suppressed

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> VisualNoveltyConfig:
        data = dict(raw or {})
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class VisualNoveltyCues:
    visual_novelty: float
    novelty_bearing: float
    novelty_confidence: float
    new_stimulus: bool
    unique_patches: int

    def to_dict(self) -> dict[str, float | bool | int]:
        return {
            "visual_novelty": float(self.visual_novelty),
            "novelty_bearing": float(self.novelty_bearing),
            "novelty_confidence": float(self.novelty_confidence),
            "new_stimulus": bool(self.new_stimulus),
            "unique_visual_patches": int(self.unique_patches),
        }


@dataclass
class VisualNoveltyTracker:
    """Tracks coarse eye-patch signatures; novel patches bias explore toward their bearing."""

    config: VisualNoveltyConfig = field(default_factory=VisualNoveltyConfig)
    _visits: dict[tuple[int, int], int] = field(default_factory=dict)
    _prev_frame: np.ndarray | None = None
    _habituation: float = 0.0

    def reset(self) -> None:
        self._visits.clear()
        self._prev_frame = None
        self._habituation = 0.0

    def _frame(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        l = np.asarray(left, dtype=np.float32)
        r = np.asarray(right, dtype=np.float32)
        while l.ndim > 2:
            l = l[0]
        while r.ndim > 2:
            r = r[0]
        return 0.5 * (l + r)

    def _column_signature(self, col: np.ndarray) -> int:
        mean = float(col.mean())
        std = float(col.std())
        # Simple edge energy
        if col.size > 1:
            edge = float(np.mean(np.abs(np.diff(col, axis=0))))
        else:
            edge = 0.0
        bins = max(2, int(self.config.signature_bins))
        m_i = int(np.clip(mean * bins, 0, bins - 1))
        s_i = int(np.clip(std * bins * 2.0, 0, bins - 1))
        e_i = int(np.clip(edge * bins * 3.0, 0, bins - 1))
        return m_i * bins * bins + s_i * bins + e_i

    def observe(self, left: np.ndarray, right: np.ndarray) -> VisualNoveltyCues:
        cfg = self.config
        if not cfg.enabled:
            return VisualNoveltyCues(0.0, 0.0, 0.0, False, 0)
        frame = self._frame(left, right)
        h, w = frame.shape
        n_cols = max(2, int(cfg.n_columns))
        col_w = max(1, w // n_cols)
        xs = np.linspace(-1.0, 1.0, n_cols, dtype=np.float32)

        novelty_weights = np.zeros(n_cols, dtype=np.float32)
        change_weights = np.zeros(n_cols, dtype=np.float32)
        new_any = False
        had_prev = self._prev_frame is not None and self._prev_frame.shape == frame.shape

        for i in range(n_cols):
            x0 = i * col_w
            x1 = w if i == n_cols - 1 else min(w, (i + 1) * col_w)
            col = frame[:, x0:x1]
            sig = self._column_signature(col)
            key = (i, sig)
            before = self._visits.get(key, 0)
            bonus = float(cfg.first_visit_bonus) * (float(cfg.repeat_decay) ** float(before))
            self._visits[key] = before + 1
            novelty_weights[i] = bonus
            if before == 0:
                new_any = True

            if self._prev_frame is not None and self._prev_frame.shape == frame.shape:
                prev_col = self._prev_frame[:, x0:x1]
                delta = float(np.mean(np.abs(col - prev_col)))
                change_weights[i] = float(np.clip(delta / 0.15, 0.0, 1.0))

        self._prev_frame = frame.copy()

        # Prefer columns that are both novel and changing (new object entering view).
        interest = (
            float(cfg.novelty_weight) * novelty_weights
            + float(cfg.change_weight) * change_weights * float(cfg.first_visit_bonus)
        )
        total = float(interest.sum()) + 1e-6
        bearing = float((interest * xs).sum() / total)
        peak = float(interest.max())
        novelty = _clamp01(0.55 * peak + 0.35 * (total / max(1.0, n_cols)) + (0.25 if new_any else 0.0))
        confidence = _clamp01(peak)

        # Temporal habituation: a static view (same wall) slowly loses novelty;
        # a genuinely changing scene recovers it. Applied as a soft suppressor so
        # the per-signature decay above stays the fast, per-patch component.
        if cfg.habituation_enabled:
            scene_change = float(change_weights.mean()) if had_prev else 1.0
            if scene_change <= float(cfg.static_change_threshold):
                self._habituation += float(cfg.habituation_rate) * (1.0 - self._habituation)
            else:
                self._habituation -= float(cfg.dishabituation_rate) * scene_change
            self._habituation = _clamp01(self._habituation)
            suppress = 1.0 - float(cfg.habituation_strength) * self._habituation
            novelty *= suppress
            confidence *= suppress

        return VisualNoveltyCues(
            visual_novelty=novelty,
            novelty_bearing=float(np.clip(bearing, -1.0, 1.0)),
            novelty_confidence=confidence,
            new_stimulus=bool(new_any and novelty >= float(cfg.min_novelty_for_pull)),
            unique_patches=len(self._visits),
        )


def apply_visual_explore_goal(
    goal: np.ndarray,
    cues: VisualNoveltyCues | dict[str, Any] | None,
    *,
    config: VisualNoveltyConfig | None = None,
    wall_hazard: float = 0.0,
) -> np.ndarray:
    """Pull motor goal toward novel visual bearing when stimuli look new."""
    cfg = config or VisualNoveltyConfig()
    g = np.asarray(goal, dtype=np.float32).reshape(-1).copy()
    if g.size < 5:
        padded = np.zeros(5, dtype=np.float32)
        padded[: g.size] = g
        g = padded
    if not cfg.enabled or cues is None:
        return g
    if isinstance(cues, VisualNoveltyCues):
        nov = float(cues.visual_novelty)
        bearing = float(cues.novelty_bearing)
        conf = float(cues.novelty_confidence)
    else:
        nov = float(cues.get("visual_novelty", 0.0))
        bearing = float(cues.get("novelty_bearing", 0.0))
        conf = float(cues.get("novelty_confidence", 0.0))
    if nov < float(cfg.min_novelty_for_pull):
        return g
    strength = _clamp01(nov * (0.5 + 0.5 * conf))
    # Turn toward the novel patch.
    g[1] = float(
        np.clip(
            (1.0 - 0.65 * strength) * float(g[1]) + 0.65 * strength * bearing * float(cfg.turn_gain),
            -1.0,
            1.0,
        )
    )
    # Walk toward it unless wall is extreme.
    forward_pull = float(cfg.forward_gain) * strength
    if wall_hazard > 0.7:
        forward_pull *= 0.35
    elif wall_hazard > 0.45:
        forward_pull *= 0.65
    g[0] = float(np.clip(max(float(g[0]), forward_pull), -1.0, 1.0))
    g[3] = float(np.clip(max(float(g[3]), 0.45 + 0.35 * strength), 0.0, 1.0))
    return g
