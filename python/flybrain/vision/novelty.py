"""Eye-driven visual novelty: independent 3×3 patches per eye (no stereo merge)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from flybrain.phase4 import PATCH_CELL_FEAT_DIM


def _clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _as_chw(eye: np.ndarray) -> np.ndarray:
    """Normalize an eye tensor to (C, H, W) with C in {1, 3}."""
    x = np.asarray(eye, dtype=np.float32)
    while x.ndim > 3:
        x = x[0]
    if x.ndim == 2:
        return x[None, ...]
    if x.ndim == 3 and x.shape[0] in (1, 3):
        return x
    if x.ndim == 3 and x.shape[-1] in (1, 3):
        return np.moveaxis(x[..., : x.shape[-1]], -1, 0)
    return np.asarray(x, dtype=np.float32).reshape(-1, x.shape[-2], x.shape[-1])[:1]


PATCH_FEAT_DIM = PATCH_CELL_FEAT_DIM  # R/G/B (or gray×3), luma std, vertical edge


def patch_feature_vector(patch: np.ndarray) -> np.ndarray:
    """Continuous embedding for one square (C,H,W) — used by interrupt history cache."""
    p = np.asarray(patch, dtype=np.float32)
    if p.ndim == 2:
        p = p[None, ...]
    if p.shape[0] >= 3:
        r_m = float(p[0].mean())
        g_m = float(p[1].mean())
        b_m = float(p[2].mean())
        gray = (p[0] + p[1] + p[2]) / 3.0
    else:
        gray = p[0]
        m = float(gray.mean())
        r_m = g_m = b_m = m
    std = float(gray.std()) if gray.size else 0.0
    if gray.size > 1:
        edge = float(np.mean(np.abs(np.diff(gray, axis=0))))
    else:
        edge = 0.0
    return np.asarray([r_m, g_m, b_m, std, edge], dtype=np.float32)


def eye_patch_feature_grid(
    eye: np.ndarray,
    *,
    grid_rows: int = 3,
    grid_cols: int = 3,
) -> np.ndarray:
    """Return (grid_rows * grid_cols, PATCH_FEAT_DIM) features for one eye."""
    frame = _as_chw(eye)
    _c, h, w = frame.shape
    n_rows = max(1, int(grid_rows))
    n_cols = max(1, int(grid_cols))
    row_h = max(1, h // n_rows)
    col_w = max(1, w // n_cols)
    feats: list[np.ndarray] = []
    for r in range(n_rows):
        y0 = r * row_h
        y1 = h if r == n_rows - 1 else min(h, (r + 1) * row_h)
        for col in range(n_cols):
            x0 = col * col_w
            x1 = w if col == n_cols - 1 else min(w, (col + 1) * col_w)
            feats.append(patch_feature_vector(frame[:, y0:y1, x0:x1]))
    return np.stack(feats, axis=0)


def stereo_patch_embedding(
    left: np.ndarray,
    right: np.ndarray,
    *,
    grid_rows: int = 3,
    grid_cols: int = 3,
) -> np.ndarray:
    """Flatten left+right 3×3 patch features into one STM vector (90-d for 3×3)."""
    left_g = eye_patch_feature_grid(left, grid_rows=grid_rows, grid_cols=grid_cols)
    right_g = eye_patch_feature_grid(right, grid_rows=grid_rows, grid_cols=grid_cols)
    return np.concatenate([left_g.reshape(-1), right_g.reshape(-1)]).astype(np.float32)


def stereo_patch_cells(
    left: np.ndarray,
    right: np.ndarray,
    *,
    grid_rows: int = 3,
    grid_cols: int = 3,
) -> dict[tuple[str, int, int], np.ndarray]:
    """Map (eye, row, col) → continuous patch feature vector."""
    out: dict[tuple[str, int, int], np.ndarray] = {}
    for eye_name, frame in (("left", left), ("right", right)):
        grid = eye_patch_feature_grid(frame, grid_rows=grid_rows, grid_cols=grid_cols)
        n_rows = max(1, int(grid_rows))
        n_cols = max(1, int(grid_cols))
        i = 0
        for r in range(n_rows):
            for c in range(n_cols):
                out[(eye_name, r, c)] = grid[i]
                i += 1
    return out


@dataclass
class VisualNoveltyConfig:
    enabled: bool = True
    # Independent square grid per eye (not stereo-averaged columns).
    grid_rows: int = 3
    grid_cols: int = 3
    signature_bins: int = 6
    first_visit_bonus: float = 1.0
    repeat_decay: float = 0.7
    change_weight: float = 0.45
    novelty_weight: float = 0.55
    # Motor pull when peak-patch visual novelty is high.
    turn_gain: float = 0.85
    forward_gain: float = 0.55
    min_novelty_for_pull: float = 0.18
    # Soft L1 explore boost.
    explore_candidate_bonus: float = 0.28
    # Per-patch habituation: a static square loses novelty; change recovers it.
    habituation_enabled: bool = True
    static_change_threshold: float = 0.08  # patch change below this ⇒ "same view"
    habituation_rate: float = 0.02  # slow rise toward full habituation when static
    dishabituation_rate: float = 0.5  # fast recovery when the patch changes
    habituation_strength: float = 0.9  # max fraction of novelty suppressed
    # Slight left/right FOV offset so each eye's azimuth stays distinct.
    eye_bearing_offset: float = 0.2

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> VisualNoveltyConfig:
        data = dict(raw or {})
        # Legacy n_columns → square grid (ignore non-square intents).
        if "grid_rows" not in data and "grid_cols" not in data and "n_columns" in data:
            n = int(data.pop("n_columns") or 3)
            side = 3 if n >= 3 else max(1, n)
            data["grid_rows"] = side
            data["grid_cols"] = side
        else:
            data.pop("n_columns", None)
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class PatchNovelty:
    """Novelty for one independent square on one eye."""

    eye: str  # "left" | "right"
    row: int
    col: int
    visual_novelty: float
    novelty_bearing: float
    novelty_confidence: float
    new_stimulus: bool
    change: float = 0.0

    def to_dict(self) -> dict[str, float | bool | int | str]:
        return {
            "eye": self.eye,
            "row": int(self.row),
            "col": int(self.col),
            "visual_novelty": float(self.visual_novelty),
            "novelty_bearing": float(self.novelty_bearing),
            "novelty_confidence": float(self.novelty_confidence),
            "new_stimulus": bool(self.new_stimulus),
            "change": float(self.change),
        }


@dataclass
class VisualNoveltyCues:
    """Per-patch novelty for left/right 3×3 grids (patches are not fused).

    Scalar ``visual_novelty`` / ``novelty_bearing`` are conveniences for gates and
    motor: they select the **peak** patch (argmax), never a weighted blend of eyes
    or squares.
    """

    patches: list[PatchNovelty] = field(default_factory=list)

    @property
    def peak(self) -> PatchNovelty | None:
        if not self.patches:
            return None
        return max(self.patches, key=lambda p: float(p.visual_novelty))

    @property
    def visual_novelty(self) -> float:
        peak = self.peak
        return float(peak.visual_novelty) if peak is not None else 0.0

    @property
    def novelty_bearing(self) -> float:
        peak = self.peak
        return float(peak.novelty_bearing) if peak is not None else 0.0

    @property
    def novelty_confidence(self) -> float:
        peak = self.peak
        return float(peak.novelty_confidence) if peak is not None else 0.0

    @property
    def new_stimulus(self) -> bool:
        return any(bool(p.new_stimulus) for p in self.patches)

    @property
    def unique_patches(self) -> int:
        return sum(1 for p in self.patches if p.new_stimulus or p.visual_novelty > 0.0)

    def to_dict(self) -> dict[str, Any]:
        peak = self.peak
        return {
            "visual_novelty": float(self.visual_novelty),
            "novelty_bearing": float(self.novelty_bearing),
            "novelty_confidence": float(self.novelty_confidence),
            "new_stimulus": bool(self.new_stimulus),
            "unique_visual_patches": int(self.unique_patches),
            "peak_eye": None if peak is None else peak.eye,
            "peak_row": None if peak is None else int(peak.row),
            "peak_col": None if peak is None else int(peak.col),
            "patch_novelties": [p.to_dict() for p in self.patches],
        }


@dataclass
class VisualNoveltyTracker:
    """Tracks independent 3×3 eye-patch signatures (left and right never merged)."""

    config: VisualNoveltyConfig = field(default_factory=VisualNoveltyConfig)
    _visits: dict[tuple[str, int, int, int], int] = field(default_factory=dict)
    _prev: dict[str, np.ndarray] = field(default_factory=dict)
    _habituation: dict[tuple[str, int, int], float] = field(default_factory=dict)

    def reset(self) -> None:
        self._visits.clear()
        self._prev.clear()
        self._habituation.clear()

    def _as_chw(self, eye: np.ndarray) -> np.ndarray:
        return _as_chw(eye)

    def _patch_signature(self, patch: np.ndarray) -> int:
        """Quantize a square. ``patch`` is (C, H, W) color or gray."""
        bins = max(2, int(self.config.signature_bins))
        if patch.ndim == 3 and patch.shape[0] >= 3:
            r_m = float(patch[0].mean())
            g_m = float(patch[1].mean())
            b_m = float(patch[2].mean())
            gray = (patch[0] + patch[1] + patch[2]) / 3.0
            std = float(gray.std())
            if gray.size > 1:
                edge = float(np.mean(np.abs(np.diff(gray, axis=0))))
            else:
                edge = 0.0
            r_i = int(np.clip(r_m * bins, 0, bins - 1))
            g_i = int(np.clip(g_m * bins, 0, bins - 1))
            b_i = int(np.clip(b_m * bins, 0, bins - 1))
            s_i = int(np.clip(std * bins * 2.0, 0, bins - 1))
            e_i = int(np.clip(edge * bins * 3.0, 0, bins - 1))
            return (((r_i * bins + g_i) * bins + b_i) * bins + s_i) * bins + e_i

        gray = patch[0] if patch.ndim == 3 else patch
        mean = float(gray.mean())
        std = float(gray.std())
        if gray.size > 1:
            edge = float(np.mean(np.abs(np.diff(gray, axis=0))))
        else:
            edge = 0.0
        m_i = int(np.clip(mean * bins, 0, bins - 1))
        s_i = int(np.clip(std * bins * 2.0, 0, bins - 1))
        e_i = int(np.clip(edge * bins * 3.0, 0, bins - 1))
        return m_i * bins * bins + s_i * bins + e_i

    def _bearing_for(self, eye: str, row: int, col: int, n_rows: int, n_cols: int) -> float:
        """Azimuth in [-1, 1] from patch column (+ optional eye offset). Row unused for yaw."""
        del row, n_rows  # elevation not wired to motor yet
        if n_cols <= 1:
            x = 0.0
        else:
            x = (float(col) / float(n_cols - 1)) * 2.0 - 1.0
        offset = float(self.config.eye_bearing_offset)
        if eye == "left":
            x = x - offset
        elif eye == "right":
            x = x + offset
        return float(np.clip(x, -1.0, 1.0))

    def _observe_eye(self, eye: str, frame: np.ndarray) -> list[PatchNovelty]:
        cfg = self.config
        c, h, w = frame.shape
        n_rows = max(1, int(cfg.grid_rows))
        n_cols = max(1, int(cfg.grid_cols))
        row_h = max(1, h // n_rows)
        col_w = max(1, w // n_cols)
        prev = self._prev.get(eye)
        had_prev = prev is not None and prev.shape == frame.shape
        out: list[PatchNovelty] = []

        for r in range(n_rows):
            y0 = r * row_h
            y1 = h if r == n_rows - 1 else min(h, (r + 1) * row_h)
            for col in range(n_cols):
                x0 = col * col_w
                x1 = w if col == n_cols - 1 else min(w, (col + 1) * col_w)
                patch = frame[:, y0:y1, x0:x1]
                sig = self._patch_signature(patch)
                key = (eye, r, col, sig)
                before = self._visits.get(key, 0)
                bonus = float(cfg.first_visit_bonus) * (float(cfg.repeat_decay) ** float(before))
                self._visits[key] = before + 1
                new_here = before == 0

                change = 0.0
                if had_prev and prev is not None:
                    prev_patch = prev[:, y0:y1, x0:x1]
                    delta = float(
                        np.mean(np.abs(patch.astype(np.float32) - prev_patch.astype(np.float32)))
                    )
                    change = float(np.clip(delta / 0.15, 0.0, 1.0))

                interest = (
                    float(cfg.novelty_weight) * bonus
                    + float(cfg.change_weight) * change * float(cfg.first_visit_bonus)
                )
                novelty = _clamp01(
                    0.55 * interest
                    + 0.35 * bonus
                    + (0.25 if new_here else 0.0)
                )
                confidence = _clamp01(interest)

                hab_key = (eye, r, col)
                if cfg.habituation_enabled:
                    hab = float(self._habituation.get(hab_key, 0.0))
                    scene_change = change if had_prev else 1.0
                    if scene_change <= float(cfg.static_change_threshold):
                        hab += float(cfg.habituation_rate) * (1.0 - hab)
                    else:
                        hab -= float(cfg.dishabituation_rate) * scene_change
                    hab = _clamp01(hab)
                    self._habituation[hab_key] = hab
                    suppress = 1.0 - float(cfg.habituation_strength) * hab
                    novelty *= suppress
                    confidence *= suppress

                bearing = self._bearing_for(eye, r, col, n_rows, n_cols)
                out.append(
                    PatchNovelty(
                        eye=eye,
                        row=r,
                        col=col,
                        visual_novelty=float(novelty),
                        novelty_bearing=bearing,
                        novelty_confidence=float(confidence),
                        new_stimulus=bool(
                            new_here and novelty >= float(cfg.min_novelty_for_pull)
                        ),
                        change=float(change),
                    )
                )

        self._prev[eye] = frame.copy()
        return out

    def observe(self, left: np.ndarray, right: np.ndarray) -> VisualNoveltyCues:
        cfg = self.config
        if not cfg.enabled:
            return VisualNoveltyCues(patches=[])
        left_f = self._as_chw(left)
        right_f = self._as_chw(right)
        patches = self._observe_eye("left", left_f) + self._observe_eye("right", right_f)
        return VisualNoveltyCues(patches=patches)


def apply_visual_explore_goal(
    goal: np.ndarray,
    cues: VisualNoveltyCues | dict[str, Any] | None,
    *,
    config: VisualNoveltyConfig | None = None,
    wall_hazard: float = 0.0,
) -> np.ndarray:
    """Pull motor goal toward the peak novel patch bearing (no blend across patches)."""
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
