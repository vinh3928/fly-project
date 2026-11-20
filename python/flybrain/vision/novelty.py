"""Eye-driven visual novelty: hexagonal aperture, 6 triangular facets per eye."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from flybrain.phase4 import HEX_FACET_COUNT, PATCH_CELL_FEAT_DIM, PATCH_VEC_DIM
from flybrain.vision.hex_eye import (
    apply_hex_aperture,
    facet_centroid_bearing,
    hex_aperture_mask,
    hex_triangle_masks,
)


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
    """Continuous embedding for one facet sample (C,H,W) or masked channels."""
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


def _facet_feature_from_masked(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Feature vector from pixels under a triangular mask (zeros ignored via crop)."""
    if not np.any(mask):
        return np.zeros(PATCH_FEAT_DIM, dtype=np.float32)
    # Compact bounding box of active pixels so std/edge stay meaningful.
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sub = frame[:, y0:y1, x0:x1].copy()
    sub_mask = mask[y0:y1, x0:x1]
    for c in range(sub.shape[0]):
        ch = sub[c]
        ch[~sub_mask] = float(ch[sub_mask].mean()) if np.any(sub_mask) else 0.0
        sub[c] = ch
    return patch_feature_vector(sub)


def eye_triangle_feature_grid(eye: np.ndarray) -> np.ndarray:
    """Return (HEX_FACET_COUNT, PATCH_FEAT_DIM) features for one hexagonal eye."""
    frame = apply_hex_aperture(_as_chw(eye))
    _c, h, w = frame.shape
    masks = hex_triangle_masks(h, w)
    feats = [_facet_feature_from_masked(frame, m) for m in masks]
    return np.stack(feats, axis=0)


def eye_patch_feature_grid(
    eye: np.ndarray,
    *,
    grid_rows: int = 3,
    grid_cols: int = 3,
) -> np.ndarray:
    """Compat wrapper: hexagonal triangular facets (ignores square grid args)."""
    del grid_rows, grid_cols
    return eye_triangle_feature_grid(eye)


def stereo_patch_embedding(
    left: np.ndarray,
    right: np.ndarray,
    *,
    grid_rows: int = 3,
    grid_cols: int = 3,
) -> np.ndarray:
    """Flatten left+right hexagonal facet features (60-d)."""
    del grid_rows, grid_cols
    left_g = eye_triangle_feature_grid(left)
    right_g = eye_triangle_feature_grid(right)
    out = np.concatenate([left_g.reshape(-1), right_g.reshape(-1)]).astype(np.float32)
    assert out.shape[0] == PATCH_VEC_DIM
    return out


def stereo_patch_cells(
    left: np.ndarray,
    right: np.ndarray,
    *,
    grid_rows: int = 3,
    grid_cols: int = 3,
) -> dict[tuple[str, int, int], np.ndarray]:
    """Map (eye, row=0, facet) → continuous facet feature vector."""
    del grid_rows, grid_cols
    out: dict[tuple[str, int, int], np.ndarray] = {}
    for eye_name, frame in (("left", left), ("right", right)):
        grid = eye_triangle_feature_grid(frame)
        for facet in range(HEX_FACET_COUNT):
            out[(eye_name, 0, facet)] = grid[facet]
    return out


@dataclass
class VisualNoveltyConfig:
    enabled: bool = True
    # Legacy aliases kept for YAML; hexagonal eyes always use HEX_FACET_COUNT.
    grid_rows: int = 1
    grid_cols: int = HEX_FACET_COUNT
    facet_count: int = HEX_FACET_COUNT
    signature_bins: int = 6
    first_visit_bonus: float = 1.0
    repeat_decay: float = 0.7
    change_weight: float = 0.45
    novelty_weight: float = 0.55
    # Motor pull when peak-facet visual novelty is high.
    turn_gain: float = 0.85
    forward_gain: float = 0.55
    min_novelty_for_pull: float = 0.18
    # Soft L1 explore boost.
    explore_candidate_bonus: float = 0.28
    # Per-facet habituation: a static triangle loses novelty; change recovers it.
    habituation_enabled: bool = True
    static_change_threshold: float = 0.08
    habituation_rate: float = 0.02
    dishabituation_rate: float = 0.5
    habituation_strength: float = 0.9
    # Slight left/right FOV offset so each eye's azimuth stays distinct.
    eye_bearing_offset: float = 0.2

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> VisualNoveltyConfig:
        data = dict(raw or {})
        data.pop("n_columns", None)
        if "facet_count" not in data:
            data["facet_count"] = HEX_FACET_COUNT
        data["grid_rows"] = 1
        data["grid_cols"] = int(data.get("facet_count") or HEX_FACET_COUNT)
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class PatchNovelty:
    """Novelty for one triangular facet on one eye."""

    eye: str  # "left" | "right"
    row: int  # always 0 for hex facets (compat)
    col: int  # facet index 0..5
    visual_novelty: float
    novelty_bearing: float
    novelty_confidence: float
    new_stimulus: bool
    change: float = 0.0

    @property
    def facet(self) -> int:
        return int(self.col)

    def to_dict(self) -> dict[str, float | bool | int | str]:
        return {
            "eye": self.eye,
            "row": int(self.row),
            "col": int(self.col),
            "facet": int(self.col),
            "visual_novelty": float(self.visual_novelty),
            "novelty_bearing": float(self.novelty_bearing),
            "novelty_confidence": float(self.novelty_confidence),
            "new_stimulus": bool(self.new_stimulus),
            "change": float(self.change),
        }


@dataclass
class VisualNoveltyCues:
    """Per-facet novelty for left/right hexagonal eyes (facets are not fused).

    Scalar ``visual_novelty`` / ``novelty_bearing`` select the **peak** facet
    (argmax), never a weighted blend of eyes or triangles.
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
            "peak_facet": None if peak is None else int(peak.col),
            "patch_novelties": [p.to_dict() for p in self.patches],
        }


@dataclass
class VisualNoveltyTracker:
    """Tracks independent hexagonal-eye facet signatures (left/right never merged)."""

    config: VisualNoveltyConfig = field(default_factory=VisualNoveltyConfig)
    _visits: dict[tuple[str, int, int, int], int] = field(default_factory=dict)
    _prev: dict[str, np.ndarray] = field(default_factory=dict)
    _habituation: dict[tuple[str, int, int], float] = field(default_factory=dict)

    def reset(self) -> None:
        self._visits.clear()
        self._prev.clear()
        self._habituation.clear()

    def _as_chw(self, eye: np.ndarray) -> np.ndarray:
        return apply_hex_aperture(_as_chw(eye))

    def _patch_signature(self, patch: np.ndarray) -> int:
        """Quantize a facet sample. ``patch`` is (C, H, W) color or gray."""
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

    def _facet_crop(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not np.any(mask):
            return np.zeros((frame.shape[0], 1, 1), dtype=np.float32)
        ys, xs = np.where(mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        sub = frame[:, y0:y1, x0:x1].copy()
        sub_mask = mask[y0:y1, x0:x1]
        for c in range(sub.shape[0]):
            ch = sub[c]
            fill = float(ch[sub_mask].mean()) if np.any(sub_mask) else 0.0
            ch[~sub_mask] = fill
            sub[c] = ch
        return sub

    def _observe_eye(self, eye: str, frame: np.ndarray) -> list[PatchNovelty]:
        cfg = self.config
        _c, h, w = frame.shape
        masks = hex_triangle_masks(h, w)
        prev = self._prev.get(eye)
        had_prev = prev is not None and prev.shape == frame.shape
        out: list[PatchNovelty] = []
        n_facets = int(cfg.facet_count) if cfg.facet_count else HEX_FACET_COUNT
        n_facets = min(HEX_FACET_COUNT, max(1, n_facets))

        for facet in range(n_facets):
            mask = masks[facet]
            patch = self._facet_crop(frame, mask)
            sig = self._patch_signature(patch)
            key = (eye, 0, facet, sig)
            before = self._visits.get(key, 0)
            bonus = float(cfg.first_visit_bonus) * (float(cfg.repeat_decay) ** float(before))
            self._visits[key] = before + 1
            new_here = before == 0

            change = 0.0
            if had_prev and prev is not None:
                prev_patch = self._facet_crop(prev, mask)
                # Align shapes if bbox drifts slightly.
                if prev_patch.shape == patch.shape:
                    delta = float(
                        np.mean(np.abs(patch.astype(np.float32) - prev_patch.astype(np.float32)))
                    )
                    change = float(np.clip(delta / 0.15, 0.0, 1.0))

            interest = float(cfg.novelty_weight) * bonus + float(
                cfg.change_weight
            ) * change * float(cfg.first_visit_bonus)
            novelty = _clamp01(0.55 * interest + 0.35 * bonus + (0.25 if new_here else 0.0))
            confidence = _clamp01(interest)

            hab_key = (eye, 0, facet)
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

            bearing = facet_centroid_bearing(
                facet, eye=eye, eye_bearing_offset=float(cfg.eye_bearing_offset)
            )
            out.append(
                PatchNovelty(
                    eye=eye,
                    row=0,
                    col=facet,
                    visual_novelty=float(novelty),
                    novelty_bearing=bearing,
                    novelty_confidence=float(confidence),
                    new_stimulus=bool(new_here and novelty >= float(cfg.min_novelty_for_pull)),
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
    """Pull motor goal toward the peak novel facet bearing (no blend across facets)."""
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
    g[1] = float(
        np.clip(
            (1.0 - 0.65 * strength) * float(g[1])
            + 0.65 * strength * bearing * float(cfg.turn_gain),
            -1.0,
            1.0,
        )
    )
    forward_pull = float(cfg.forward_gain) * strength
    if wall_hazard > 0.7:
        forward_pull *= 0.35
    elif wall_hazard > 0.45:
        forward_pull *= 0.65
    g[0] = float(np.clip(max(float(g[0]), forward_pull), -1.0, 1.0))
    g[3] = float(np.clip(max(float(g[3]), 0.45 + 0.35 * strength), 0.0, 1.0))
    return g


# Re-export for callers that need aperture helpers.
__all__ = [
    "PATCH_FEAT_DIM",
    "PatchNovelty",
    "VisualNoveltyConfig",
    "VisualNoveltyCues",
    "VisualNoveltyTracker",
    "apply_hex_aperture",
    "apply_visual_explore_goal",
    "eye_patch_feature_grid",
    "eye_triangle_feature_grid",
    "hex_aperture_mask",
    "patch_feature_vector",
    "stereo_patch_cells",
    "stereo_patch_embedding",
]
