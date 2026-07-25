"""Interrupt Architecture-B: 3×3 novelty cue → STM patch history → hard-stop."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from flybrain.memory.embedder import EmbeddingParts
from flybrain.memory.pgvector import ShortTermMemory, _cosine_distance
from flybrain.phase4 import (
    EYE_VEC_DIM,
    PATCH_CELL_FEAT_DIM,
    PATCH_VEC_DIM,
    SENSORS_VEC_DIM,
)
from flybrain.thinking.retrospective import RetrospectiveConfig, can_investigate_interest
from flybrain.vision.novelty import stereo_patch_embedding

# Hard-stop goal: zero locomotion; height/freq mid; no proboscis.
STOP_GOAL = np.asarray([0.0, 0.0, 0.5, 0.5, 0.0], dtype=np.float32)

DEFAULT_MODALITIES: tuple[str, ...] = ("left_eye", "right_eye", "sensors")

_MODALITY_PART_ATTR = {
    "left_eye": "left_eye_vec",
    "right_eye": "right_eye_vec",
    "sensors": "sensors_vec",
}


@dataclass
class StimulusInterruptConfig:
    enabled: bool = True
    # History backend after a candidate cue.
    #   patch_grid — 3×3 stereo embedding checked/saved against STM ``patch_vec``
    #   stm        — legacy CNN eye/sensor cosine (experiments only)
    history_backend: str = "patch_grid"
    patch_match_min: float = 0.92  # worst-square cosine ≥ this ⇒ already seen (not novel)
    patch_grid_rows: int = 3
    patch_grid_cols: int = 3
    # How many STM grids to pool per-square matches over (best square per position).
    patch_neighbors: int = 8
    # Legacy / unused by patch_grid path (kept for stm backend + old yaml keys).
    patch_cache_size: int = 48
    history_match_min: float = 0.55
    history_relative: bool = True
    history_baseline_window: int = 40
    history_relative_drop: float = 0.04
    history_min_baseline_samples: int = 8
    history_bootstrap_baseline: float = 1.0
    min_visual_novelty: float = 0.18
    cooldown_steps: int = 20
    modalities: tuple[str, ...] = DEFAULT_MODALITIES

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> StimulusInterruptConfig:
        data = dict(raw or {})
        mods = data.get("modalities")
        if mods is not None:
            data["modalities"] = tuple(str(m) for m in mods)
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class StimulusInterruptDecision:
    """Result of one interrupt evaluation."""

    candidate: bool = False
    history_miss: bool = False
    should_arm: bool = False
    soft_low_load: bool = False
    crisis: bool = False
    reason: str = ""
    best_similarity: float = 1.0
    modality_scores: dict[str, float] = field(default_factory=dict)
    history_baselines: dict[str, float] = field(default_factory=dict)
    novel_modalities: list[str] = field(default_factory=list)
    cue: str = ""
    # Index (0..17) of the 3×3 square that matched STM worst; -1 if N/A.
    patch_worst_cell: int = -1
    # Stashed when arming so the confirmed interrupt can persist to STM.
    patch_embedding: np.ndarray | None = None


@dataclass
class RelativeHistoryBaseline:
    """Per-modality rolling cosine baseline for relative STM novelty (legacy)."""

    window: int = 40
    drop: float = 0.04
    min_samples: int = 8
    bootstrap_baseline: float = 1.0
    _hist: dict[str, deque[float]] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: StimulusInterruptConfig) -> RelativeHistoryBaseline:
        return cls(
            window=max(2, int(config.history_baseline_window)),
            drop=float(config.history_relative_drop),
            min_samples=max(1, int(config.history_min_baseline_samples)),
            bootstrap_baseline=float(config.history_bootstrap_baseline),
        )

    def reset(self) -> None:
        self._hist.clear()

    def baseline_for(self, modality: str) -> float:
        hist = self._hist.get(modality)
        if hist is None or len(hist) < self.min_samples:
            return float(self.bootstrap_baseline)
        return float(sum(hist) / len(hist))

    def check(
        self,
        scores: dict[str, float],
        *,
        empty: bool,
    ) -> tuple[bool, dict[str, float], list[str]]:
        if empty:
            return True, {}, []
        baselines: dict[str, float] = {}
        novel: list[str] = []
        for mod, sim in scores.items():
            base = self.baseline_for(mod)
            baselines[mod] = base
            if float(sim) < base - float(self.drop):
                novel.append(mod)
        return bool(novel), baselines, novel

    def observe(self, scores: dict[str, float]) -> None:
        for mod, sim in scores.items():
            hist = self._hist.setdefault(mod, deque(maxlen=self.window))
            hist.append(float(sim))


@dataclass
class PatchGridHistoryCache:
    """Deprecated local cache (kept for import/API compat). Gate now uses STM."""

    cache_size: int = 48
    match_min: float = 0.92
    grid_rows: int = 3
    grid_cols: int = 3

    @classmethod
    def from_config(cls, config: StimulusInterruptConfig) -> PatchGridHistoryCache:
        return cls(
            cache_size=max(1, int(config.patch_cache_size)),
            match_min=float(config.patch_match_min),
            grid_rows=max(1, int(config.patch_grid_rows)),
            grid_cols=max(1, int(config.patch_grid_cols)),
        )

    def reset(self) -> None:
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - _cosine_distance(a, b))


def has_candidate_cue(
    *,
    signals: dict[str, Any],
    visual_novelty: float = 0.0,
    new_stimulus: bool = False,
    config: StimulusInterruptConfig,
    stimulus_thresholds: dict[str, float] | None = None,
) -> tuple[bool, str]:
    """Cheap gate before history queries."""
    thr = stimulus_thresholds or {}
    sugar_thr = float(thr.get("sugar", 0.15))
    bitter_thr = float(thr.get("bitter", 0.2))
    sugar = float(signals.get("sugar", 0.0) or 0.0)
    bitter = float(signals.get("bitter", 0.0) or 0.0)
    if sugar >= sugar_thr:
        return True, "sugar"
    if bitter >= bitter_thr:
        return True, "bitter"
    if bool(new_stimulus):
        return True, "new_stimulus"
    if float(visual_novelty) >= float(config.min_visual_novelty):
        return True, "visual_novelty"
    return False, ""


def best_history_similarity(
    memory: ShortTermMemory,
    parts: EmbeddingParts,
    *,
    fly_id: str,
    modalities: Sequence[str] = DEFAULT_MODALITIES,
) -> tuple[float, dict[str, float], bool]:
    """Legacy STM CNN cosine probe (min_sim, per-modality sims, stm_empty)."""
    recent = memory.recent(fly_id=fly_id, limit=1)
    if not recent:
        return 0.0, {}, True

    scores: dict[str, float] = {}
    for mod in modalities:
        attr = _MODALITY_PART_ATTR.get(mod)
        if attr is None:
            continue
        query = np.asarray(getattr(parts, attr), dtype=np.float32).reshape(-1)
        neighbors = memory.similar_by(query, modality=mod, fly_id=fly_id, top_k=1)  # type: ignore[arg-type]
        if not neighbors:
            scores[mod] = 0.0
            continue
        neigh_attr = {
            "left_eye": "left_eye_vec",
            "right_eye": "right_eye_vec",
            "sensors": "sensors_vec",
        }[mod]
        neigh = np.asarray(getattr(neighbors[0], neigh_attr), dtype=np.float32).reshape(-1)
        scores[mod] = cosine_similarity(query, neigh)
    if not scores:
        return 0.0, {}, False
    return float(min(scores.values())), scores, False


def per_square_best_similarity(
    query: np.ndarray,
    neighbors: Sequence[np.ndarray],
    *,
    feat_dim: int = PATCH_CELL_FEAT_DIM,
) -> np.ndarray:
    """Per-square (eye,row,col) best cosine of ``query`` vs stored 3×3 grids.

    The flat 90-d patch vector is ordered left-9-squares then right-9-squares, each
    ``feat_dim`` floats, so slice ``i`` is one fixed square/position. Matching each
    square independently (and taking the closest stored square at that position)
    stops a single novel square from being averaged away by ~17 familiar ones —
    which is why the whole-vector cosine sat at ~0.99 and never armed.
    """
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    step = max(1, int(feat_dim))
    n_cells = q.shape[0] // step
    mats = [np.asarray(n, dtype=np.float32).reshape(-1) for n in neighbors]
    best = np.zeros(n_cells, dtype=np.float32)
    for i in range(n_cells):
        s = i * step
        e = s + step
        q_cell = q[s:e]
        cell_best = 0.0
        for m in mats:
            if m.shape[0] < e:
                continue
            cell_best = max(cell_best, cosine_similarity(q_cell, m[s:e]))
        best[i] = cell_best
    return best


def best_patch_stm_similarity(
    memory: ShortTermMemory,
    patch_emb: np.ndarray,
    *,
    fly_id: str,
    top_k: int = 8,
    feat_dim: int = PATCH_CELL_FEAT_DIM,
) -> tuple[float, bool, int]:
    """Return (worst_square_cosine, empty, worst_square_index).

    The grid is treated as "already seen" only when *every* 3×3 square has a close
    match at its own position across recent STM grids; the least-similar square
    (the min over positions) is what the caller gates on.
    """
    neighbors = memory.similar_by(
        np.asarray(patch_emb, dtype=np.float32),
        modality="patches",  # type: ignore[arg-type]
        fly_id=fly_id,
        top_k=max(1, int(top_k)),
    )
    valid: list[np.ndarray] = []
    for n in neighbors:
        pv = np.asarray(n.patch_vec, dtype=np.float32).reshape(-1)
        if float(np.linalg.norm(pv)) >= 1e-6:
            valid.append(pv)
    if not valid:
        return 0.0, True, -1
    per_cell = per_square_best_similarity(patch_emb, valid, feat_dim=feat_dim)
    if per_cell.size == 0:
        return 1.0, False, -1
    worst = int(np.argmin(per_cell))
    return float(per_cell[worst]), False, worst


def evaluate_stimulus_interrupt(
    *,
    memory: ShortTermMemory | None = None,
    parts: EmbeddingParts | None = None,
    signals: dict[str, Any],
    affect: dict[str, Any],
    fly_id: str = "fly_001",
    step: int,
    cooldown_until_step: int,
    config: StimulusInterruptConfig,
    stimulus_thresholds: dict[str, float] | None = None,
    retrospective: RetrospectiveConfig | None = None,
    visual_novelty: float = 0.0,
    new_stimulus: bool = False,
    baseline: RelativeHistoryBaseline | None = None,
    patch_cache: PatchGridHistoryCache | None = None,
    left_eye: np.ndarray | None = None,
    right_eye: np.ndarray | None = None,
) -> StimulusInterruptDecision:
    """Decide whether to arm a stimulus interrupt.

    Default (``history_backend=patch_grid``):
      1) 3×3 novelty / taste opens the candidate cue.
      2) Each 3×3 square is matched to the closest stored square at the *same*
         eye/row/col position across recent STM ``patch_vec`` grids.
      3) Every square matching (worst square ≥ ``patch_match_min``) ⇒ not novel
         (history_hit); a single unfamiliar square keeps the miss alive.
      4) STM miss ⇒ may arm; on confirm the embedding is saved into STM.
    """
    del patch_cache  # local cache no longer gates; STM is the durable history
    out = StimulusInterruptDecision()
    if not config.enabled:
        out.reason = "disabled"
        return out
    if step < int(cooldown_until_step):
        out.reason = "cooldown"
        return out

    candidate, cue = has_candidate_cue(
        signals=signals,
        visual_novelty=visual_novelty,
        new_stimulus=new_stimulus,
        config=config,
        stimulus_thresholds=stimulus_thresholds,
    )
    out.candidate = candidate
    out.cue = cue

    backend = str(config.history_backend or "patch_grid").lower()
    miss = False
    empty = False
    patch_emb: np.ndarray | None = None

    if backend == "patch_grid":
        if left_eye is not None and right_eye is not None and memory is not None:
            patch_emb = stereo_patch_embedding(
                left_eye,
                right_eye,
                grid_rows=int(config.patch_grid_rows),
                grid_cols=int(config.patch_grid_cols),
            )
            assert patch_emb.shape[0] == PATCH_VEC_DIM
            sim, empty, worst_cell = best_patch_stm_similarity(
                memory,
                patch_emb,
                fly_id=fly_id,
                top_k=int(config.patch_neighbors),
                feat_dim=PATCH_CELL_FEAT_DIM,
            )
            out.best_similarity = float(sim)
            out.patch_worst_cell = int(worst_cell)
            out.modality_scores = {"patches": float(sim)}
            thr = float(config.patch_match_min)
            if empty:
                miss = True
                out.novel_modalities = ["patches"]
            elif sim < thr:
                miss = True
                out.novel_modalities = ["patches"]
            else:
                # STM already has this 3×3 grid — not novelty; do not cache/save.
                miss = False
                out.novel_modalities = []
        else:
            empty = True
            miss = True
            out.best_similarity = 0.0
            out.modality_scores = {}
            out.novel_modalities = ["patches"]
    else:
        if memory is None or parts is None:
            empty = True
            miss = True
            out.best_similarity = 0.0
            out.modality_scores = {}
            out.novel_modalities = []
        else:
            min_sim, scores, empty = best_history_similarity(
                memory,
                parts,
                fly_id=fly_id,
                modalities=config.modalities,
            )
            out.best_similarity = min_sim
            out.modality_scores = scores
            if config.history_relative and baseline is not None:
                miss, bases, novel_mods = baseline.check(scores, empty=empty)
                out.history_baselines = bases
                out.novel_modalities = novel_mods
                if scores and not empty:
                    baseline.observe(scores)
            else:
                thr = float(config.history_match_min)
                novel_mods = [m for m, s in scores.items() if float(s) < thr]
                miss = empty or bool(novel_mods)
                out.novel_modalities = novel_mods if not empty else []
                out.history_baselines = {}

    if not candidate:
        out.reason = "no_cue"
        out.history_miss = miss
        return out

    soft = can_investigate_interest(signals=signals, affect=affect, config=retrospective)
    out.soft_low_load = soft
    out.crisis = not soft
    if out.crisis:
        out.reason = "crisis_skip"
        return out

    # Taste always counts as novel history (vision STM patches are vision-only).
    if cue in ("sugar", "bitter"):
        miss = True
        out.novel_modalities = list(dict.fromkeys([*out.novel_modalities, cue]))

    out.history_miss = miss
    if not out.history_miss:
        out.reason = "history_hit"
        return out

    out.should_arm = True
    out.patch_embedding = patch_emb
    if empty:
        out.reason = "empty_patch_stm" if backend == "patch_grid" else "empty_stm"
    else:
        out.reason = "novel_vs_patches" if backend == "patch_grid" else "novel_vs_stm"
    return out


def part_vector_for_tests(*, eye: float = 0.0, sensors: float = 0.0) -> EmbeddingParts:
    """Tiny helper for unit tests — deterministic EmbeddingParts."""
    left = np.zeros(EYE_VEC_DIM, dtype=np.float32)
    right = np.zeros(EYE_VEC_DIM, dtype=np.float32)
    sens = np.zeros(SENSORS_VEC_DIM, dtype=np.float32)
    left[0] = eye
    right[0] = eye * 0.9
    sens[0] = sensors
    emb = np.zeros(272, dtype=np.float32)
    emb[0] = eye
    return EmbeddingParts(
        embedding=emb,
        left_eye_vec=left,
        right_eye_vec=right,
        sensors_vec=sens,
        status_vec=np.zeros(32, dtype=np.float32),
        affect_vec=np.zeros(16, dtype=np.float32),
    )
