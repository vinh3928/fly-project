"""Interrupt Architecture-B: cue → joint STM history → deep-thought freeze."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from flybrain.memory.embedder import EmbeddingParts
from flybrain.memory.pgvector import ShortTermMemory, _cosine_distance
from flybrain.phase4 import (
    EYE_VEC_DIM,
    INTERRUPT_JOINT_VEC_DIM,
    PATCH_CELL_FEAT_DIM,
    PATCH_VEC_DIM,
    SENSORS_VEC_DIM,
    SMELL_VEC_DIM,
)
from flybrain.signals.builder import SensorySignals
from flybrain.thinking.retrospective import RetrospectiveConfig, can_investigate_interest
from flybrain.vision.novelty import stereo_patch_embedding

# Hard-stop goal: zero locomotion; height/freq mid; no proboscis.
STOP_GOAL = np.asarray([0.0, 0.0, 0.5, 0.5, 0.0], dtype=np.float32)

# Summary keys treated as external senses (amplified during deep thought).
_DEEP_THOUGHT_SENSE_KEYS = (
    "sugar",
    "bitter",
    "odor",
    "odor_left",
    "odor_right",
    "mean_contact",
    "instability",
    "proboscis_contact",
    "visual_novelty",
    "new_stimulus",
    "wall_hazard",
    "target_confidence",
    "target_bearing",
    "novelty_bearing",
    "forward_bias",
    "turn_bias",
)

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
    #   joint      — L1: retrieve by joint_vec, compare only the cue column
    #   patch_grid — legacy per-square patch + same-row smell (experiments)
    #   stm        — legacy CNN eye/sensor cosine (experiments only)
    history_backend: str = "joint"
    # L1 neighbor pool size (joint retrieval). L2 joint-vs-joint design deferred.
    joint_neighbors: int = 8
    joint_match_min: float = 0.92  # reserved for future L2 whole-joint gate
    # Smell STM-first cue + L1 smell-column threshold when cue=smell_novelty.
    smell_enabled: bool = True
    smell_match_min: float = 0.92
    smell_neighbors: int = 8
    # L1 patch-column threshold when cue is visual_novelty / new_stimulus.
    patch_match_min: float = 0.92
    patch_grid_rows: int = 1  # legacy; hex facets use facet_count=6
    patch_grid_cols: int = 6
    patch_neighbors: int = 8
    patch_cache_size: int = 48
    # L1 sensors-column threshold (future sensor cues).
    sensors_match_min: float = 0.92
    # Legacy / unused by joint path (kept for stm backend + old yaml keys).
    history_match_min: float = 0.55
    history_relative: bool = True
    history_baseline_window: int = 40
    history_relative_drop: float = 0.04
    history_min_baseline_samples: int = 8
    history_bootstrap_baseline: float = 1.0
    min_visual_novelty: float = 0.18
    cooldown_steps: int = 20
    # Hard motor freeze length after arm; mind deep-thought stub runs on interval.
    deep_thought_steps: int = 80
    # Order-of-magnitude boost for mind/devil senses during the freeze (STM unscaled).
    deep_thought_gain: float = 10.0
    # Mind ticks during freeze (normal mind_interval stays for non-freeze steps).
    deep_thought_mind_interval: int = 4
    # Hazard conditioning: repeated devil risks → memory block; N agreeing
    # accept/acknowledge judgements can early-unlock the freeze.
    conditioning_hit_threshold: int = 3
    judgement_confirmations: int = 2
    # Passive smell: cheap always-on antenna watcher that gates the expensive
    # STM smell probe (which used to run every step and dominated step latency).
    # Backed by the generic ModalChangeGate (reusable for vision / sensors).
    passive_smell_enabled: bool = True
    passive_smell_interval: int = 4  # sample antenna every N steps
    passive_smell_change_min: float = 0.05  # per-channel delta counting as change
    passive_smell_refresh_ms: float = 500.0  # forced probe cadence
    passive_smell_ttl_seconds: float = 0.0  # optional wall-clock sample cadence
    modalities: tuple[str, ...] = DEFAULT_MODALITIES

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> StimulusInterruptConfig:
        data = dict(raw or {})
        mods = data.get("modalities")
        if mods is not None:
            data["modalities"] = tuple(str(m) for m in mods)
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


def heighten_sensory_for_deep_thought(
    sensory: SensorySignals,
    *,
    gain: float = 10.0,
) -> SensorySignals:
    """Clone sensory with mind-facing senses ×gain; leave internals/raw unscaled.

    Eyes are amplified then clipped to [0, 1]. Chemical / smell / contacts may
    exceed 1 so the devil LLM sees intensity. STM / embedder must keep using the
    original (unscaled) sensory.
    """
    g = float(gain)
    summary = dict(sensory.summary)
    for key in _DEEP_THOUGHT_SENSE_KEYS:
        if key in summary:
            summary[key] = float(summary[key]) * g
    left = np.clip(np.asarray(sensory.left_eye, dtype=np.float32) * g, 0.0, 1.0)
    right = np.clip(np.asarray(sensory.right_eye, dtype=np.float32) * g, 0.0, 1.0)
    return SensorySignals(
        body=np.asarray(sensory.body, dtype=np.float32) * g,
        chemical=np.asarray(sensory.chemical, dtype=np.float32) * g,
        internal=np.asarray(sensory.internal, dtype=np.float32).copy(),
        contacts=np.asarray(sensory.contacts, dtype=np.float32) * g,
        left_eye=left,
        right_eye=right,
        raw=np.asarray(sensory.raw, dtype=np.float32),
        smell_vec=np.asarray(sensory.smell_vec, dtype=np.float32) * g,
        summary=summary,
    )


@dataclass
class PassiveSmellSample:
    """One passive-smell observation (no STM access)."""

    sampled: bool = False
    changed: bool = False
    active: bool = False
    delta: float = 0.0
    reason: str = ""
    l1_hit: bool = False


@dataclass
class PassiveSmellGate:
    """Smell-specialized wrapper around :class:`ModalChangeGate`.

    Samples the antenna vector on a step/time cadence and compares it to the
    last stored reading in memory only — it never queries or writes STM. The
    expensive smell-novelty probe opens when the reading changed, vision saw
    novelty, or ``refresh_ms`` elapsed since the last probe. Unchanged samples
    are L1 hits (``l1_hit=True``).
    """

    interval: int = 4
    change_min: float = 0.05
    refresh_ms: float = 500.0
    ttl_seconds: float = 0.0  # 0 → step-interval only (live default via config)
    _gate: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        from flybrain.sensors.modal_gate import ModalChangeGate

        self._gate = ModalChangeGate(
            modality="smell",
            ttl_seconds=float(self.ttl_seconds),
            step_interval=max(1, int(self.interval)),
            change_min=float(self.change_min),
            refresh_seconds=float(self.refresh_ms) / 1000.0,
            compare="max_abs",
            dim=int(SMELL_VEC_DIM),
            first_nonzero_is_change=True,
        )

    @classmethod
    def from_config(cls, config: StimulusInterruptConfig) -> PassiveSmellGate:
        return cls(
            interval=max(1, int(config.passive_smell_interval)),
            change_min=float(config.passive_smell_change_min),
            refresh_ms=float(config.passive_smell_refresh_ms),
            ttl_seconds=float(getattr(config, "passive_smell_ttl_seconds", 0.0) or 0.0),
        )

    def reset(self) -> None:
        if self._gate is not None:
            self._gate.reset()

    @property
    def _last(self) -> np.ndarray | None:
        """Back-compat for tests that inspect the L1 baseline."""
        return None if self._gate is None else self._gate._last

    def observe(
        self,
        smell_vec: np.ndarray | None,
        *,
        step: int,
        now_ms: float,
        visual_novel: bool = False,
    ) -> PassiveSmellSample:
        sample = self._gate.observe(
            smell_vec,
            now_ms=float(now_ms),
            step=int(step),
            force_open=bool(visual_novel),
            force_reason="vision_novelty",
        )
        return PassiveSmellSample(
            sampled=bool(sample.sampled),
            changed=bool(sample.changed),
            active=bool(sample.active),
            delta=float(sample.delta),
            reason=str(sample.reason),
            l1_hit=bool(sample.l1_hit),
        )


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
    # Passive smell gate telemetry (why/whether the STM smell probe ran).
    passive_smell_changed: bool = False
    passive_smell_delta: float = 0.0
    passive_smell_reason: str = ""
    smell_probe_ran: bool = False
    # Index (0..11) of the hexagonal facet that matched STM worst; -1 if N/A.
    patch_worst_cell: int = -1
    # Stashed when arming so the confirmed interrupt can persist to STM.
    patch_embedding: np.ndarray | None = None
    smell_embedding: np.ndarray | None = None
    sensors_embedding: np.ndarray | None = None
    joint_embedding: np.ndarray | None = None


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
    grid_rows: int = 1
    grid_cols: int = 6

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


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros_like(v)
    return (v / n).astype(np.float32)


def build_interrupt_joint_vec(
    patch: np.ndarray | None,
    smell: np.ndarray | None,
    sensors: np.ndarray | None,
) -> np.ndarray:
    """L2-normalize each modality block, concat, then L2-normalize the joint.

    Equal block weight keeps 16-d smell from being buried by 60-d patch / 192-d sensors.
    """
    p = _l2_normalize(
        np.asarray(patch if patch is not None else np.zeros(PATCH_VEC_DIM), dtype=np.float32)
    )
    if p.shape[0] != PATCH_VEC_DIM:
        padded = np.zeros(PATCH_VEC_DIM, dtype=np.float32)
        n = min(PATCH_VEC_DIM, p.shape[0])
        padded[:n] = p[:n]
        p = _l2_normalize(padded)
    s = _l2_normalize(
        np.asarray(smell if smell is not None else np.zeros(SMELL_VEC_DIM), dtype=np.float32)
    )
    if s.shape[0] != SMELL_VEC_DIM:
        padded = np.zeros(SMELL_VEC_DIM, dtype=np.float32)
        n = min(SMELL_VEC_DIM, s.shape[0])
        padded[:n] = s[:n]
        s = _l2_normalize(padded)
    sens = _l2_normalize(
        np.asarray(
            sensors if sensors is not None else np.zeros(SENSORS_VEC_DIM), dtype=np.float32
        )
    )
    if sens.shape[0] != SENSORS_VEC_DIM:
        padded = np.zeros(SENSORS_VEC_DIM, dtype=np.float32)
        n = min(SENSORS_VEC_DIM, sens.shape[0])
        padded[:n] = sens[:n]
        sens = _l2_normalize(padded)
    joint = np.concatenate([p, s, sens]).astype(np.float32)
    assert joint.shape[0] == INTERRUPT_JOINT_VEC_DIM
    return _l2_normalize(joint)


def has_candidate_cue(
    *,
    signals: dict[str, Any],
    visual_novelty: float = 0.0,
    new_stimulus: bool = False,
    config: StimulusInterruptConfig,
    stimulus_thresholds: dict[str, float] | None = None,
) -> tuple[bool, str]:
    """Cheap gate before history queries (taste / visual). Smell cue is STM-first."""
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


def smell_novelty_vs_stm(
    memory: ShortTermMemory | None,
    smell_emb: np.ndarray | None,
    *,
    fly_id: str,
    match_min: float,
    top_k: int = 8,
) -> tuple[bool, float, bool]:
    """Return (is_novel, best_sim, empty). Zero query smell is never a smell cue."""
    if smell_emb is None or memory is None:
        return False, 1.0, True
    q = np.asarray(smell_emb, dtype=np.float32).reshape(-1)
    if q.shape[0] != SMELL_VEC_DIM:
        padded = np.zeros(SMELL_VEC_DIM, dtype=np.float32)
        n = min(SMELL_VEC_DIM, q.shape[0])
        padded[:n] = q[:n]
        q = padded
    if float(np.linalg.norm(q)) < 1e-6:
        return False, 1.0, True
    neighbors = memory.similar_by(
        q,
        modality="smell",  # type: ignore[arg-type]
        fly_id=fly_id,
        top_k=max(1, int(top_k)),
    )
    best = 0.0
    any_valid = False
    for n in neighbors:
        sv = np.asarray(n.smell_vec, dtype=np.float32).reshape(-1)
        if float(np.linalg.norm(sv)) < 1e-6:
            continue
        any_valid = True
        best = max(best, cosine_similarity(q, sv))
    if not any_valid:
        return True, 0.0, True
    return bool(best < float(match_min)), float(best), False


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
    """Per-facet (eye,facet) best cosine of ``query`` vs stored hex facet grids."""
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
    """Return (worst_square_cosine, empty, worst_square_index). Legacy patch_grid helper."""
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


@dataclass
class PatchSmellMatch:
    """Legacy joint vision+smell history check (patch_grid backend)."""

    empty: bool = True
    vision_miss: bool = True
    smell_miss: bool = False
    patch_sim: float = 0.0
    smell_sim: float = 1.0
    worst_cell: int = -1
    reason: str = "empty_patch_stm"


def match_patch_smell_stm(
    memory: ShortTermMemory,
    patch_emb: np.ndarray,
    smell_emb: np.ndarray | None,
    *,
    fly_id: str,
    patch_match_min: float,
    smell_match_min: float,
    smell_enabled: bool = True,
    top_k: int = 8,
    feat_dim: int = PATCH_CELL_FEAT_DIM,
) -> PatchSmellMatch:
    """Legacy vision-primary neighbor lookup; smell on the same STM row."""
    neighbors = memory.similar_by(
        np.asarray(patch_emb, dtype=np.float32),
        modality="patches",  # type: ignore[arg-type]
        fly_id=fly_id,
        top_k=max(1, int(top_k)),
    )
    valid = []
    for n in neighbors:
        pv = np.asarray(n.patch_vec, dtype=np.float32).reshape(-1)
        if float(np.linalg.norm(pv)) >= 1e-6:
            valid.append(n)
    out = PatchSmellMatch()
    if not valid:
        return out

    smell_q = (
        np.asarray(smell_emb, dtype=np.float32).reshape(-1)
        if smell_emb is not None
        else np.zeros(SMELL_VEC_DIM, dtype=np.float32)
    )
    if smell_q.shape[0] != SMELL_VEC_DIM:
        padded = np.zeros(SMELL_VEC_DIM, dtype=np.float32)
        n = min(SMELL_VEC_DIM, smell_q.shape[0])
        padded[:n] = smell_q[:n]
        smell_q = padded

    patch_thr = float(patch_match_min)
    smell_thr = float(smell_match_min)
    best_vision = -1.0
    best_worst_cell = -1
    vision_hits: list[tuple[float, float, int]] = []

    for rec in valid:
        per_cell = per_square_best_similarity(
            patch_emb, [np.asarray(rec.patch_vec, dtype=np.float32)], feat_dim=feat_dim
        )
        if per_cell.size == 0:
            continue
        worst_i = int(np.argmin(per_cell))
        patch_sim = float(per_cell[worst_i])
        if patch_sim > best_vision:
            best_vision = patch_sim
            best_worst_cell = worst_i
        if patch_sim < patch_thr:
            continue

        stored_smell = np.asarray(rec.smell_vec, dtype=np.float32).reshape(-1)
        if (not smell_enabled) or float(np.linalg.norm(stored_smell)) < 1e-6:
            smell_sim = 1.0
            smell_ok = True
        else:
            smell_sim = cosine_similarity(smell_q, stored_smell)
            smell_ok = smell_sim >= smell_thr
        vision_hits.append((patch_sim, smell_sim, worst_i))
        if smell_ok:
            out.empty = False
            out.vision_miss = False
            out.smell_miss = False
            out.patch_sim = patch_sim
            out.smell_sim = float(smell_sim)
            out.worst_cell = worst_i
            out.reason = "history_hit"
            return out

    out.empty = False
    out.patch_sim = float(best_vision) if best_vision >= 0.0 else 0.0
    out.worst_cell = int(best_worst_cell)
    if not vision_hits:
        out.vision_miss = True
        out.smell_miss = False
        out.smell_sim = 1.0
        out.reason = "novel_vs_patches"
        return out

    out.vision_miss = False
    out.smell_miss = True
    out.smell_sim = float(max(s for _, s, _ in vision_hits))
    out.patch_sim = float(max(p for p, _, _ in vision_hits))
    out.worst_cell = int(max(vision_hits, key=lambda t: t[0])[2])
    out.reason = "novel_vs_smell"
    return out


@dataclass
class JointMatch:
    """L1 history check: joint neighbor pool, cue-column familiarity."""

    empty: bool = True
    miss: bool = True
    best_sim: float = 0.0
    column: str = "patches"
    reason: str = "empty_joint_stm"


def cue_l1_column(cue: str) -> str:
    """Map interrupt cue → STM column compared at L1 (after joint retrieval)."""
    if cue == "smell_novelty":
        return "smell"
    if cue in ("sugar", "bitter"):
        # Taste still force-misses later; column unused for arming.
        return "sensors"
    # visual_novelty, new_stimulus, and unknown → patches
    return "patches"


def _pad_to(vec: np.ndarray | None, dim: int) -> np.ndarray:
    if vec is None:
        return np.zeros(dim, dtype=np.float32)
    a = np.asarray(vec, dtype=np.float32).reshape(-1)
    if a.shape[0] == dim:
        return a
    out = np.zeros(dim, dtype=np.float32)
    n = min(dim, a.shape[0])
    out[:n] = a[:n]
    return out


def match_joint_l1_stm(
    memory: ShortTermMemory,
    joint_emb: np.ndarray,
    *,
    cue: str,
    patch_emb: np.ndarray | None,
    smell_emb: np.ndarray | None,
    sensors_emb: np.ndarray | None,
    fly_id: str,
    patch_match_min: float,
    smell_match_min: float,
    sensors_match_min: float,
    top_k: int = 8,
) -> JointMatch:
    """L1: retrieve top-k by joint_vec, then compare only the cue-triggering column.

    L2 whole-joint familiarity is deferred — retrieval stays joint, gate is column-local.
    """
    neighbors = memory.similar_by(
        np.asarray(joint_emb, dtype=np.float32),
        modality="joint",  # type: ignore[arg-type]
        fly_id=fly_id,
        top_k=max(1, int(top_k)),
    )
    out = JointMatch()
    column = cue_l1_column(cue)
    out.column = column

    col_attr = {
        "patches": "patch_vec",
        "smell": "smell_vec",
        "sensors": "sensors_vec",
    }[column]
    col_dim = {
        "patches": PATCH_VEC_DIM,
        "smell": SMELL_VEC_DIM,
        "sensors": SENSORS_VEC_DIM,
    }[column]
    match_min = {
        "patches": float(patch_match_min),
        "smell": float(smell_match_min),
        "sensors": float(sensors_match_min),
    }[column]
    query = {
        "patches": _pad_to(patch_emb, PATCH_VEC_DIM),
        "smell": _pad_to(smell_emb, SMELL_VEC_DIM),
        "sensors": _pad_to(sensors_emb, SENSORS_VEC_DIM),
    }[column]

    best = 0.0
    any_joint = False
    any_col = False
    for n in neighbors:
        jv = np.asarray(n.joint_vec, dtype=np.float32).reshape(-1)
        if float(np.linalg.norm(jv)) < 1e-6:
            continue
        any_joint = True
        stored = _pad_to(getattr(n, col_attr), col_dim)
        if float(np.linalg.norm(stored)) < 1e-6:
            continue
        if float(np.linalg.norm(query)) < 1e-6:
            continue
        any_col = True
        best = max(best, cosine_similarity(query, stored))

    if not any_joint:
        return out

    out.empty = False
    out.best_sim = float(best)
    novel_reason = {
        "patches": "novel_vs_patches",
        "smell": "novel_vs_smell",
        "sensors": "novel_vs_sensors",
    }[column]
    if not any_col or best < match_min:
        out.miss = True
        out.reason = novel_reason
    else:
        out.miss = False
        out.reason = "history_hit"
    return out


def match_joint_stm(
    memory: ShortTermMemory,
    joint_emb: np.ndarray,
    *,
    fly_id: str,
    match_min: float,
    top_k: int = 8,
) -> JointMatch:
    """Deprecated whole-joint compare; prefer ``match_joint_l1_stm``."""
    neighbors = memory.similar_by(
        np.asarray(joint_emb, dtype=np.float32),
        modality="joint",  # type: ignore[arg-type]
        fly_id=fly_id,
        top_k=max(1, int(top_k)),
    )
    out = JointMatch()
    out.column = "joint"
    best = 0.0
    any_valid = False
    for n in neighbors:
        jv = np.asarray(n.joint_vec, dtype=np.float32).reshape(-1)
        if float(np.linalg.norm(jv)) < 1e-6:
            continue
        any_valid = True
        best = max(best, cosine_similarity(joint_emb, jv))
    if not any_valid:
        return out
    out.empty = False
    out.best_sim = float(best)
    if best >= float(match_min):
        out.miss = False
        out.reason = "history_hit"
    else:
        out.miss = True
        out.reason = "novel_vs_joint"
    return out


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
    smell_vec: np.ndarray | None = None,
    sensors_vec: np.ndarray | None = None,
    passive_smell: PassiveSmellGate | None = None,
    now_ms: float | None = None,
) -> StimulusInterruptDecision:
    """Decide whether to arm a stimulus interrupt.

    Default (``history_backend=joint``):
      0) Passive smell (cheap, in-memory) gates the STM smell probe, and a
         no-cue step returns before any STM query.
      1) Taste / visual / STM smell-novelty opens the candidate cue.
      2) Build L2-block-normalized joint(patch, smell, sensors).
      3) L1: retrieve top-k by joint_vec, then compare only the cue column.
      4) STM miss ⇒ may arm; on confirm patch+smell+sensors+joint are saved.
         (L2 whole-joint familiarity deferred.)
    """
    del patch_cache
    out = StimulusInterruptDecision()
    if not config.enabled:
        out.reason = "disabled"
        return out
    if step < int(cooldown_until_step):
        out.reason = "cooldown"
        return out

    smell_emb: np.ndarray | None = None
    if smell_vec is not None:
        smell_emb = np.asarray(smell_vec, dtype=np.float32).reshape(-1).copy()
        if smell_emb.shape[0] != SMELL_VEC_DIM:
            padded = np.zeros(SMELL_VEC_DIM, dtype=np.float32)
            n = min(SMELL_VEC_DIM, smell_emb.shape[0])
            padded[:n] = smell_emb[:n]
            smell_emb = padded

    candidate, cue = has_candidate_cue(
        signals=signals,
        visual_novelty=visual_novelty,
        new_stimulus=new_stimulus,
        config=config,
        stimulus_thresholds=stimulus_thresholds,
    )
    smell_cue_sim = 1.0
    if (not candidate) and bool(config.smell_enabled):
        probe = True
        if passive_smell is not None and bool(config.passive_smell_enabled):
            sample = passive_smell.observe(
                smell_emb,
                step=int(step),
                now_ms=float(now_ms) if now_ms is not None else time.monotonic() * 1000.0,
                # Sub-cue visual motion still refreshes smell; the cue-level
                # threshold would already have opened a candidate above.
                visual_novel=bool(new_stimulus)
                or float(visual_novelty) >= 0.5 * float(config.min_visual_novelty),
            )
            probe = bool(sample.active)
            out.passive_smell_changed = bool(sample.changed)
            out.passive_smell_delta = float(sample.delta)
            out.passive_smell_reason = sample.reason
        if probe:
            out.smell_probe_ran = True
            novel_smell, smell_cue_sim, _ = smell_novelty_vs_stm(
                memory,
                smell_emb,
                fly_id=fly_id,
                match_min=float(config.smell_match_min),
                top_k=int(config.smell_neighbors),
            )
            if novel_smell:
                candidate = True
                cue = "smell_novelty"
    out.candidate = candidate
    out.cue = cue

    backend = str(config.history_backend or "joint").lower()
    if not candidate and backend == "joint":
        # No cue ⇒ nothing to compare against history; skip the joint STM query.
        out.reason = "no_cue"
        out.modality_scores = {"smell_cue": float(smell_cue_sim)}
        return out

    miss = False
    empty = False
    patch_emb: np.ndarray | None = None
    sensors_emb: np.ndarray | None = None
    joint_emb: np.ndarray | None = None
    match_reason = ""

    if sensors_vec is not None:
        sensors_emb = np.asarray(sensors_vec, dtype=np.float32).reshape(-1).copy()
    elif parts is not None:
        sensors_emb = np.asarray(parts.sensors_vec, dtype=np.float32).reshape(-1).copy()

    if backend == "joint":
        if left_eye is not None and right_eye is not None:
            patch_emb = stereo_patch_embedding(
                left_eye,
                right_eye,
                grid_rows=int(config.patch_grid_rows),
                grid_cols=int(config.patch_grid_cols),
            )
            assert patch_emb.shape[0] == PATCH_VEC_DIM
        if patch_emb is not None and memory is not None:
            joint_emb = build_interrupt_joint_vec(patch_emb, smell_emb, sensors_emb)
            matched = match_joint_l1_stm(
                memory,
                joint_emb,
                cue=cue or "visual_novelty",
                patch_emb=patch_emb,
                smell_emb=smell_emb,
                sensors_emb=sensors_emb,
                fly_id=fly_id,
                patch_match_min=float(config.patch_match_min),
                smell_match_min=float(config.smell_match_min),
                sensors_match_min=float(config.sensors_match_min),
                top_k=int(config.joint_neighbors),
            )
            empty = bool(matched.empty)
            miss = bool(matched.miss)
            out.best_similarity = float(matched.best_sim)
            out.modality_scores = {
                matched.column: float(matched.best_sim),
                "smell_cue": float(smell_cue_sim),
            }
            match_reason = matched.reason
            out.novel_modalities = [matched.column] if miss else []
        else:
            empty = True
            miss = True
            out.best_similarity = 0.0
            out.modality_scores = {}
            out.novel_modalities = [cue_l1_column(cue or "visual_novelty")]
            match_reason = "empty_joint_stm"
    elif backend == "patch_grid":
        if left_eye is not None and right_eye is not None and memory is not None:
            patch_emb = stereo_patch_embedding(
                left_eye,
                right_eye,
                grid_rows=int(config.patch_grid_rows),
                grid_cols=int(config.patch_grid_cols),
            )
            assert patch_emb.shape[0] == PATCH_VEC_DIM
            matched = match_patch_smell_stm(
                memory,
                patch_emb,
                smell_emb,
                fly_id=fly_id,
                patch_match_min=float(config.patch_match_min),
                smell_match_min=float(config.smell_match_min),
                smell_enabled=bool(config.smell_enabled),
                top_k=int(config.patch_neighbors),
                feat_dim=PATCH_CELL_FEAT_DIM,
            )
            empty = bool(matched.empty)
            out.best_similarity = float(matched.patch_sim)
            out.patch_worst_cell = int(matched.worst_cell)
            out.modality_scores = {
                "patches": float(matched.patch_sim),
                "smell": float(matched.smell_sim),
            }
            match_reason = matched.reason
            if matched.empty:
                miss = True
                out.novel_modalities = ["patches"]
            elif matched.vision_miss:
                miss = True
                out.novel_modalities = ["patches"]
            elif matched.smell_miss:
                miss = True
                out.novel_modalities = ["smell"]
            else:
                miss = False
                out.novel_modalities = []
        else:
            empty = True
            miss = True
            out.best_similarity = 0.0
            out.modality_scores = {}
            out.novel_modalities = ["patches"]
            match_reason = "empty_patch_stm"
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

    # Taste always counts as novel history.
    if cue in ("sugar", "bitter"):
        miss = True
        out.novel_modalities = list(dict.fromkeys([*out.novel_modalities, cue]))

    out.history_miss = miss
    if not out.history_miss:
        out.reason = "history_hit"
        return out

    out.should_arm = True
    out.patch_embedding = patch_emb
    out.smell_embedding = smell_emb
    out.sensors_embedding = sensors_emb
    out.joint_embedding = joint_emb
    if backend == "joint":
        if empty:
            out.reason = "empty_joint_stm"
        else:
            out.reason = match_reason or "novel_vs_joint"
    elif backend == "patch_grid":
        if empty:
            out.reason = "empty_patch_stm"
        elif match_reason == "novel_vs_smell":
            out.reason = "novel_vs_smell"
        else:
            out.reason = "novel_vs_patches"
    else:
        out.reason = "empty_stm" if empty else "novel_vs_stm"
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
