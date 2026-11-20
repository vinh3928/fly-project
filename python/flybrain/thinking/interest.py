"""Deferred interest points: acknowledge now, investigate during low load.

When judgement chooses ``acknowledge``, a salient stimulus is parked here
(linked to an STM id) instead of being acted on. Low-load retrospective pops
pending points and returns an investigation directive the motor can follow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

InterestStatus = Literal["pending", "investigating", "done", "expired"]
InterestKind = Literal["wall", "novelty", "toxin", "odor", "energy", "fall", "other"]


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


@dataclass
class InterestConfig:
    enabled: bool = True
    max_pending: int = 32
    ttl_ticks: int = 800
    investigate_per_low_load: int = 1
    min_novelty_to_enqueue: float = 0.12
    investigate_forward_scale: float = 0.25
    investigate_turn_gain: float = 0.7
    investigate_ttl_ticks: int = 40
    # Re-seeing the same kind+bearing bump priority instead of duplicating.
    merge_bearing_tol: float = 0.25
    # Eye snapshots stored on acknowledge for later "have I seen this?" checks.
    eye_store_size: int = 32
    max_eye_archives: int = 64
    # Above this similarity (0 identical? no — 1=identical), judgement blocks acknowledge.
    acknowledge_eye_match_max: float = 0.4

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> InterestConfig:
        data = dict(raw or {})
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


def _to_gray2d(frame: np.ndarray | None) -> np.ndarray | None:
    if frame is None:
        return None
    arr = np.asarray(frame, dtype=np.float32)
    if arr.size == 0:
        return None
    if arr.ndim == 3:
        # (C,H,W) with C in {1,3} or (H,W,C)
        if arr.shape[0] in (1, 3) and arr.shape[0] < min(arr.shape[1], arr.shape[2]):
            arr = arr.mean(axis=0) if arr.shape[0] == 3 else arr[0]
        elif arr.shape[-1] in (1, 3):
            arr = arr.mean(axis=-1) if arr.shape[-1] == 3 else arr[..., 0]
        else:
            arr = arr.reshape(arr.shape[-2], arr.shape[-1])
    elif arr.ndim != 2:
        arr = arr.reshape(arr.shape[-2], arr.shape[-1])
    if float(np.max(arr)) > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _resize_square(frame: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbor resize without an image dependency."""
    h, w = frame.shape
    if h == size and w == size:
        return frame.astype(np.float32, copy=False)
    ys = (np.linspace(0, h - 1, size)).astype(np.int32)
    xs = (np.linspace(0, w - 1, size)).astype(np.int32)
    return frame[ys][:, xs].astype(np.float32)


def snapshot_eye(frame: np.ndarray | None, *, size: int = 32) -> np.ndarray | None:
    """Downsample an eye frame for compact interest storage."""
    gray = _to_gray2d(frame)
    if gray is None:
        return None
    return _resize_square(gray, int(size))


def frame_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Similarity in [0, 1]: 1=identical, 0=completely different."""
    if a is None or b is None:
        return 0.0
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    if aa.size == 0 or bb.size == 0:
        return 0.0
    if aa.shape != bb.shape:
        size = int(min(aa.shape[0], aa.shape[1], bb.shape[0], bb.shape[1], 32))
        size = max(4, size)
        aa = _resize_square(
            _to_gray2d(aa) or aa.reshape(-1)[: size * size].reshape(size, size), size
        )
        bb = _resize_square(
            _to_gray2d(bb) or bb.reshape(-1)[: size * size].reshape(size, size), size
        )
    # 1 - mean absolute error on [0,1] pixels.
    mae = float(np.mean(np.abs(aa - bb)))
    return _clamp(1.0 - mae, 0.0, 1.0)


def eye_pair_similarity(
    left_a: np.ndarray | None,
    right_a: np.ndarray | None,
    left_b: np.ndarray | None,
    right_b: np.ndarray | None,
) -> float:
    """Mean left/right eye similarity in [0, 1]."""
    scores = [
        frame_similarity(left_a, left_b),
        frame_similarity(right_a, right_b),
    ]
    usable = [s for s, src in zip(scores, (left_a, right_a), strict=False) if src is not None]
    if not usable:
        # Fall back to whichever side we have.
        usable = [s for s in scores if s > 0.0] or [0.0]
    return float(sum(usable) / max(1, len(usable)))


@dataclass
class InterestPoint:
    """A stimulus parked for later investigation (STM-linked bookmark)."""

    id: int
    stm_id: int
    step: int
    kind: InterestKind
    bearing: float
    novelty: float
    priority: float
    reason: str
    status: InterestStatus = "pending"
    fly_id: str = "fly_001"
    created_tick: int = 0
    updated_tick: int = 0
    hits: int = 1
    proposal_risk: str = "none"
    # Compact grayscale eye snapshots at acknowledge time (not logged in full).
    left_eye: np.ndarray | None = None
    right_eye: np.ndarray | None = None
    stimuli_signature: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "stm_id": self.stm_id,
            "step": self.step,
            "kind": self.kind,
            "bearing": float(self.bearing),
            "novelty": float(self.novelty),
            "priority": float(self.priority),
            "reason": self.reason,
            "status": self.status,
            "fly_id": self.fly_id,
            "created_tick": self.created_tick,
            "updated_tick": self.updated_tick,
            "hits": self.hits,
            "proposal_risk": self.proposal_risk,
            "has_eyes": self.left_eye is not None or self.right_eye is not None,
            "eye_shape": None if self.left_eye is None else list(np.asarray(self.left_eye).shape),
            "stimuli_signature": dict(self.stimuli_signature),
        }


@dataclass
class InvestigationDirective:
    """Motor bias while the fly revisits a deferred interest point."""

    point: InterestPoint
    forward_scale: float
    turn: float
    ttl_ticks: int
    remaining: int
    started_tick: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "interest_id": self.point.id,
            "stm_id": self.point.stm_id,
            "kind": self.point.kind,
            "bearing": float(self.point.bearing),
            "forward_scale": float(self.forward_scale),
            "turn": float(self.turn),
            "ttl_ticks": int(self.ttl_ticks),
            "remaining": int(self.remaining),
            "started_tick": int(self.started_tick),
            "reason": self.point.reason,
        }


@dataclass
class InterestPointStore:
    """In-process queue of deferred interest points (STM remains the archive)."""

    config: InterestConfig = field(default_factory=InterestConfig)
    _points: list[InterestPoint] = field(default_factory=list)
    _next_id: int = 1
    _active: InvestigationDirective | None = None

    def reset(self) -> None:
        self._points.clear()
        self._next_id = 1
        self._active = None

    @property
    def active(self) -> InvestigationDirective | None:
        return self._active

    def pending(self) -> list[InterestPoint]:
        return [p for p in self._points if p.status == "pending"]

    def all(self) -> list[InterestPoint]:
        return list(self._points)

    def acknowledge(
        self,
        *,
        stm_id: int,
        step: int,
        tick: int,
        kind: str,
        bearing: float,
        novelty: float,
        reason: str,
        fly_id: str = "fly_001",
        proposal_risk: str = "none",
        priority: float | None = None,
        left_eye: np.ndarray | None = None,
        right_eye: np.ndarray | None = None,
        stimuli_signature: dict[str, Any] | None = None,
    ) -> InterestPoint | None:
        """Park a stimulus for later. Merges with a nearby pending point of same kind."""
        cfg = self.config
        if not cfg.enabled:
            return None
        nov = _clamp(novelty, 0.0, 1.0)
        if nov < float(cfg.min_novelty_to_enqueue) and str(kind) not in (
            "wall",
            "toxin",
            "odor",
            "fall",
            "energy",
        ):
            return None

        kind_n = _normalize_kind(kind)
        bearing_n = _clamp(bearing, -1.0, 1.0)
        pri = float(
            priority
            if priority is not None
            else _clamp(0.35 * nov + 0.4 * _kind_salience(kind_n) + 0.25, 0.0, 1.0)
        )
        left_snap = snapshot_eye(left_eye, size=int(cfg.eye_store_size))
        right_snap = snapshot_eye(right_eye, size=int(cfg.eye_store_size))
        sig = _slim_stimuli(stimuli_signature)

        # Merge: same kind + similar bearing still pending → bump priority/hits.
        for existing in self._points:
            if existing.status != "pending" or existing.kind != kind_n:
                continue
            if abs(existing.bearing - bearing_n) <= float(cfg.merge_bearing_tol):
                existing.hits += 1
                existing.priority = _clamp(existing.priority + 0.08 * nov, 0.0, 1.0)
                existing.novelty = max(existing.novelty, nov)
                existing.updated_tick = int(tick)
                existing.stm_id = int(stm_id)
                existing.step = int(step)
                if reason:
                    existing.reason = reason[:80]
                # Keep the first clear eye pair; refresh if we previously had none.
                if existing.left_eye is None and left_snap is not None:
                    existing.left_eye = left_snap
                if existing.right_eye is None and right_snap is not None:
                    existing.right_eye = right_snap
                if sig:
                    existing.stimuli_signature = sig
                return existing

        point = InterestPoint(
            id=self._next_id,
            stm_id=int(stm_id),
            step=int(step),
            kind=kind_n,
            bearing=bearing_n,
            novelty=nov,
            priority=pri,
            reason=(reason or "")[:80],
            status="pending",
            fly_id=fly_id,
            created_tick=int(tick),
            updated_tick=int(tick),
            proposal_risk=str(proposal_risk or "none"),
            left_eye=left_snap,
            right_eye=right_snap,
            stimuli_signature=sig,
        )
        self._next_id += 1
        self._points.append(point)
        self._trim(tick)
        self._trim_eye_archives()
        return point

    def max_eye_similarity(
        self,
        left_eye: np.ndarray | None,
        right_eye: np.ndarray | None,
        *,
        kind: str | None = None,
        stimuli_signature: dict[str, Any] | None = None,
    ) -> tuple[float, InterestPoint | None]:
        """Best match of current eyes against archived acknowledge snapshots.

        Compares against pending/investigating/done points that still hold eye frames.
        Optionally prefers the same kind / similar stimuli signature.
        Returns (score in [0,1], matched point or None).
        """
        cfg = self.config
        left_snap = snapshot_eye(left_eye, size=int(cfg.eye_store_size))
        right_snap = snapshot_eye(right_eye, size=int(cfg.eye_store_size))
        if left_snap is None and right_snap is None:
            return 0.0, None

        kind_n = _normalize_kind(kind) if kind else None
        sig = _slim_stimuli(stimuli_signature)
        best = 0.0
        best_point: InterestPoint | None = None
        for point in self._points:
            if point.left_eye is None and point.right_eye is None:
                continue
            if kind_n is not None and point.kind != kind_n:
                continue
            # Same stimuli bag (wall_hazard / bitter / …) before trusting the image match.
            if sig and point.stimuli_signature:
                if _stimuli_overlap(sig, point.stimuli_signature) < 0.35:
                    continue
            score = eye_pair_similarity(left_snap, right_snap, point.left_eye, point.right_eye)
            if score > best:
                best = score
                best_point = point
        return float(best), best_point

    def _trim_eye_archives(self) -> None:
        """Cap how many points retain eye pixels (keep newest with eyes)."""
        with_eyes = [p for p in self._points if p.left_eye is not None or p.right_eye is not None]
        overflow = len(with_eyes) - int(self.config.max_eye_archives)
        if overflow <= 0:
            return
        with_eyes.sort(key=lambda p: p.updated_tick)
        for p in with_eyes[:overflow]:
            p.left_eye = None
            p.right_eye = None

    def expire_stale(self, tick: int) -> int:
        ttl = int(self.config.ttl_ticks)
        n = 0
        for p in self._points:
            if p.status == "pending" and tick - p.created_tick > ttl:
                p.status = "expired"
                p.updated_tick = int(tick)
                n += 1
        return n

    def pop_for_investigation(self, *, tick: int) -> list[InvestigationDirective]:
        """Pop up to N highest-priority pending points into investigation directives."""
        cfg = self.config
        if not cfg.enabled:
            return []
        self.expire_stale(tick)
        if self._active is not None and self._active.remaining > 0:
            return []  # finish current investigation first

        pending = sorted(
            (p for p in self._points if p.status == "pending"),
            key=lambda p: (p.priority, p.novelty, p.hits),
            reverse=True,
        )
        out: list[InvestigationDirective] = []
        limit = max(0, int(cfg.investigate_per_low_load))
        for point in pending[:limit]:
            point.status = "investigating"
            point.updated_tick = int(tick)
            turn = _clamp(point.bearing * float(cfg.investigate_turn_gain), -1.0, 1.0)
            # Walls: steer away / look along the wall rather than drive into it.
            if point.kind == "wall" and abs(turn) < 0.15:
                turn = 0.55 if point.bearing >= 0 else -0.55
            directive = InvestigationDirective(
                point=point,
                forward_scale=float(cfg.investigate_forward_scale),
                turn=float(turn),
                ttl_ticks=int(cfg.investigate_ttl_ticks),
                remaining=int(cfg.investigate_ttl_ticks),
                started_tick=int(tick),
            )
            out.append(directive)
            self._active = directive
            break  # one active investigation at a time
        return out

    def advance_tick(self, tick: int) -> InvestigationDirective | None:
        """Count down the active investigation; mark done when TTL expires."""
        active = self._active
        if active is None:
            return None
        # Don't burn TTL on the tick the investigation started.
        if int(tick) <= int(active.started_tick):
            return active
        active.remaining = max(0, int(active.remaining) - 1)
        if active.remaining <= 0:
            active.point.status = "done"
            active.point.updated_tick = int(tick)
            self._active = None
            return None
        return active

    def _trim(self, tick: int) -> None:
        self.expire_stale(tick)
        pending = [p for p in self._points if p.status == "pending"]
        overflow = len(pending) - int(self.config.max_pending)
        if overflow <= 0:
            return
        # Drop lowest priority first.
        pending.sort(key=lambda p: (p.priority, p.novelty, p.created_tick))
        for p in pending[:overflow]:
            p.status = "expired"
            p.updated_tick = int(tick)


def _slim_stimuli(raw: dict[str, Any] | None) -> dict[str, float]:
    if not raw:
        return {}
    keys = (
        "wall_hazard",
        "visual_novelty",
        "novelty_bearing",
        "bitter",
        "odor",
        "energy",
        "survival_risk",
        "sugar",
    )
    out: dict[str, float] = {}
    for key in keys:
        if key not in raw:
            continue
        try:
            out[key] = round(float(raw[key]), 3)
        except (TypeError, ValueError):
            continue
    return out


def _stimuli_overlap(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    shared = set(a) & set(b)
    if not shared:
        return 0.0
    diffs = [1.0 - min(1.0, abs(a[k] - b[k])) for k in shared]
    return float(sum(diffs) / len(diffs))


def apply_investigation_goal(goal: np.ndarray, directive: InvestigationDirective) -> np.ndarray:
    """Bias motor goal toward the deferred interest point."""
    g = np.asarray(goal, dtype=np.float32).reshape(-1).copy()
    if g.size < 5:
        padded = np.zeros(5, dtype=np.float32)
        padded[: g.size] = g
        g = padded
    g[0] = float(np.clip(float(g[0]) * float(directive.forward_scale), -1.0, 1.0))
    g[1] = float(np.clip(float(directive.turn), -1.0, 1.0))
    return g


def _normalize_kind(kind: str) -> InterestKind:
    k = str(kind or "other").strip().lower()
    if k in ("wall", "novelty", "toxin", "odor", "energy", "fall"):
        return k  # type: ignore[return-value]
    return "other"


def _kind_salience(kind: InterestKind) -> float:
    return {
        "toxin": 0.95,
        "fall": 0.85,
        "odor": 0.75,
        "wall": 0.7,
        "novelty": 0.65,
        "energy": 0.5,
        "other": 0.4,
    }.get(kind, 0.4)
