"""Generic L1 sample-and-diff gate for any modality.

Samples a feature vector on a time (and/or step) cadence, compares it to the
last stored reading with a simple diff, and reports whether the expensive path
(STM / L2 / Neo4j) should open.

If the sample is unchanged → ``l1_hit=True`` and ``active=False``: stay on the
cheap L1 baseline, no round-trip. Works for smell, vision patches, sensors, or
any other fixed-size vector modality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

CompareMode = Literal["max_abs", "mean_abs", "l2", "cosine_distance"]


def _now_seconds(*, now_s: float | None, now_ms: float | None) -> float:
    if now_s is not None:
        return float(now_s)
    if now_ms is not None:
        return float(now_ms) / 1000.0
    return 0.0


def pad_modal_vec(vec: np.ndarray | None, dim: int | None) -> np.ndarray:
    """Flatten and optionally pad/truncate a modality vector."""
    if vec is None:
        if dim is None:
            return np.zeros(0, dtype=np.float32)
        return np.zeros(int(dim), dtype=np.float32)
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if dim is None:
        return v.copy()
    d = int(dim)
    if v.size == d:
        return v.copy()
    out = np.zeros(d, dtype=np.float32)
    n = min(d, int(v.size))
    if n > 0:
        out[:n] = v[:n]
    return out


def modal_delta(
    cur: np.ndarray,
    prev: np.ndarray,
    *,
    mode: CompareMode = "max_abs",
) -> float:
    """Scalar distance between two modality vectors."""
    a = np.asarray(cur, dtype=np.float32).reshape(-1)
    b = np.asarray(prev, dtype=np.float32).reshape(-1)
    if a.size != b.size:
        n = max(a.size, b.size)
        aa = np.zeros(n, dtype=np.float32)
        bb = np.zeros(n, dtype=np.float32)
        aa[: a.size] = a
        bb[: b.size] = b
        a, b = aa, bb
    if a.size == 0:
        return 0.0
    diff = a - b
    if mode == "mean_abs":
        return float(np.mean(np.abs(diff)))
    if mode == "l2":
        return float(np.linalg.norm(diff))
    if mode == "cosine_distance":
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-8 or nb < 1e-8:
            return 1.0 if na + nb > 1e-8 else 0.0
        return float(1.0 - np.dot(a, b) / (na * nb))
    # max_abs (default): cheap, channel-wise spike friendly
    return float(np.max(np.abs(diff)))


@dataclass
class ModalGateConfig:
    """Settings for a reusable modality change gate."""

    modality: str = "generic"
    # Sample cadence: wall-clock L1 window (default 10s) and/or step interval.
    ttl_seconds: float = 10.0
    step_interval: int | None = None
    change_min: float = 0.05
    # Forced reopen of the expensive path even when the sample is unchanged.
    refresh_seconds: float | None = None
    compare: CompareMode = "max_abs"
    dim: int | None = None
    # First non-zero reading counts as a change (smell/vision presence).
    first_nonzero_is_change: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> ModalGateConfig:
        data = dict(raw or {})
        compare = str(data.get("compare", "max_abs"))
        if compare not in ("max_abs", "mean_abs", "l2", "cosine_distance"):
            compare = "max_abs"
        step_raw = data.get("step_interval", None)
        refresh_raw = data.get("refresh_seconds", data.get("refresh_ms"))
        refresh: float | None
        if refresh_raw is None:
            refresh = None
        elif "refresh_seconds" in data:
            refresh = float(refresh_raw)
        else:
            # refresh_ms alias
            refresh = float(refresh_raw) / 1000.0
        return cls(
            modality=str(data.get("modality", "generic")),
            ttl_seconds=float(data.get("ttl_seconds", 10.0)),
            step_interval=(None if step_raw is None else max(1, int(step_raw))),
            change_min=float(data.get("change_min", 0.05)),
            refresh_seconds=refresh,
            compare=compare,  # type: ignore[arg-type]
            dim=(None if data.get("dim") is None else int(data["dim"])),
            first_nonzero_is_change=bool(data.get("first_nonzero_is_change", True)),
        )


@dataclass
class ModalSample:
    """One sample-and-diff observation (no STM / Neo4j access)."""

    modality: str = "generic"
    sampled: bool = False
    changed: bool = False
    l1_hit: bool = False
    active: bool = False
    delta: float = 0.0
    reason: str = ""
    age_seconds: float = 0.0


@dataclass
class ModalChangeGate:
    """Cheap always-on watcher: sample → diff → stay L1 or open expensive path.

    Typical use::

        gate = ModalChangeGate.from_config({"modality": "smell", "ttl_seconds": 10})
        sample = gate.observe(smell_vec, now_ms=clock_ms)
        if sample.l1_hit:
            # unchanged — skip STM / L2 / Neo4j
            ...
        if sample.active:
            # run expensive probe
            ...
    """

    modality: str = "generic"
    ttl_seconds: float = 10.0
    step_interval: int | None = None
    change_min: float = 0.05
    refresh_seconds: float | None = None
    compare: CompareMode = "max_abs"
    dim: int | None = None
    first_nonzero_is_change: bool = True
    _last: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_sample_s: float | None = field(default=None, init=False, repr=False)
    _last_active_s: float | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_config(
        cls,
        config: ModalGateConfig | dict[str, Any] | None = None,
    ) -> ModalChangeGate:
        cfg = config if isinstance(config, ModalGateConfig) else ModalGateConfig.from_dict(config)
        return cls(
            modality=str(cfg.modality),
            ttl_seconds=float(cfg.ttl_seconds),
            step_interval=cfg.step_interval,
            change_min=float(cfg.change_min),
            refresh_seconds=cfg.refresh_seconds,
            compare=cfg.compare,
            dim=cfg.dim,
            first_nonzero_is_change=bool(cfg.first_nonzero_is_change),
        )

    def reset(self) -> None:
        self._last = None
        self._last_sample_s = None
        self._last_active_s = None

    @property
    def last(self) -> np.ndarray | None:
        return None if self._last is None else self._last.copy()

    def _should_sample(self, *, step: int | None, now_s: float) -> bool:
        if self._last_sample_s is None:
            return True
        ttl = float(self.ttl_seconds)
        if ttl > 0.0 and (now_s - float(self._last_sample_s)) >= ttl:
            return True
        interval = self.step_interval
        if interval is not None and step is not None:
            if int(step) % max(1, int(interval)) == 0:
                return True
        return False

    def observe(
        self,
        vec: np.ndarray | None,
        *,
        now_s: float | None = None,
        now_ms: float | None = None,
        step: int | None = None,
        force_open: bool = False,
        force_reason: str = "force",
    ) -> ModalSample:
        """Sample on cadence; if unchanged stay L1; else mark expensive path active."""
        t = _now_seconds(now_s=now_s, now_ms=now_ms)
        out = ModalSample(modality=self.modality)
        if self._last_sample_s is not None:
            out.age_seconds = max(0.0, t - float(self._last_sample_s))

        if self._should_sample(step=step, now_s=t):
            out.sampled = True
            cur = pad_modal_vec(vec, self.dim)
            if self._last is None:
                peak = float(np.max(np.abs(cur))) if cur.size else 0.0
                out.delta = peak
                if self.first_nonzero_is_change:
                    out.changed = peak > 1e-6
                else:
                    out.changed = True
            else:
                out.delta = modal_delta(cur, self._last, mode=self.compare)
                out.changed = out.delta >= float(self.change_min)
            # Update L1 baseline only on change (or first sample) so slow drift
            # accumulates against the last meaningful reading.
            if out.changed or self._last is None:
                self._last = cur
            else:
                out.l1_hit = True
            self._last_sample_s = t

        refresh = self.refresh_seconds
        due = refresh is not None and (
            self._last_active_s is None or (t - float(self._last_active_s)) >= float(refresh)
        )
        out.active = bool(out.changed or force_open or due)
        if out.changed:
            out.reason = f"{self.modality}_change"
        elif force_open:
            out.reason = str(force_reason or "force")
        elif due:
            out.reason = "timer"
        if out.active:
            self._last_active_s = t
        return out
