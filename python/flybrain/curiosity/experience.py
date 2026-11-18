"""Experience novelty: how far the current blended embedding is from remembered ones.

Each embedding is normalized per dimension (running mean / spread), compared by
mean distance to the nearest remembered embeddings in that normalized space, and
that distance is scaled against the run's own distance history (median → 0,
95th pct → 1).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from flybrain.phase8 import (
    EXPERIENCE_BASELINE_PERCENTILE,
    EXPERIENCE_DIMS,
    EXPERIENCE_HISTORY_SIZE,
    EXPERIENCE_KNN,
    EXPERIENCE_MEMORY_SIZE,
    EXPERIENCE_MIN_STD,
    EXPERIENCE_PERCENTILE,
    EXPERIENCE_RECENT_EXCLUDE,
    EXPERIENCE_WARMUP,
)

_EPS = 1e-8


@dataclass
class ExperienceNoveltyConfig:
    dims: int = EXPERIENCE_DIMS
    memory_size: int = EXPERIENCE_MEMORY_SIZE
    knn: int = EXPERIENCE_KNN
    recent_exclude: int = EXPERIENCE_RECENT_EXCLUDE
    history_size: int = EXPERIENCE_HISTORY_SIZE
    percentile: float = EXPERIENCE_PERCENTILE
    baseline_percentile: float = EXPERIENCE_BASELINE_PERCENTILE
    warmup: int = EXPERIENCE_WARMUP
    min_std: float = EXPERIENCE_MIN_STD

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> ExperienceNoveltyConfig:
        data = dict(raw or {})
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ExperienceNoveltyResult:
    novelty: float
    distance: float | None
    memory: int


class RunningStats:
    """Per-dimension running mean and spread (Welford)."""

    def __init__(self, dims: int, min_std: float = EXPERIENCE_MIN_STD) -> None:
        self.dims = int(dims)
        self.min_std = float(min_std)
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.mean = np.zeros(self.dims, dtype=np.float64)
        self._m2 = np.zeros(self.dims, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self._m2 += delta * (x - self.mean)

    @property
    def std(self) -> np.ndarray:
        var = self._m2 / max(1, self.count)
        return np.maximum(np.sqrt(var), self.min_std)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.std


class ExperienceNovelty:
    """Rolling memory of embeddings scored by normalized nearest-neighbour distance."""

    def __init__(self, config: ExperienceNoveltyConfig | None = None) -> None:
        self.config = config or ExperienceNoveltyConfig()
        cfg = self.config
        self._stats = RunningStats(cfg.dims, cfg.min_std)
        self._memory = np.zeros((max(1, int(cfg.memory_size)), cfg.dims), dtype=np.float32)
        self._history: deque[float] = deque(maxlen=max(1, int(cfg.history_size)))
        self._written = 0

    def reset(self) -> None:
        self._stats.reset()
        self._memory.fill(0.0)
        self._history.clear()
        self._written = 0

    @property
    def memory_count(self) -> int:
        return min(self._written, self._memory.shape[0])

    def observe(self, embedding: np.ndarray) -> ExperienceNoveltyResult:
        x = self._take(embedding)
        self._stats.update(x)
        distance = self._distance(self._stats.normalize(x))
        novelty = 0.0
        if distance is not None:
            novelty = self._scale(distance)
            self._history.append(distance)
        self._memory[self._written % self._memory.shape[0]] = x
        self._written += 1
        return ExperienceNoveltyResult(novelty=novelty, distance=distance, memory=self.memory_count)

    def _take(self, embedding: np.ndarray) -> np.ndarray:
        vec = np.asarray(embedding, dtype=np.float64).reshape(-1)
        out = np.zeros(self.config.dims, dtype=np.float64)
        n = min(self.config.dims, vec.size)
        out[:n] = vec[:n]
        return out

    def _past(self) -> np.ndarray:
        """Remembered embeddings, minus the most recent ``recent_exclude`` entries."""
        size = self._memory.shape[0]
        stored = self.memory_count
        skip = min(stored, max(0, int(self.config.recent_exclude)))
        keep = np.ones(stored, dtype=bool)
        for i in range(skip):
            keep[(self._written - 1 - i) % size] = False
        return self._memory[:stored][keep]

    def _distance(self, z: np.ndarray) -> float | None:
        """Mean RMS z-score distance to the ``knn`` nearest remembered embeddings."""
        past = self._past()
        if past.shape[0] == 0:
            return None
        diffs = self._stats.normalize(past) - z
        dists = np.sqrt(np.mean(diffs * diffs, axis=1))
        k = min(max(1, int(self.config.knn)), dists.size)
        return float(np.mean(np.partition(dists, k - 1)[:k]))

    def _scale(self, distance: float) -> float:
        if len(self._history) < max(1, int(self.config.warmup)):
            return 0.0
        lo, hi = np.percentile(
            np.asarray(self._history, dtype=np.float64),
            [self.config.baseline_percentile, self.config.percentile],
        )
        if hi - lo <= _EPS:
            return 1.0 if distance > hi else 0.0
        return float(np.clip((distance - lo) / (hi - lo), 0.0, 1.0))
