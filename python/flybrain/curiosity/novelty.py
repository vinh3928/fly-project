"""Novelty tracking: first-visit bonuses, coverage, diminishing returns, exploration XP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from flybrain.phase8 import (
    CHEM_SIGNATURE_BINS,
    EMBEDDING_HASH_BINS,
    FIRST_VISIT_BONUS,
    GRID_CELL_SIZE,
    REPEAT_VISIT_DECAY,
)


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


@dataclass
class NoveltyConfig:
    grid_cell_size: float = GRID_CELL_SIZE
    first_visit_bonus: float = FIRST_VISIT_BONUS
    repeat_decay: float = REPEAT_VISIT_DECAY
    embedding_bins: int = EMBEDDING_HASH_BINS
    chem_bins: int = CHEM_SIGNATURE_BINS
    arena_half_extent: float = 0.5


@dataclass
class NoveltyObservation:
    """Result of observing one step for novelty / progression."""

    novelty_score: float
    first_visit: bool
    spatial_new: bool
    embedding_new: bool
    chem_new: bool
    event_new: bool
    coverage: float
    unique_cells: int
    exploration_xp: float
    streak: int
    details: dict[str, Any] = field(default_factory=dict)


class NoveltyTracker:
    """Gamified exploration counters with diminishing returns on repeats."""

    def __init__(self, config: NoveltyConfig | None = None) -> None:
        self.config = config or NoveltyConfig()
        self._cell_visits: dict[tuple[int, int], int] = {}
        self._embed_visits: dict[tuple[int, ...], int] = {}
        self._chem_visits: dict[tuple[int, ...], int] = {}
        self._events: set[str] = set()
        self.exploration_xp: float = 0.0
        self.streak: int = 0
        self.total_steps: int = 0

    def reset(self) -> None:
        self._cell_visits.clear()
        self._embed_visits.clear()
        self._chem_visits.clear()
        self._events.clear()
        self.exploration_xp = 0.0
        self.streak = 0
        self.total_steps = 0

    def _cell_key(self, position: np.ndarray | None) -> tuple[int, int] | None:
        if position is None:
            return None
        pos = np.asarray(position, dtype=np.float32).reshape(-1)
        if pos.size < 2:
            return None
        size = max(1e-6, float(self.config.grid_cell_size))
        return (int(np.floor(float(pos[0]) / size)), int(np.floor(float(pos[1]) / size)))

    def _hash_vector(self, vec: np.ndarray | None, bins: int, dims: int = 8) -> tuple[int, ...] | None:
        if vec is None:
            return None
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return None
        take = arr[: min(dims, arr.size)]
        # Map roughly [-inf,inf]/[0,1] then quantize.
        normed = 0.5 * (np.tanh(take) + 1.0)
        quantized = np.clip((normed * bins).astype(np.int32), 0, bins - 1)
        return tuple(int(x) for x in quantized.tolist())

    def _visit_bonus(self, count_before: int) -> float:
        """First visit pays full bonus; repeats decay."""
        if count_before <= 0:
            return float(self.config.first_visit_bonus)
        return float(self.config.first_visit_bonus) * (
            float(self.config.repeat_decay) ** float(count_before)
        )

    def possible_cells(self) -> int:
        half = float(self.config.arena_half_extent)
        size = max(1e-6, float(self.config.grid_cell_size))
        n = int(np.floor(half / size)) * 2 + 1
        return max(1, n * n)

    def coverage(self) -> float:
        return float(len(self._cell_visits)) / float(self.possible_cells())

    def observe(
        self,
        *,
        position: np.ndarray | None = None,
        embedding: np.ndarray | None = None,
        chem_vector: np.ndarray | None = None,
        events: list[str] | None = None,
    ) -> NoveltyObservation:
        self.total_steps += 1
        spatial_new = False
        embedding_new = False
        chem_new = False
        event_new = False
        score = 0.0
        details: dict[str, Any] = {}

        cell = self._cell_key(position)
        if cell is not None:
            before = self._cell_visits.get(cell, 0)
            bonus = self._visit_bonus(before)
            self._cell_visits[cell] = before + 1
            score += 0.45 * bonus
            spatial_new = before == 0
            details["cell"] = cell
            details["cell_visits"] = before + 1

        embed_key = self._hash_vector(embedding, self.config.embedding_bins, dims=8)
        if embed_key is not None:
            before = self._embed_visits.get(embed_key, 0)
            bonus = self._visit_bonus(before)
            self._embed_visits[embed_key] = before + 1
            score += 0.3 * bonus
            embedding_new = before == 0
            details["embed_visits"] = before + 1

        chem_key = self._hash_vector(chem_vector, self.config.chem_bins, dims=4)
        if chem_key is not None:
            before = self._chem_visits.get(chem_key, 0)
            bonus = self._visit_bonus(before)
            self._chem_visits[chem_key] = before + 1
            score += 0.15 * bonus
            chem_new = before == 0
            details["chem_visits"] = before + 1

        for name in events or []:
            if name and name not in self._events:
                self._events.add(name)
                score += 0.35 * float(self.config.first_visit_bonus)
                event_new = True
                details.setdefault("new_events", []).append(name)

        first_visit = spatial_new or embedding_new or chem_new or event_new
        if first_visit:
            self.streak += 1
            self.exploration_xp += max(0.05, float(score))
        else:
            self.streak = 0
            # Tiny XP drip for continued roaming so streaks matter more than camping.
            self.exploration_xp += 0.01 * float(score)

        novelty_score = _clamp01(score)
        return NoveltyObservation(
            novelty_score=novelty_score,
            first_visit=first_visit,
            spatial_new=spatial_new,
            embedding_new=embedding_new,
            chem_new=chem_new,
            event_new=event_new,
            coverage=self.coverage(),
            unique_cells=len(self._cell_visits),
            exploration_xp=float(self.exploration_xp),
            streak=int(self.streak),
            details=details,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "coverage": self.coverage(),
            "unique_cells": len(self._cell_visits),
            "unique_embeddings": len(self._embed_visits),
            "unique_chem": len(self._chem_visits),
            "unique_events": len(self._events),
            "exploration_xp": float(self.exploration_xp),
            "streak": int(self.streak),
            "steps": int(self.total_steps),
        }
