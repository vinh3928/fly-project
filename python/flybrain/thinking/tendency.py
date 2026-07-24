"""Developmental behavioral tendency: curiosity early, sex drive later; shaped by outcomes.

When the fly is in \"safe mode\" (no real forage demand — the old stop/feed loophole),
L1 defaults follow the dominant tendency instead of hard-wiring feed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


# Action affinities — no wall/object terms; pure drive→motor mapping.
ACTION_AFFINITY: dict[str, dict[str, float]] = {
    "curiosity": {
        "explore": 1.0,
        "forward": 0.75,
        "turn_left": 0.55,
        "turn_right": 0.55,
        "court": 0.25,
    },
    "sex": {
        "court": 1.0,
        "explore": 0.55,
        "turn_left": 0.45,
        "turn_right": 0.45,
        "forward": 0.35,
    },
    "forage": {
        "feed": 1.0,
        "forward": 0.55,
        "explore": 0.25,
    },
}


@dataclass
class TendencyConfig:
    enabled: bool = True
    # Developmental baselines (interpolated by maturity).
    juvenile_curiosity: float = 0.85
    juvenile_sex: float = 0.05
    juvenile_forage: float = 0.10
    adult_curiosity: float = 0.35
    adult_sex: float = 0.45
    adult_forage: float = 0.20
    # Learning from labeled outcomes.
    outcome_lr: float = 0.08
    # Soft L1 bias always; larger when safe-mode default would have been feed/stop.
    base_bonus: float = 0.06
    safe_mode_bonus: float = 0.30
    # Residual learned offset caps.
    max_offset: float = 0.45

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> TendencyConfig:
        data = dict(raw or {})
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class TendencyState:
    curiosity: float
    sex: float
    forage: float
    dominant: str
    maturity: float
    safe_mode: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "curiosity": float(self.curiosity),
            "sex": float(self.sex),
            "forage": float(self.forage),
            "dominant": self.dominant,
            "maturity": float(self.maturity),
            "safe_mode": bool(self.safe_mode),
        }


def _normalize(curiosity: float, sex: float, forage: float) -> tuple[float, float, float]:
    c, s, f = max(0.0, curiosity), max(0.0, sex), max(0.0, forage)
    total = c + s + f
    if total <= 1e-9:
        return 0.85, 0.05, 0.10
    return c / total, s / total, f / total


@dataclass
class TendencyDrive:
    """Tracks curiosity / sex / forage tendencies across development + outcomes."""

    config: TendencyConfig = field(default_factory=TendencyConfig)
    # Learned residual on top of developmental baseline (can go negative).
    offset_curiosity: float = 0.0
    offset_sex: float = 0.0
    offset_forage: float = 0.0

    def reset(self) -> None:
        self.offset_curiosity = 0.0
        self.offset_sex = 0.0
        self.offset_forage = 0.0

    def developmental_baseline(self, maturity: float) -> tuple[float, float, float]:
        cfg = self.config
        m = _clamp01(maturity)
        curiosity = (1.0 - m) * cfg.juvenile_curiosity + m * cfg.adult_curiosity
        sex = (1.0 - m) * cfg.juvenile_sex + m * cfg.adult_sex
        forage = (1.0 - m) * cfg.juvenile_forage + m * cfg.adult_forage
        return _normalize(curiosity, sex, forage)

    def levels(self, maturity: float) -> tuple[float, float, float]:
        b_c, b_s, b_f = self.developmental_baseline(maturity)
        cfg = self.config
        return _normalize(
            b_c + float(np.clip(self.offset_curiosity, -cfg.max_offset, cfg.max_offset)),
            b_s + float(np.clip(self.offset_sex, -cfg.max_offset, cfg.max_offset)),
            b_f + float(np.clip(self.offset_forage, -cfg.max_offset, cfg.max_offset)),
        )

    def state(self, *, maturity: float, safe_mode: bool = False) -> TendencyState:
        c, s, f = self.levels(maturity)
        dominant = "curiosity"
        if s >= c and s >= f:
            dominant = "sex"
        elif f >= c and f >= s:
            dominant = "forage"
        return TendencyState(
            curiosity=c,
            sex=s,
            forage=f,
            dominant=dominant,
            maturity=float(maturity),
            safe_mode=bool(safe_mode),
        )

    def affinity(self, action_name: str, *, maturity: float) -> float:
        c, s, f = self.levels(maturity)
        name = str(action_name)
        score = (
            c * float(ACTION_AFFINITY["curiosity"].get(name, 0.0))
            + s * float(ACTION_AFFINITY["sex"].get(name, 0.0))
            + f * float(ACTION_AFFINITY["forage"].get(name, 0.0))
        )
        return float(score)

    @staticmethod
    def is_safe_mode(
        signals: dict[str, Any],
        peaks: list[str] | None = None,
    ) -> bool:
        """Ambivalent / no real forage demand — where stop↔feed used to dominate."""
        peaks = list(peaks or [])
        energy = float(signals.get("energy", 0.5))
        sugar = float(signals.get("sugar", 0.0))
        bitter = float(signals.get("bitter", 0.0))
        if "sugar" in peaks or "low_energy" in peaks:
            return False
        if sugar >= 0.25:
            return False
        if energy < 0.45:
            return False
        if bitter >= 0.35 or "bitter" in peaks:
            return False
        return True

    def apply_bonus(
        self,
        *,
        action_name: str,
        value: float,
        maturity: float,
        safe_mode: bool,
    ) -> float:
        if not self.config.enabled:
            return float(value)
        aff = self.affinity(action_name, maturity=maturity)
        bonus = float(self.config.base_bonus) * aff
        if safe_mode:
            bonus += float(self.config.safe_mode_bonus) * aff
            # Explicitly demote the old feed loophole when there is nothing to eat.
            if action_name == "feed":
                bonus -= 0.5 * float(self.config.safe_mode_bonus)
            if action_name == "stop":
                bonus -= 0.25 * float(self.config.safe_mode_bonus)
        return float(value + bonus)

    def update_from_outcome(
        self,
        *,
        action_name: str,
        outcome_score: float,
        kind: str | None = None,
    ) -> None:
        """Nudge tendency residuals from a labeled transition outcome."""
        if not self.config.enabled:
            return
        lr = float(self.config.outcome_lr)
        score = float(outcome_score)
        if kind == "desirable":
            score = max(score, 0.15)
        elif kind == "undesirable":
            score = min(score, -0.15)
        name = str(action_name)
        # Credit the single best-matching drive for this action (avoid cross-talk).
        best_drive = None
        best_w = 0.0
        for drive, table in ACTION_AFFINITY.items():
            weight = float(table.get(name, 0.0))
            if weight > best_w:
                best_w = weight
                best_drive = drive
        if best_drive is None or best_w <= 1e-9:
            return
        delta = lr * score * best_w
        if best_drive == "curiosity":
            self.offset_curiosity = float(
                np.clip(self.offset_curiosity + delta, -self.config.max_offset, self.config.max_offset)
            )
        elif best_drive == "sex":
            self.offset_sex = float(
                np.clip(self.offset_sex + delta, -self.config.max_offset, self.config.max_offset)
            )
        elif best_drive == "forage":
            self.offset_forage = float(
                np.clip(self.offset_forage + delta, -self.config.max_offset, self.config.max_offset)
            )

    def preferred_actions(self, *, maturity: float, top_k: int = 3) -> list[str]:
        names = sorted(
            {n for table in ACTION_AFFINITY.values() for n in table},
            key=lambda n: self.affinity(n, maturity=maturity),
            reverse=True,
        )
        return names[: max(1, top_k)]
