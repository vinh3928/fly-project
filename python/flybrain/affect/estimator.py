"""Rule-based affective factor estimation from sensory signals (Phase 4 v1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from flybrain.phase4 import AFFECT_DIM, AFFECT_KEYS
from flybrain.signals.builder import SensorySignals


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


@dataclass
class AffectState:
    """Emotion / drive scores in [0, 1], fixed key order for embedding concat."""

    scores: dict[str, float]

    def as_vector(self) -> np.ndarray:
        return np.asarray([self.scores.get(k, 0.0) for k in AFFECT_KEYS], dtype=np.float32)

    def to_jsonable(self) -> dict[str, Any]:
        return {k: float(self.scores.get(k, 0.0)) for k in AFFECT_KEYS}


class AffectEstimator:
    """Maps signal summaries to surprise/joy/sad/anger/... plus survival drives."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = {
            "sugar_to_joy": 1.0,
            "bitter_to_anger": 0.8,
            "bitter_to_fear": 0.6,
            "energy_to_survival": 1.0,
            "instability_to_surprise": 1.0,
            "hunger_to_sad": 0.5,
            "damage_to_fear": 0.7,
            **(weights or {}),
        }

    def estimate(self, signals: SensorySignals) -> AffectState:
        s = signals.summary
        sugar = float(s.get("sugar", 0.0))
        bitter = float(s.get("bitter", 0.0))
        energy = float(s.get("energy", 0.0))
        hunger = float(s.get("hunger", 0.0))
        damage = float(s.get("damage", 0.0))
        instability = float(s.get("instability", 0.0))
        contact = float(s.get("mean_contact", 0.0))
        wall_hazard = float(s.get("wall_hazard", 0.0))
        target_confidence = float(s.get("target_confidence", 0.0))

        joy = _clamp01(
            self.weights["sugar_to_joy"] * sugar
            + 0.35 * target_confidence
            + 0.25 * float(s.get("proboscis_contact", 0.0)) * sugar
        )
        anger = _clamp01(self.weights["bitter_to_anger"] * bitter)
        fear = _clamp01(
            self.weights["bitter_to_fear"] * bitter
            + self.weights["damage_to_fear"] * damage
            + 0.55 * wall_hazard
        )
        sad = _clamp01(self.weights["hunger_to_sad"] * hunger * 1.2 + (1.0 - energy) * 0.35)
        surprise = _clamp01(
            self.weights["instability_to_surprise"] * instability + bitter * 0.2 + 0.4 * wall_hazard
        )
        disgust = _clamp01(bitter * 1.0)
        energy_threat = _clamp01((1.0 - energy) * self.weights["energy_to_survival"])
        survival_risk = _clamp01(
            0.5 * energy_threat + 0.3 * damage + 0.2 * fear + 0.15 * wall_hazard
        )
        pain = _clamp01(damage)
        comfort = _clamp01(
            0.5 * joy
            + 0.3 * contact
            + 0.35 * energy
            - 0.4 * bitter
            - 0.3 * wall_hazard
            - 0.2 * hunger
        )
        trust = _clamp01(0.4 * joy + 0.3 * contact - 0.5 * bitter)
        anticipation = _clamp01(
            0.55 * hunger + 0.35 * sugar + 0.2 * surprise + 0.45 * target_confidence
        )
        altruism = _clamp01(0.3 * trust + 0.2 * joy - 0.4 * survival_risk)
        social_drive = _clamp01(0.2 * trust + 0.2 * anticipation)
        exploration = _clamp01(0.4 * surprise + 0.3 * anticipation + 0.2 * (1.0 - fear))
        novelty = _clamp01(0.5 * surprise + 0.3 * exploration)

        scores = {
            "surprise": surprise,
            "joy": joy,
            "sad": sad,
            "anger": anger,
            "fear": fear,
            "disgust": disgust,
            "trust": trust,
            "anticipation": anticipation,
            "altruism": altruism,
            "survival_risk": survival_risk,
            "energy_threat": energy_threat,
            "social_drive": social_drive,
            "exploration": exploration,
            "comfort": comfort,
            "pain": pain,
            "novelty": novelty,
        }
        assert len(scores) == AFFECT_DIM
        return AffectState(scores=scores)
