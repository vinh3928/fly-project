"""Outcome labeling, spiritual growth, and life-experience priority weights."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

OutcomeKind = Literal["desirable", "undesirable", "neutral"]


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def smoothstep(x: float) -> float:
    """Hermite smoothstep on [0, 1]."""
    t = _clamp01(x)
    return t * t * (3.0 - 2.0 * t)


@dataclass
class PriorityWeights:
    survival: float
    altruism: float
    spiritual: float

    def as_dict(self) -> dict[str, float]:
        return {
            "survival": self.survival,
            "altruism": self.altruism,
            "spiritual": self.spiritual,
        }


@dataclass
class LifeExperience:
    """Tracks age and labeled data that unlock advanced priorities."""

    lifetime_steps: int = 0
    labeled_outcomes: int = 0
    episodes: int = 0
    age_horizon: int = 500
    data_horizon: int = 100
    max_altruism: float = 0.35
    max_spiritual: float = 0.35
    min_survival: float = 0.3

    def tick_step(self) -> None:
        self.lifetime_steps += 1

    def record_label(self, n: int = 1) -> None:
        self.labeled_outcomes += max(0, int(n))

    def end_episode(self) -> None:
        self.episodes += 1

    def maturity(self) -> float:
        age = smoothstep(self.lifetime_steps / max(1, self.age_horizon))
        data = smoothstep(self.labeled_outcomes / max(1, self.data_horizon))
        return float(age * data)

    def priority_weights(self, *, survival_risk: float = 0.0) -> PriorityWeights:
        """Beginners: survival-only. Advanced: mix altruism + spiritual growth.

        High survival_risk always boosts survival weight so advanced values never
        fully override urgent homeostasis.
        """
        m = self.maturity()
        altruism = self.max_altruism * m
        spiritual = self.max_spiritual * m
        survival = 1.0 - altruism - spiritual
        # Urgency floor under threat
        urgency = _clamp01(survival_risk)
        survival = max(survival, self.min_survival + 0.5 * urgency)
        # Renormalize remaining mass into advanced axes
        rem = max(0.0, 1.0 - survival)
        adv = altruism + spiritual
        if adv > 1e-9 and rem > 0.0:
            scale = rem / adv
            altruism *= scale
            spiritual *= scale
        else:
            altruism = 0.0
            spiritual = rem * 0.5 if rem > 0 else 0.0
            survival = 1.0 - altruism - spiritual
        total = survival + altruism + spiritual
        if total > 1e-9:
            survival /= total
            altruism /= total
            spiritual /= total
        return PriorityWeights(survival=survival, altruism=altruism, spiritual=spiritual)

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None = None) -> LifeExperience:
        c = cfg or {}
        return cls(
            age_horizon=int(c.get("age_horizon", 500)),
            data_horizon=int(c.get("data_horizon", 100)),
            max_altruism=float(c.get("max_altruism", 0.35)),
            max_spiritual=float(c.get("max_spiritual", 0.35)),
            min_survival=float(c.get("min_survival", 0.3)),
        )


@dataclass
class OutcomeLabel:
    kind: OutcomeKind
    score: float
    spiritual_delta: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "score": self.score,
            "spiritual_delta": self.spiritual_delta,
            "reasons": list(self.reasons),
        }


def spiritual_growth_score(
    *,
    affect: dict[str, Any],
    prev_affect: dict[str, Any] | None = None,
    novelty: float | None = None,
    learning_progress: float = 0.0,
    weights: dict[str, float] | None = None,
) -> float:
    """Novelty/learning-progress + affective coherence (1C)."""
    w = {
        "novelty": 0.25,
        "learning": 0.2,
        "coherence_pos": 0.35,
        "coherence_neg": 0.2,
        **(weights or {}),
    }
    nov = float(novelty if novelty is not None else affect.get("novelty", affect.get("exploration", 0.0)))
    comfort = float(affect.get("comfort", 0.0))
    trust = float(affect.get("trust", 0.0))
    altruism = float(affect.get("altruism", 0.0))
    exploration = float(affect.get("exploration", 0.0))
    fear = float(affect.get("fear", 0.0))
    pain = float(affect.get("pain", 0.0))
    survival = float(affect.get("survival_risk", 0.0))

    pos = 0.35 * comfort + 0.25 * trust + 0.25 * altruism + 0.15 * exploration
    neg = 0.4 * fear + 0.3 * pain + 0.3 * survival

    if prev_affect is not None:
        prev_pos = (
            0.35 * float(prev_affect.get("comfort", 0.0))
            + 0.25 * float(prev_affect.get("trust", 0.0))
            + 0.25 * float(prev_affect.get("altruism", 0.0))
            + 0.15 * float(prev_affect.get("exploration", 0.0))
        )
        prev_neg = (
            0.4 * float(prev_affect.get("fear", 0.0))
            + 0.3 * float(prev_affect.get("pain", 0.0))
            + 0.3 * float(prev_affect.get("survival_risk", 0.0))
        )
        coherence = 0.5 * (pos - prev_pos) - 0.5 * (neg - prev_neg) + 0.5 * (pos - neg)
    else:
        coherence = pos - neg

    raw = (
        w["novelty"] * nov
        + w["learning"] * _clamp01(learning_progress)
        + w["coherence_pos"] * _clamp01(0.5 + 0.5 * coherence)
        - w["coherence_neg"] * _clamp01(neg)
    )
    return _clamp01(0.5 + 0.5 * raw) if raw < 0 else _clamp01(raw)


def label_transition(
    *,
    prev_signals: dict[str, Any],
    prev_affect: dict[str, Any],
    next_signals: dict[str, Any],
    next_affect: dict[str, Any],
    learning_progress: float = 0.0,
) -> OutcomeLabel:
    """Label a transition desirable / undesirable / neutral from Δ signals + affect."""
    reasons: list[str] = []
    score = 0.0

    d_energy = float(next_signals.get("energy", 0.5)) - float(prev_signals.get("energy", 0.5))
    d_sugar = float(next_signals.get("sugar", 0.0)) - float(prev_signals.get("sugar", 0.0))
    bitter = float(next_signals.get("bitter", 0.0))
    damage = float(next_signals.get("damage", next_affect.get("pain", 0.0)))
    d_comfort = float(next_affect.get("comfort", 0.0)) - float(prev_affect.get("comfort", 0.0))
    d_survival = float(next_affect.get("survival_risk", 0.0)) - float(prev_affect.get("survival_risk", 0.0))
    sugar_consumed = float(next_signals.get("sugar_consumed", 0.0)) - float(
        prev_signals.get("sugar_consumed", 0.0)
    )

    if d_energy > 0.02:
        score += 0.35 * d_energy
        reasons.append("energy_up")
    if d_energy < -0.02:
        score += 0.4 * d_energy
        reasons.append("energy_down")
    if sugar_consumed > 0.0 or d_sugar > 0.05:
        score += 0.3
        reasons.append("sugar_gain")
    if bitter > 0.4:
        score -= 0.45 * bitter
        reasons.append("bitter")
    if damage > 0.3:
        score -= 0.4 * damage
        reasons.append("damage")
    if d_comfort > 0.02:
        score += 0.2 * d_comfort
        reasons.append("comfort_up")
    if d_survival > 0.05:
        score -= 0.35 * d_survival
        reasons.append("survival_worse")
    if d_survival < -0.05:
        score += 0.25 * (-d_survival)
        reasons.append("survival_better")

    growth_prev = spiritual_growth_score(affect=prev_affect)
    growth_next = spiritual_growth_score(
        affect=next_affect,
        prev_affect=prev_affect,
        learning_progress=learning_progress,
    )
    spiritual_delta = growth_next - growth_prev
    score += 0.15 * spiritual_delta

    if score > 0.08:
        kind: OutcomeKind = "desirable"
    elif score < -0.08:
        kind = "undesirable"
    else:
        kind = "neutral"
    return OutcomeLabel(
        kind=kind,
        score=float(score),
        spiritual_delta=float(spiritual_delta),
        reasons=reasons,
    )


def priority_value(
    *,
    weights: PriorityWeights,
    survival_utility: float,
    altruism_utility: float,
    spiritual_utility: float,
) -> float:
    return float(
        weights.survival * survival_utility
        + weights.altruism * altruism_utility
        + weights.spiritual * spiritual_utility
    )
