"""Two-layer present/future reasoning: quick priority scan + outcome reflection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from flybrain.phase4 import HIGH_LEVEL_ACTION_DIM
from flybrain.world_model.outcomes import (
    LifeExperience,
    OutcomeLabel,
    PriorityWeights,
    label_transition,
    priority_value,
    spiritual_growth_score,
)
from flybrain.world_model.predictor import NextStateWorldModel


# Discrete candidate high-level actions for Layer-1 scan.
_CANDIDATES: list[tuple[str, np.ndarray]] = [
    ("forward", np.asarray([0.8, 0.0, 0.5, 0.5, 0.0], dtype=np.float32)),
    ("turn_left", np.asarray([0.2, -0.7, 0.5, 0.5, 0.0], dtype=np.float32)),
    ("turn_right", np.asarray([0.2, 0.7, 0.5, 0.5, 0.0], dtype=np.float32)),
    ("stop", np.asarray([0.0, 0.0, 0.5, 0.3, 0.0], dtype=np.float32)),
    ("feed", np.asarray([0.15, 0.0, 0.5, 0.4, 0.9], dtype=np.float32)),
    ("explore", np.asarray([0.45, 0.25, 0.5, 0.7, 0.0], dtype=np.float32)),
]


@dataclass
class Layer1Result:
    name: str
    action: np.ndarray
    value: float
    survival_u: float
    altruism_u: float
    spiritual_u: float
    improves: bool


@dataclass
class FuturePrediction:
    present: str
    future_hypothesis: str
    chosen_action: np.ndarray
    chosen_name: str
    layer1_values: list[dict[str, Any]] = field(default_factory=list)
    layer2_override: bool = False
    layer2_reason: str = ""
    confidence: float = 0.5
    priority_weights: dict[str, float] = field(default_factory=dict)
    outcome_hint: dict[str, Any] | None = None

    def as_present_future(self) -> dict[str, str]:
        return {"present": self.present, "future_hypothesis": self.future_hypothesis}


def _stimulus_peaks(signals: dict[str, Any], affect: dict[str, Any]) -> list[str]:
    peaks: list[str] = []
    if float(signals.get("bitter", 0.0)) > 0.35:
        peaks.append("bitter")
    if float(signals.get("sugar", 0.0)) > 0.3:
        peaks.append("sugar")
    if float(signals.get("wall_hazard", 0.0)) > 0.4:
        peaks.append("wall")
    if float(affect.get("survival_risk", 0.0)) > 0.55:
        peaks.append("survival")
    if float(signals.get("energy", 0.5)) < 0.4:
        peaks.append("low_energy")
    if float(affect.get("novelty", affect.get("exploration", 0.0))) > 0.55:
        peaks.append("novelty")
    return peaks


def _heuristic_next_state(
    *,
    signals: dict[str, Any],
    affect: dict[str, Any],
    action: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Deterministic Δ-state for tests / no checkpoint (hybrid fallback)."""
    forward, turn, _h, _f, proboscis = [float(x) for x in action[:5]]
    sugar = float(signals.get("sugar", 0.0))
    bitter = float(signals.get("bitter", 0.0))
    energy = float(signals.get("energy", 0.5))
    wall = float(signals.get("wall_hazard", 0.0))
    survival = float(affect.get("survival_risk", 0.0))
    altruism = float(affect.get("altruism", 0.0))
    comfort = float(affect.get("comfort", 0.0))
    fear = float(affect.get("fear", 0.0))
    exploration = float(affect.get("exploration", 0.0))
    novelty = float(affect.get("novelty", exploration))
    trust = float(affect.get("trust", 0.0))

    next_signals = dict(signals)
    next_affect = dict(affect)

    # Forward into bitter/wall worsens; toward sugar improves when hungry.
    if bitter > 0.3 and forward > 0.3:
        next_signals["bitter"] = min(1.0, bitter + 0.2 * forward)
        next_affect["pain"] = min(1.0, float(affect.get("pain", 0.0)) + 0.15)
        next_affect["survival_risk"] = min(1.0, survival + 0.2)
        next_signals["energy"] = max(0.0, energy - 0.05)
    elif sugar > 0.25 and (forward > 0.2 or proboscis > 0.5):
        gain = 0.12 * max(forward, proboscis)
        next_signals["energy"] = min(1.0, energy + gain)
        next_signals["sugar"] = min(1.0, sugar + 0.05)
        next_affect["joy"] = min(1.0, float(affect.get("joy", 0.0)) + 0.1)
        next_affect["survival_risk"] = max(0.0, survival - 0.12)
        next_affect["comfort"] = min(1.0, comfort + 0.08)
    elif wall > 0.4 and forward > 0.35:
        next_affect["survival_risk"] = min(1.0, survival + 0.25)
        next_affect["fear"] = min(1.0, fear + 0.15)
        next_signals["energy"] = max(0.0, energy - 0.04)
    elif abs(turn) > 0.4 and (bitter > 0.3 or wall > 0.4):
        next_affect["survival_risk"] = max(0.0, survival - 0.1)
        next_affect["fear"] = max(0.0, fear - 0.08)
        next_signals["bitter"] = max(0.0, bitter - 0.1)
        next_signals["wall_hazard"] = max(0.0, wall - 0.15)
    elif forward < 0.15 and survival > 0.5:
        # Stop conserves a bit when threatened
        next_affect["survival_risk"] = max(0.0, survival - 0.05)
        next_signals["energy"] = max(0.0, energy - 0.01)
    else:
        # Mild exploration cost/benefit
        next_affect["exploration"] = min(1.0, exploration + 0.05 * abs(forward) + 0.03 * abs(turn))
        next_affect["novelty"] = min(1.0, novelty + 0.04)
        next_signals["energy"] = max(0.0, energy - 0.015 * max(0.0, forward))

    # Altruism / growth: explore + social_drive when not in crisis
    if survival < 0.45 and (abs(turn) > 0.2 or forward > 0.3):
        next_affect["altruism"] = min(1.0, altruism + 0.04)
        next_affect["trust"] = min(1.0, trust + 0.03)
        next_affect["comfort"] = min(1.0, float(next_affect.get("comfort", comfort)) + 0.02)

    return next_signals, next_affect


def _utilities_from_state(
    *,
    signals: dict[str, Any],
    affect: dict[str, Any],
    prev_affect: dict[str, Any] | None = None,
    learning_progress: float = 0.0,
) -> tuple[float, float, float]:
    energy = float(signals.get("energy", 0.5))
    bitter = float(signals.get("bitter", 0.0))
    survival = float(affect.get("survival_risk", 0.0))
    damage = float(signals.get("damage", affect.get("pain", 0.0)))
    survival_u = float(
        np.clip(0.55 * energy + 0.35 * (1.0 - survival) + 0.1 * (1.0 - bitter) - 0.2 * damage, 0.0, 1.0)
    )
    altruism_u = float(np.clip(affect.get("altruism", 0.0), 0.0, 1.0))
    spiritual_u = spiritual_growth_score(
        affect=affect,
        prev_affect=prev_affect,
        learning_progress=learning_progress,
    )
    return survival_u, altruism_u, spiritual_u


def _latent_from_embedding(embedding: np.ndarray | None, latent_dim: int) -> np.ndarray:
    if embedding is None:
        return np.zeros(latent_dim, dtype=np.float32)
    vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
    out = np.zeros(latent_dim, dtype=np.float32)
    n = min(latent_dim, vec.shape[0])
    out[:n] = vec[:n]
    return out


def score_candidate(
    *,
    name: str,
    action: np.ndarray,
    signals: dict[str, Any],
    affect: dict[str, Any],
    weights: PriorityWeights,
    world_model: NextStateWorldModel | None = None,
    embedding: np.ndarray | None = None,
    learning_progress: float = 0.0,
) -> Layer1Result:
    cur_s, cur_a, cur_g = _utilities_from_state(signals=signals, affect=affect)
    cur_v = priority_value(
        weights=weights,
        survival_utility=cur_s,
        altruism_utility=cur_a,
        spiritual_utility=cur_g,
    )

    next_signals, next_affect = _heuristic_next_state(signals=signals, affect=affect, action=action)
    # Optional WM: blend latent distance as learning/novelty proxy (does not replace heuristic valence).
    if world_model is not None:
        latent_dim = int(world_model.latent_dim)
        z = _latent_from_embedding(embedding, latent_dim)
        with torch.no_grad():
            z_t = torch.as_tensor(z, dtype=torch.float32).unsqueeze(0)
            a_t = torch.as_tensor(action[: world_model.action_dim], dtype=torch.float32).unsqueeze(0)
            z_next = world_model(z_t, a_t).squeeze(0).numpy()
        delta = float(np.linalg.norm(z_next - z))
        learning_progress = max(learning_progress, float(np.clip(delta / (1.0 + delta), 0.0, 1.0)))

    nxt_s, nxt_a, nxt_g = _utilities_from_state(
        signals=next_signals,
        affect=next_affect,
        prev_affect=affect,
        learning_progress=learning_progress,
    )
    nxt_v = priority_value(
        weights=weights,
        survival_utility=nxt_s,
        altruism_utility=nxt_a,
        spiritual_utility=nxt_g,
    )
    return Layer1Result(
        name=name,
        action=np.asarray(action, dtype=np.float32),
        value=nxt_v,
        survival_u=nxt_s,
        altruism_u=nxt_a,
        spiritual_u=nxt_g,
        improves=nxt_v > cur_v + 1e-4,
    )


def layer1_quick_scan(
    *,
    signals: dict[str, Any],
    affect: dict[str, Any],
    experience: LifeExperience,
    world_model: NextStateWorldModel | None = None,
    embedding: np.ndarray | None = None,
    candidates: list[tuple[str, np.ndarray]] | None = None,
) -> tuple[Layer1Result, list[Layer1Result], PriorityWeights]:
    """Scan latent stimuli / candidate actions; pick best priority-improving action."""
    survival_risk = float(affect.get("survival_risk", 0.0))
    weights = experience.priority_weights(survival_risk=survival_risk)
    peaks = _stimulus_peaks(signals, affect)
    pool = list(candidates or _CANDIDATES)

    # Bias pool toward stimuli (still score all, but ensure relevant actions present).
    if "bitter" in peaks or "wall" in peaks:
        pool = [c for c in pool if c[0] in ("turn_left", "turn_right", "stop", "forward", "explore", "feed")]
    if "sugar" in peaks or "low_energy" in peaks:
        # Prefer feed/forward early in list for tie-breaks
        pool = sorted(pool, key=lambda c: 0 if c[0] in ("feed", "forward") else 1)

    scored = [
        score_candidate(
            name=name,
            action=action,
            signals=signals,
            affect=affect,
            weights=weights,
            world_model=world_model,
            embedding=embedding,
        )
        for name, action in pool
    ]
    scored.sort(key=lambda r: r.value, reverse=True)
    best = scored[0]
    # Prefer an improving action if any exists near the top
    for cand in scored:
        if cand.improves:
            best = cand
            break
    return best, scored, weights


def layer2_reflect_outcomes(
    *,
    layer1: Layer1Result,
    scored: list[Layer1Result],
    past_outcomes: list[dict[str, Any]],
    signals: dict[str, Any],
    affect: dict[str, Any],
) -> tuple[Layer1Result, bool, str]:
    """Prefer actions historically followed by desirable outcomes; veto chronic undesirables."""
    if not past_outcomes:
        return layer1, False, "no_past_outcomes"

    # Aggregate historical score by action name (or nearest action L2).
    by_name: dict[str, list[float]] = {}
    for row in past_outcomes:
        name = str(row.get("action_name", ""))
        kind = str(row.get("kind", "neutral"))
        score = float(row.get("score", 0.0))
        if kind == "desirable":
            score = max(score, 0.1)
        elif kind == "undesirable":
            score = min(score, -0.1)
        by_name.setdefault(name, []).append(score)

    def hist_mean(name: str) -> float:
        vals = by_name.get(name) or []
        return float(np.mean(vals)) if vals else 0.0

    l1_hist = hist_mean(layer1.name)
    reason = f"history[{layer1.name}]={l1_hist:.2f}"

    # Veto if Layer1 action has strongly undesirable history and an alternative is better.
    if l1_hist < -0.15:
        alternatives = sorted(
            scored,
            key=lambda r: (hist_mean(r.name), r.value),
            reverse=True,
        )
        for alt in alternatives:
            if alt.name == layer1.name:
                continue
            if hist_mean(alt.name) > l1_hist + 0.05:
                return (
                    alt,
                    True,
                    f"veto_{layer1.name}_undesirable; prefer_{alt.name} ({reason})",
                )

    # Soft prefer historically desirable when values are close
    for cand in scored:
        if cand.name == layer1.name:
            continue
        if abs(cand.value - layer1.value) < 0.05 and hist_mean(cand.name) > l1_hist + 0.12:
            return (
                cand,
                True,
                f"prefer_history_{cand.name}_over_{layer1.name}",
            )

    sugar = float(signals.get("sugar", 0.0))
    if sugar > 0.4 and layer1.name != "feed":
        for cand in scored:
            if cand.name == "feed" and hist_mean("feed") >= 0.0:
                # only nudge if feed is competitive
                if cand.value >= layer1.value - 0.08:
                    return cand, True, "history_feed_desirable"

    _ = affect  # reserved for future similarity gating
    return layer1, False, reason


def predict_present_future(
    *,
    signals: dict[str, Any],
    affect: dict[str, Any],
    experience: LifeExperience,
    past_outcomes: list[dict[str, Any]] | None = None,
    world_model: NextStateWorldModel | None = None,
    embedding: np.ndarray | None = None,
) -> FuturePrediction:
    best, scored, weights = layer1_quick_scan(
        signals=signals,
        affect=affect,
        experience=experience,
        world_model=world_model,
        embedding=embedding,
    )
    chosen, overridden, reason = layer2_reflect_outcomes(
        layer1=best,
        scored=scored,
        past_outcomes=past_outcomes or [],
        signals=signals,
        affect=affect,
    )
    peaks = _stimulus_peaks(signals, affect)
    present = (
        f"Present: energy={float(signals.get('energy', 0.0)):.2f}, "
        f"sugar={float(signals.get('sugar', 0.0)):.2f}, "
        f"bitter={float(signals.get('bitter', 0.0)):.2f}, "
        f"survival_risk={float(affect.get('survival_risk', 0.0)):.2f}, "
        f"stimuli={peaks or ['calm']}, "
        f"maturity={experience.maturity():.2f}."
    )
    hypo = (
        f"Future hypothesis (L1+L2): act={chosen.name} "
        f"value={chosen.value:.3f} improves={chosen.improves}; "
        f"weights={weights.as_dict()}; "
        f"{'override: ' + reason if overridden else 'confirm: ' + reason}."
    )
    conf = float(np.clip(0.35 + 0.4 * float(chosen.improves) + 0.2 * experience.maturity(), 0.1, 0.95))
    if overridden:
        conf = float(np.clip(conf + 0.05, 0.1, 0.95))

    # Optional immediate outcome hint for logging
    next_s, next_a = _heuristic_next_state(signals=signals, affect=affect, action=chosen.action)
    hint = label_transition(
        prev_signals=signals,
        prev_affect=affect,
        next_signals=next_s,
        next_affect=next_a,
    )

    return FuturePrediction(
        present=present,
        future_hypothesis=hypo,
        chosen_action=np.asarray(chosen.action, dtype=np.float32),
        chosen_name=chosen.name,
        layer1_values=[
            {
                "name": s.name,
                "value": s.value,
                "improves": s.improves,
                "survival_u": s.survival_u,
                "altruism_u": s.altruism_u,
                "spiritual_u": s.spiritual_u,
            }
            for s in scored
        ],
        layer2_override=overridden,
        layer2_reason=reason,
        confidence=conf,
        priority_weights=weights.as_dict(),
        outcome_hint=hint.to_dict(),
    )


def action_name_from_vector(action: np.ndarray) -> str:
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.shape[0] < HIGH_LEVEL_ACTION_DIM:
        pad = np.zeros(HIGH_LEVEL_ACTION_DIM, dtype=np.float32)
        pad[: a.shape[0]] = a
        a = pad
    best_name = "explore"
    best_d = float("inf")
    for name, cand in _CANDIDATES:
        d = float(np.linalg.norm(a[:5] - cand[:5]))
        if d < best_d:
            best_d = d
            best_name = name
    return best_name
