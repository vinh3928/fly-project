"""Compose refined thoughts and high-level motor intents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from flybrain.phase4 import HIGH_LEVEL_ACTION_DIM


@dataclass
class ThoughtResult:
    draft: str
    reflection: str
    present_future: dict[str, str]
    devil_critique: str
    refined_thought: str
    action: np.ndarray
    intent: dict[str, Any]


def compose_thought(
    *,
    signals: dict[str, Any],
    affect: dict[str, Any],
    reflection: str,
    present_future: dict[str, str],
    devil_critique: str,
    action_clamps: dict[str, float] | None = None,
) -> ThoughtResult:
    clamps = {
        "forward_max": 1.0,
        "turn_max": 1.0,
        "height_default": 0.5,
        "frequency_default": 0.5,
        "survival_forward_scale": 0.25,
        **(action_clamps or {}),
    }

    sugar = float(signals.get("sugar", 0.0))
    bitter = float(signals.get("bitter", 0.0))
    survival = float(affect.get("survival_risk", 0.0))
    exploration = float(affect.get("exploration", 0.0))
    joy = float(affect.get("joy", 0.0))
    anger = float(affect.get("anger", 0.0))
    altruism = float(affect.get("altruism", 0.0))
    hunger = float(signals.get("hunger", affect.get("sad", 0.0)))
    energy = float(signals.get("energy", 1.0 - hunger))
    proboscis_contact = float(signals.get("proboscis_contact", 0.0))
    sugar_bearing = float(signals.get("sugar_bearing", signals.get("target_bearing", 0.0)))
    bitter_bearing = float(signals.get("bitter_bearing", 0.0))
    target_bearing = float(signals.get("target_bearing", 0.0))
    target_confidence = float(signals.get("target_confidence", 0.0))
    wall_hazard = float(signals.get("wall_hazard", 0.0))
    forward_bias = float(signals.get("forward_bias", 0.0))
    turn_bias = float(signals.get("turn_bias", 0.0))

    hungry = hunger >= 0.45 or energy < 0.55
    draft = (
        f"Draft: joy={joy:.2f} anger={anger:.2f} survival={survival:.2f} "
        f"hunger={hunger:.2f} energy={energy:.2f}; sugar={sugar:.2f} bitter={bitter:.2f}. "
        f"Plan to {'seek/consume sugar' if hungry and sugar >= bitter else ('avoid bitter' if bitter > sugar else 'explore')}."
    )

    # High-level action: [forward, turn, height, frequency, proboscis]
    forward = exploration * 0.6 + joy * 0.4 - bitter * 0.5
    turn = anger * 0.4 - sugar * 0.2
    # Phase 5 visual nav cues
    forward = 0.55 * forward + 0.45 * forward_bias + 0.25 * target_confidence
    turn = 0.45 * turn + 0.55 * turn_bias + 0.5 * target_bearing * target_confidence
    # Phase 6 food-search
    if hungry:
        forward = 0.5 * forward + 0.5 * (0.55 + 0.4 * sugar - 0.5 * bitter)
        turn = 0.4 * turn + 0.6 * (0.8 * sugar_bearing - 0.55 * bitter_bearing)
        if sugar > 0.25:
            turn = float(np.clip(turn + 0.35 * sugar_bearing, -1.0, 1.0))
    else:
        forward *= 0.35
        turn *= 0.4
    if bitter > sugar:
        turn = np.sign(turn - bitter_bearing + 1e-6) * max(abs(turn), 0.45)
        turn = float(np.clip(turn - 0.4 * np.sign(bitter_bearing + 1e-6), -1.0, 1.0))
    if survival > 0.6:
        forward *= float(clamps["survival_forward_scale"])
    if wall_hazard > 0.4:
        scale = float(clamps.get("wall_hazard_forward_scale", 0.2))
        forward *= max(scale, 1.0 - wall_hazard)
        turn = float(np.clip(turn + 0.35 * np.sign(turn + 1e-6), -1.0, 1.0))
    height = float(clamps["height_default"])
    frequency = float(clamps["frequency_default"]) + 0.2 * exploration
    if hungry and sugar > bitter:
        proboscis = float(np.clip(0.35 + 0.7 * sugar + 0.4 * proboscis_contact - 0.8 * bitter, 0.0, 1.0))
    else:
        proboscis = float(np.clip(sugar * 0.3 - bitter * 0.8, 0.0, 1.0))

    forward = float(np.clip(forward, -clamps["forward_max"], clamps["forward_max"]))
    turn = float(np.clip(turn, -clamps["turn_max"], clamps["turn_max"]))
    frequency = float(np.clip(frequency, 0.0, 1.0))

    # Devil's critique softens aggressive forward if critique mentions fall/toxin
    critique_l = devil_critique.lower()
    if "fall" in critique_l or "upright" in critique_l:
        forward *= 0.5
    if "toxin" in critique_l or "bitter" in critique_l:
        forward *= 0.7
        turn = float(np.clip(turn + 0.2, -1.0, 1.0))

    action = np.asarray([forward, turn, height, frequency, proboscis], dtype=np.float32)
    assert action.shape[0] == HIGH_LEVEL_ACTION_DIM

    refined = (
        f"{present_future.get('present', '')} {present_future.get('future_hypothesis', '')} "
        f"Refined after critique: forward={forward:.2f} turn={turn:.2f}. "
        f"Critique noted: {devil_critique[:180]}"
    )

    intent = {
        "forward": forward,
        "turn": turn,
        "height": height,
        "frequency": frequency,
        "proboscis": proboscis,
        "confidence": float(np.clip(1.0 - 0.5 * survival, 0.1, 1.0)),
        "action": action.tolist(),
    }

    return ThoughtResult(
        draft=draft,
        reflection=reflection,
        present_future=present_future,
        devil_critique=devil_critique,
        refined_thought=refined,
        action=action,
        intent=intent,
    )
