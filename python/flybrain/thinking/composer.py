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

    draft = (
        f"Draft: joy={joy:.2f} anger={anger:.2f} survival={survival:.2f} "
        f"altruism={altruism:.2f}; sugar={sugar:.2f} bitter={bitter:.2f}. "
        f"Plan to {'approach sugar' if sugar > bitter else 'avoid bitter / explore'}."
    )

    # High-level action: [forward, turn, height, frequency, proboscis]
    forward = exploration * 0.6 + joy * 0.4 - bitter * 0.5
    turn = anger * 0.4 - sugar * 0.2
    if bitter > sugar:
        turn = np.sign(turn + 1e-6) * max(abs(turn), 0.4)
    if survival > 0.6:
        forward *= float(clamps["survival_forward_scale"])
    height = float(clamps["height_default"])
    frequency = float(clamps["frequency_default"]) + 0.2 * exploration
    proboscis = float(np.clip(sugar * 0.8 - bitter * 0.5, 0.0, 1.0))

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
