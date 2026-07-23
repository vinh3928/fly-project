"""Present/future reasoning — Phase 7 two-layer path with stub fallback."""

from __future__ import annotations

from typing import Any

from flybrain.thinking.two_layer_future import FuturePrediction, predict_present_future
from flybrain.world_model.outcomes import LifeExperience


def present_future_stub(
    *,
    signals: dict[str, Any],
    affect: dict[str, Any],
) -> dict[str, str]:
    """Backward-compatible text stub; uses two-layer predictor with zero experience."""
    pred = predict_present_future(
        signals=signals,
        affect=affect,
        experience=LifeExperience(lifetime_steps=0, labeled_outcomes=0),
        past_outcomes=[],
    )
    return pred.as_present_future()


def present_future_from_prediction(pred: FuturePrediction) -> dict[str, str]:
    return pred.as_present_future()
