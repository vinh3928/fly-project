from __future__ import annotations

import numpy as np

from flybrain.affect import AffectEstimator
from flybrain.phase4 import AFFECT_DIM, AFFECT_KEYS, HIGH_LEVEL_ACTION_DIM, TOTAL_EMBEDDING_DIM
from flybrain.signals import SignalBuilder


def test_phase4_constants() -> None:
    assert TOTAL_EMBEDDING_DIM == 272
    assert len(AFFECT_KEYS) == AFFECT_DIM
    assert HIGH_LEVEL_ACTION_DIM == 5


def test_signal_builder_splits_observation() -> None:
    builder = SignalBuilder()
    obs = np.linspace(-0.5, 0.5, 197, dtype=np.float32)
    signals = builder.build(obs)
    assert signals.body.shape[0] == 85
    assert signals.chemical.shape[0] == 64
    assert signals.internal.shape[0] == 8
    assert signals.contacts.shape[0] == 30
    assert signals.left_eye.shape == (1, 64, 64)
    assert "energy" in signals.summary
    assert "sugar" in signals.to_jsonable()


def test_affect_estimator_sugar_and_bitter() -> None:
    builder = SignalBuilder()
    obs = np.zeros(197, dtype=np.float32)
    signals = builder.build(obs)
    signals.summary.update({"sugar": 0.9, "bitter": 0.0, "energy": 0.9, "hunger": 0.0, "damage": 0.0, "instability": 0.0})
    joy_state = AffectEstimator().estimate(signals)
    assert joy_state.scores["joy"] > 0.5
    assert joy_state.scores["anger"] < 0.2

    signals.summary.update({"sugar": 0.0, "bitter": 0.9, "energy": 0.2, "damage": 0.5})
    fear_state = AffectEstimator().estimate(signals)
    assert fear_state.scores["anger"] > 0.5
    assert fear_state.scores["survival_risk"] > 0.3
    assert fear_state.as_vector().shape == (AFFECT_DIM,)
