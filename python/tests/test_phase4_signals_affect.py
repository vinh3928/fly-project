from __future__ import annotations

import numpy as np

from flybrain.affect import AffectEstimator
from flybrain.phase4 import (
    AFFECT_DIM,
    AFFECT_KEYS,
    AFFECT_VEC_DIM,
    EYE_VEC_DIM,
    HIGH_LEVEL_ACTION_DIM,
    SENSORS_VEC_DIM,
    STATUS_VEC_DIM,
    STM_TABLE,
    TOTAL_EMBEDDING_DIM,
)
from flybrain.signals import SignalBuilder


def test_phase4_constants() -> None:
    assert TOTAL_EMBEDDING_DIM == 272
    assert len(AFFECT_KEYS) == AFFECT_DIM == AFFECT_VEC_DIM
    assert HIGH_LEVEL_ACTION_DIM == 5
    assert EYE_VEC_DIM == 128
    assert SENSORS_VEC_DIM == 192
    assert STATUS_VEC_DIM == 32
    assert STM_TABLE == "fly_mind.short_term_memory"


def test_signal_builder_reads_unity_energy_index() -> None:
    builder = SignalBuilder()
    obs = np.zeros(197, dtype=np.float32)
    obs[176] = 0.87  # INTERNAL_OFFSET
    obs[182] = 0.13  # hunger
    signals = builder.build(obs)
    assert abs(signals.summary["energy"] - 0.87) < 1e-5
    assert abs(signals.summary["hunger"] - 0.13) < 1e-5
    assert signals.summary.get("internal_all_zero", 0.0) == 0.0


def test_fake_env_energy_at_unity_layout_index() -> None:
    from flybrain.env import FakeFlyEnv
    from flybrain.signals.builder import INTERNAL_OFFSET, SignalBuilder

    env = FakeFlyEnv()
    obs, _ = env.reset()
    assert abs(float(obs[INTERNAL_OFFSET]) - 1.0) < 1e-5
    signals = SignalBuilder().build(obs)
    assert signals.summary["energy"] > 0.9


def test_affect_estimator_sugar_and_bitter() -> None:
    builder = SignalBuilder()
    obs = np.zeros(197, dtype=np.float32)
    signals = builder.build(obs)
    signals.summary.update(
        {
            "sugar": 0.9,
            "bitter": 0.0,
            "energy": 0.9,
            "hunger": 0.0,
            "damage": 0.0,
            "instability": 0.0,
        }
    )
    joy_state = AffectEstimator().estimate(signals)
    assert joy_state.scores["joy"] > 0.5
    assert joy_state.scores["anger"] < 0.2

    signals.summary.update({"sugar": 0.0, "bitter": 0.9, "energy": 0.2, "damage": 0.5})
    fear_state = AffectEstimator().estimate(signals)
    assert fear_state.scores["anger"] > 0.5
    assert fear_state.scores["survival_risk"] > 0.3
    assert fear_state.as_vector().shape == (AFFECT_DIM,)
