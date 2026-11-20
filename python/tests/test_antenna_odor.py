"""Antenna odor: stronger near a sugar/bitter source (mirrors Unity falloff)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from flybrain.phase4 import DEFAULT_OBS_DIM, SMELL_VEC_DIM
from flybrain.pipeline.live_mind import antenna_debug_payload, format_live_step_log
from flybrain.signals.builder import (
    BITTER_CHANNEL,
    ODOR_CHANNEL,
    SUGAR_CHANNEL,
    SignalBuilder,
    antenna_bearing,
    antenna_smell_vec,
)


def _unity_plume(strength: float, distance: float, spread: float) -> float:
    """Match ChemicalSource: hard cutoff outside spread, exp falloff inside."""
    radius = max(1e-4, spread)
    if distance > radius:
        return 0.0
    return float(strength * math.exp(-distance / radius))


def _obs_with_antenna_odor(*, left_odor: float, right_odor: float = 0.0) -> np.ndarray:
    obs = np.zeros(DEFAULT_OBS_DIM, dtype=np.float32)
    start = 72 + 30 + 48  # joints + contacts + foot taste
    obs[start + ODOR_CHANNEL] = left_odor
    obs[start + 8 + ODOR_CHANNEL] = right_odor
    return obs


def test_plume_falloff_stronger_closer_to_source() -> None:
    strength, spread = 1.0, 0.100
    near = _unity_plume(strength, 0.02, spread)
    mid = _unity_plume(strength, spread, spread)
    far = _unity_plume(strength, spread + 0.001, spread)
    assert near > mid > far
    assert abs(near - math.exp(-0.02 / 0.100)) < 1e-6
    assert abs(mid - math.exp(-1.0)) < 1e-6
    assert far == 0.0


def test_signal_builder_odor_stronger_when_antenna_nearer() -> None:
    """Python obs path: higher antenna Odor channel → higher summary odor."""
    near = _unity_plume(1.0, 0.02, 0.100)
    far = _unity_plume(1.0, 0.101, 0.100)
    builder = SignalBuilder()
    near_s = builder.build(_obs_with_antenna_odor(left_odor=near, right_odor=near * 0.95))
    far_s = builder.build(_obs_with_antenna_odor(left_odor=far, right_odor=far * 0.95))
    assert near_s.summary["odor"] > far_s.summary["odor"]
    assert near_s.summary["odor_left"] > far_s.summary["odor_left"]
    assert float(near_s.smell_vec[ODOR_CHANNEL]) == near_s.summary["odor_left"]
    smell = antenna_smell_vec(_obs_with_antenna_odor(left_odor=near))
    assert smell.shape == (SMELL_VEC_DIM,)
    assert abs(float(smell[ODOR_CHANNEL]) - near) < 1e-6


def _obs_with_antenna_channel(channel: int, *, left: float, right: float) -> np.ndarray:
    obs = np.zeros(DEFAULT_OBS_DIM, dtype=np.float32)
    start = 72 + 30 + 48
    obs[start + channel] = left
    obs[start + 8 + channel] = right
    return obs


def test_antenna_sugar_smell_reaches_summary_without_taste() -> None:
    s = SignalBuilder().build(_obs_with_antenna_channel(SUGAR_CHANNEL, left=0.10, right=0.12))
    assert s.summary["sugar_smell"] == pytest.approx(0.12)
    assert s.summary["sugar_smell_left"] == pytest.approx(0.10)
    assert s.summary["sugar_smell_right"] == pytest.approx(0.12)
    # Taste (feet/proboscis contact) stays separate from smell at range.
    assert s.summary["sugar"] == 0.0
    assert s.to_jsonable()["sugar_smell"] == pytest.approx(0.12)


def test_sugar_bearing_points_to_stronger_antenna() -> None:
    builder = SignalBuilder()
    # Long (38 mm apart) antennae see a few % left/right difference off-axis.
    right = builder.build(_obs_with_antenna_channel(SUGAR_CHANNEL, left=0.100, right=0.105))
    left = builder.build(_obs_with_antenna_channel(SUGAR_CHANNEL, left=0.105, right=0.100))
    assert right.summary["sugar_bearing"] > 0.3
    assert left.summary["sugar_bearing"] < -0.3
    bitter = builder.build(_obs_with_antenna_channel(BITTER_CHANNEL, left=0.2, right=0.1))
    assert bitter.summary["bitter_bearing"] < 0.0
    assert bitter.summary["bitter_smell"] == pytest.approx(0.2)


def test_antenna_bearing_zero_without_smell_and_clamped() -> None:
    assert antenna_bearing(0.0, 0.0) == 0.0
    assert antenna_bearing(0.0, 1.0) == 1.0
    assert antenna_bearing(1.0, 0.0) == -1.0
    assert antenna_bearing(0.5, 0.5) == 0.0


def test_proboscis_contact_and_extension_follow_unity_layout() -> None:
    from flybrain.signals.builder import (
        INTERNAL_OFFSET,
        PROBOSCIS_CONTACT_INDEX,
        PROBOSCIS_DIM,
        PROBOSCIS_EXTENSION_INDEX,
    )

    start = INTERNAL_OFFSET - PROBOSCIS_DIM
    obs = np.zeros(DEFAULT_OBS_DIM, dtype=np.float32)
    obs[start + PROBOSCIS_EXTENSION_INDEX] = 0.8  # Unity buffer[offset] = extension
    obs[start + PROBOSCIS_CONTACT_INDEX] = 0.0  # buffer[offset + 1] = contact
    summary = SignalBuilder().build(obs).summary
    assert summary["proboscis_contact"] == pytest.approx(0.0)
    assert summary["proboscis_extension"] == pytest.approx(0.8)

    obs[start + PROBOSCIS_CONTACT_INDEX] = 1.0
    obs[start + 2 + SUGAR_CHANNEL] = 0.9
    summary = SignalBuilder().build(obs).summary
    assert summary["proboscis_contact"] == pytest.approx(1.0)
    assert summary["sugar"] == pytest.approx(0.9)


def test_fake_food_env_writes_unity_proboscis_layout() -> None:
    from flybrain.env_food_search import FakeFoodSearchEnv

    env = FakeFoodSearchEnv()
    env.reset()
    obs = env._write_obs(proboscis_ext=0.9, sugar_near=True)
    summary = SignalBuilder().build(obs).summary
    assert summary["proboscis_extension"] == pytest.approx(0.9)
    assert summary["proboscis_contact"] == pytest.approx(1.0)


def test_live_log_shows_sugar_sniff() -> None:
    line = format_live_step_log(
        5,
        [0.1, 0.0, 0.5, 0.5, 0.0],
        {"sugar_smell": 0.0832, "sugar_bearing": 0.41, "energy": 0.3},
    )
    assert "sniff(sugar=0.083,bear=+0.41)" in line


def test_antenna_debug_payload_preserves_sugar_bitter_and_odor() -> None:
    obs = np.zeros(DEFAULT_OBS_DIM, dtype=np.float32)
    start = 72 + 30 + 48
    obs[start + 0] = 0.4
    obs[start + 1] = 0.1
    obs[start + 7] = 0.4
    obs[start + 8 + 0] = 0.3
    obs[start + 8 + 1] = 0.2
    obs[start + 8 + 7] = 0.3
    debug = antenna_debug_payload(SignalBuilder().build(obs))
    assert debug == {
        "left_sugar": pytest.approx(0.4),
        "left_bitter": pytest.approx(0.1),
        "left_odor": pytest.approx(0.4),
        "right_sugar": pytest.approx(0.3),
        "right_bitter": pytest.approx(0.2),
        "right_odor": pytest.approx(0.3),
    }
