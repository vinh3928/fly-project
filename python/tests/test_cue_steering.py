"""Layer-1 roam/feed candidates steer toward sensed cues (both directions)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from flybrain.signals.builder import antenna_bearing
from flybrain.thinking.composer import is_hungry
from flybrain.thinking.two_layer_future import (
    CUE_TURN_MAX,
    _CANDIDATES,
    cue_turn,
    layer1_quick_scan,
    steer_candidate,
)
from flybrain.world_model.outcomes import LifeExperience

_HUNGRY = {"hunger": 0.8, "energy": 0.2}
_FED = {"hunger": 0.0, "energy": 1.0}


def _candidate(name: str) -> np.ndarray:
    return dict(_CANDIDATES)[name]


def test_is_hungry_thresholds() -> None:
    assert is_hungry(_HUNGRY)
    assert not is_hungry(_FED)
    assert is_hungry({"energy": 0.25})
    assert is_hungry({"energy": 0.5})  # below ENERGY_SEEK_MAX 0.55
    assert not is_hungry({"energy": 0.7})


@pytest.mark.parametrize("bearing", [-0.6, 0.6])
def test_hungry_fly_turns_toward_sugar_either_way(bearing: float) -> None:
    signals = {**_HUNGRY, "sugar_smell": 0.2, "sugar_bearing": bearing}
    turn, source = cue_turn(signals)
    assert source == "sugar"
    assert turn is not None and math.copysign(1.0, turn) == math.copysign(1.0, bearing)
    explore = steer_candidate("explore", _candidate("explore"), signals)
    feed = steer_candidate("feed", _candidate("feed"), signals)
    assert explore[1] == pytest.approx(turn)
    assert feed[1] == pytest.approx(turn)
    # Forward / proboscis untouched.
    assert explore[0] == pytest.approx(_candidate("explore")[0])
    assert feed[4] == pytest.approx(_candidate("feed")[4])


def test_fed_fly_ignores_sugar_and_follows_novelty() -> None:
    signals = {
        **_FED,
        "sugar_smell": 0.3,
        "sugar_bearing": 0.8,
        "visual_novelty": 0.5,
        "novelty_bearing": -0.6,
    }
    turn, source = cue_turn(signals)
    assert source == "novelty"
    assert turn is not None and turn < 0.0
    # Feed only steers toward sugar, not novelty.
    feed = steer_candidate("feed", _candidate("feed"), signals)
    assert feed[1] == pytest.approx(_candidate("feed")[1])


def test_wall_bias_used_when_no_other_cue() -> None:
    turn, source = cue_turn({**_FED, "turn_bias": -0.4})
    assert source == "wall"
    assert turn == pytest.approx(-0.4)


def test_no_cue_keeps_default_candidate() -> None:
    assert cue_turn(_FED) == (None, "")
    out = steer_candidate("explore", _candidate("explore"), _FED)
    np.testing.assert_allclose(out, _candidate("explore"))


def test_turn_is_clamped_and_fixed_candidates_unchanged() -> None:
    signals = {**_HUNGRY, "sugar_smell": 0.5, "sugar_bearing": 5.0}
    turn, _ = cue_turn(signals)
    assert turn == pytest.approx(CUE_TURN_MAX)
    for name in ("forward", "turn_left", "turn_right", "stop"):
        np.testing.assert_allclose(steer_candidate(name, _candidate(name), signals), _candidate(name))


def test_steering_does_not_mutate_candidate_table() -> None:
    before = _candidate("explore").copy()
    steer_candidate("explore", _candidate("explore"), {**_HUNGRY, "sugar_smell": 0.2, "sugar_bearing": -1.0})
    np.testing.assert_allclose(_candidate("explore"), before)


def test_layer1_prefers_feed_over_explore_when_low_energy() -> None:
    """Survival/low_energy must rank feed above explore (not only route-lock plan)."""
    best, scored, _ = layer1_quick_scan(
        signals={"hunger": 0.8, "energy": 0.2, "sugar": 0.0, "curiosity_drive": 0.0},
        affect={"survival_risk": 0.4, "novelty": 0.0, "exploration": 0.1},
        experience=LifeExperience(),
    )
    by_name = {r.name: r.value for r in scored}
    assert best.name == "feed"
    assert by_name["feed"] > by_name["explore"]


def test_layer1_hungry_without_sugar_contact_still_chooses_feed() -> None:
    best, scored, _ = layer1_quick_scan(
        signals={"hunger": 0.5, "energy": 0.5, "sugar": 0.0, "sugar_smell": 0.0},
        affect={"survival_risk": 0.35},
        experience=LifeExperience(),
    )
    assert best.name == "feed"
    assert scored[0].name == "feed" or {r.name: r for r in scored}["feed"].value >= scored[0].value - 1e-6


@pytest.mark.parametrize("bearing", [-0.7, 0.7])
def test_layer1_scan_roam_actions_follow_sugar_bearing(bearing: float) -> None:
    signals = {**_HUNGRY, "sugar_smell": 0.2, "sugar_bearing": bearing}
    _best, scored, _w = layer1_quick_scan(
        signals=signals,
        affect={"survival_risk": 0.2},
        experience=LifeExperience(),
    )
    by_name = {r.name: r for r in scored}
    for name in ("explore", "court", "feed"):
        assert math.copysign(1.0, float(by_name[name].action[1])) == math.copysign(1.0, bearing)


def test_long_antennae_give_proportional_bearing_on_one_meter_plume() -> None:
    """Mirror Unity: antenna tips 38 mm apart (AntennaOdorSensor.HalfSpacing), 0.01 rim at 1 m."""

    def plume(distance: float) -> float:
        return 0.01 ** (distance / 1.0)

    half = 0.019
    # Sugar straight to the right: saturated bearing → pivot.
    assert antenna_bearing(plume(0.2 + half), plume(0.2 - half)) == pytest.approx(1.0)
    # Sugar ~10° off the nose: proportional, not saturated (≈ 1.75·sin 10°).
    lateral = half * math.sin(math.radians(10.0))
    bearing = antenna_bearing(plume(0.2 + lateral), plume(0.2 - lateral))
    assert 0.2 < bearing < 0.45
    # Old head-scaled spacing (~27 µm) was far too small to steer.
    assert abs(antenna_bearing(plume(0.2 + 1.35e-5), plume(0.2 - 1.35e-5))) < 0.05
