"""Developmental tendency drive (curiosity early, sex later)."""

from __future__ import annotations

from flybrain.thinking.tendency import TendencyConfig, TendencyDrive
from flybrain.thinking.two_layer_future import layer1_quick_scan
from flybrain.world_model.outcomes import LifeExperience


def test_juvenile_tendency_is_curiosity() -> None:
    drive = TendencyDrive(TendencyConfig())
    state = drive.state(maturity=0.0)
    assert state.dominant == "curiosity"
    assert state.curiosity > state.sex
    assert state.curiosity > state.forage


def test_adult_tendency_can_be_sex() -> None:
    drive = TendencyDrive(TendencyConfig())
    state = drive.state(maturity=1.0)
    assert state.dominant == "sex"
    assert state.sex > state.curiosity


def test_outcomes_shift_tendency() -> None:
    drive = TendencyDrive(TendencyConfig(outcome_lr=0.2))
    before = drive.state(maturity=0.5).curiosity
    for _ in range(8):
        drive.update_from_outcome(action_name="explore", outcome_score=0.5, kind="desirable")
    after = drive.state(maturity=0.5).curiosity
    assert after > before


def test_safe_mode_prefers_explore_not_feed_for_juvenile() -> None:
    drive = TendencyDrive(TendencyConfig(safe_mode_bonus=0.35))
    # Immature fly: maturity ~0 via no labels / age.
    exp = LifeExperience(lifetime_steps=0, labeled_outcomes=0, age_horizon=500, data_horizon=100)
    signals = {
        "energy": 1.0,
        "sugar": 0.0,
        "bitter": 0.0,
        "wall_hazard": 0.8,
        "boredom": 0.0,
    }
    affect = {
        "survival_risk": 0.2,
        "altruism": 0.0,
        "fear": 0.4,
        "novelty": 0.3,
        "exploration": 0.3,
        "comfort": 0.5,
        "joy": 0.2,
        "pain": 0.0,
    }
    assert TendencyDrive.is_safe_mode(signals, ["wall", "novelty"])
    best, scored, _ = layer1_quick_scan(
        signals=signals,
        affect=affect,
        experience=exp,
        tendency=drive,
    )
    names = [s.name for s in scored[:3]]
    assert best.name != "feed"
    assert best.name in ("explore", "forward", "turn_left", "turn_right", "court")
    assert "explore" in names or best.name == "explore"


def test_safe_mode_false_when_hungry_for_sugar() -> None:
    signals = {"energy": 0.3, "sugar": 0.5, "bitter": 0.0}
    assert TendencyDrive.is_safe_mode(signals, ["sugar", "low_energy"]) is False


def test_safe_mode_false_on_hunger_peak_or_rising_hunger() -> None:
    from flybrain.thinking.two_layer_future import _stimulus_peaks

    calm = {"energy": 0.8, "sugar": 0.0, "bitter": 0.0, "hunger": 0.1}
    assert TendencyDrive.is_safe_mode(calm, ["novelty"]) is True
    assert TendencyDrive.is_safe_mode(calm, ["hunger"]) is False
    rising = {"energy": 0.8, "sugar": 0.0, "bitter": 0.0, "hunger": 0.3}
    assert TendencyDrive.is_safe_mode(rising, ["novelty"]) is False
    peaks = _stimulus_peaks(rising, {"novelty": 0.0, "survival_risk": 0.0})
    assert "hunger" in peaks
    assert "low_energy" not in peaks
