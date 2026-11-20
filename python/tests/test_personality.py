"""Live personality traits."""

from __future__ import annotations

from flybrain.thinking.personality import TRAIT_NAMES, PersonalityState, PersonalityTraits


def test_seed_and_clamp() -> None:
    traits = PersonalityTraits.from_dict({"risk_appetite": 1.5, "wisdom": -0.2})
    assert traits.risk_appetite == 1.0
    assert traits.wisdom == 0.0


def test_adjust_and_set_with_source_history() -> None:
    state = PersonalityState.from_config(
        {"risk_appetite": 0.4, "curiosity_to_test": 0.7, "wisdom": 0.3}
    )
    state.adjust({"wisdom": 0.1, "bogus": 9.0}, source="retrospective")
    assert abs(state.snapshot().wisdom - 0.4) < 1e-6
    state.set_traits({"risk_appetite": 0.1}, source="outcome")
    assert abs(state.snapshot().risk_appetite - 0.1) < 1e-6
    hist = state.history(2)
    assert len(hist) == 2
    assert hist[-1]["source"] == "outcome"
    assert hist[0]["source"] == "retrospective"
    assert "bogus" not in TRAIT_NAMES
    assert state.as_dict()["last_source"] == "outcome"


def test_adjust_clamps_at_bounds() -> None:
    state = PersonalityState(traits=PersonalityTraits(playfulness=0.95))
    state.adjust({"playfulness": 0.5}, source="test")
    assert state.snapshot().playfulness == 1.0
