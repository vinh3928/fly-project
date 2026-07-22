from __future__ import annotations

from flybrain.phase2 import (
    AUTOMATED_FIXED_DELTA_TIME,
    AUTOMATED_MANEUVER_FIXED_STEPS,
    AUTOMATED_STAND_FIXED_STEPS,
    AUTOMATED_STAND_SECONDS,
    STAND_SECONDS,
    TURN_MIN_YAW_DEGREES,
    WALK_MIN_DISPLACEMENT,
)


def test_phase2_exit_criteria_mirror_unity() -> None:
    assert STAND_SECONDS == 30.0
    assert AUTOMATED_STAND_FIXED_STEPS == 2000
    assert AUTOMATED_MANEUVER_FIXED_STEPS == 800
    assert AUTOMATED_FIXED_DELTA_TIME == 0.005
    assert AUTOMATED_STAND_SECONDS == 10.0
    assert WALK_MIN_DISPLACEMENT == 0.01
    assert TURN_MIN_YAW_DEGREES == 20.0
