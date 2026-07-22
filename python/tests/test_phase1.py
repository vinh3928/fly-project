from __future__ import annotations

from flybrain.phase1 import (
    AUTOMATED_FIXED_DELTA_TIME,
    AUTOMATED_STABILITY_FIXED_STEPS,
    AUTOMATED_STABILITY_SECONDS,
    STABILITY_SECONDS,
)


def test_phase1_exit_criteria_mirror_unity() -> None:
    assert STABILITY_SECONDS == 600.0
    assert AUTOMATED_STABILITY_FIXED_STEPS == 2000
    assert AUTOMATED_FIXED_DELTA_TIME == 0.005
    assert AUTOMATED_STABILITY_SECONDS == 10.0
