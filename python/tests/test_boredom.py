"""Stimulus-agnostic boredom: stuck thoughts / stuck body."""

from __future__ import annotations

import numpy as np

from flybrain.curiosity.boredom import (
    BoredomConfig,
    BoredomTracker,
    apply_boredom_escape_goal,
    apply_boredom_to_candidate_value,
    thought_fingerprint,
)
from flybrain.thinking.two_layer_future import score_candidate
from flybrain.world_model.outcomes import PriorityWeights


def test_thought_fingerprint_ignores_whitespace() -> None:
    a = thought_fingerprint("stop", "Present: energy=1.00.  Act=stop.")
    b = thought_fingerprint("stop", "Present: energy=1.00. Act=stop.")
    assert a == b
    assert thought_fingerprint("explore", "Present: energy=1.00. Act=stop.") != a


def test_physical_stuck_raises_boredom_without_objects() -> None:
    tracker = BoredomTracker(
        BoredomConfig(physical_horizon_steps=5, force_threshold=0.5, thought_horizon_ticks=99)
    )
    stop = np.zeros(5, dtype=np.float32)
    pos = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    state = None
    for _ in range(6):
        state = tracker.observe_physical(position=pos, action=stop)
    assert state is not None
    assert state.physical_stuck >= 0.99
    assert state.boredom >= 0.5
    assert state.force_locomotion


def test_same_thought_raises_boredom() -> None:
    tracker = BoredomTracker(
        BoredomConfig(thought_horizon_ticks=3, force_threshold=0.5, physical_horizon_steps=99)
    )
    state = tracker.observe_thought(plan_name="stop", refined_thought="freeze here")
    assert state.thought_streak == 0
    for _ in range(3):
        state = tracker.observe_thought(plan_name="stop", refined_thought="freeze here")
    assert state.thought_streak == 3
    assert state.thought_stuck >= 0.99
    assert state.force_locomotion
    # New thought clears streak.
    state = tracker.observe_thought(plan_name="explore", refined_thought="walk now")
    assert state.thought_streak == 0
    assert state.force_locomotion is False


def test_escape_goal_replaces_stop_when_forced() -> None:
    cfg = BoredomConfig(force_threshold=0.5, escape_forward=0.4, escape_turn=0.6)
    tracker = BoredomTracker(cfg)
    stop = np.zeros(5, dtype=np.float32)
    for _ in range(12):
        tracker.observe_physical(position=None, action=stop)
    state = tracker.state()
    assert state.force_locomotion
    escaped = apply_boredom_escape_goal(stop, state, config=cfg, escape_sign=1.0)
    assert float(escaped[0]) >= 0.39
    assert abs(float(escaped[1])) >= 0.59


def test_l1_stop_penalized_when_bored() -> None:
    signals = {
        "energy": 1.0,
        "sugar": 0.0,
        "bitter": 0.0,
        "wall_hazard": 0.8,
        "boredom": 0.9,
    }
    affect = {"survival_risk": 0.2, "altruism": 0.0, "fear": 0.4, "novelty": 0.2}
    weights = PriorityWeights(survival=1.0, altruism=0.0, spiritual=0.0)
    stop = score_candidate(
        name="stop",
        action=np.asarray([0.0, 0.0, 0.5, 0.3, 0.0], dtype=np.float32),
        signals=signals,
        affect=affect,
        weights=weights,
    )
    turn = score_candidate(
        name="turn_left",
        action=np.asarray([0.15, 0.85, 0.5, 0.5, 0.0], dtype=np.float32),
        signals=signals,
        affect=affect,
        weights=weights,
    )
    assert turn.value > stop.value
    calm_stop = apply_boredom_to_candidate_value(name="stop", value=0.9, boredom=0.0)
    bored_stop = apply_boredom_to_candidate_value(name="stop", value=0.9, boredom=1.0)
    assert bored_stop < calm_stop
