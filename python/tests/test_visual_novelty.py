"""Eye-driven visual novelty for explore motor."""

from __future__ import annotations

import numpy as np

from flybrain.vision.novelty import (
    VisualNoveltyConfig,
    VisualNoveltyTracker,
    apply_visual_explore_goal,
)


def _frame_with_blob(*, bearing: float = 0.6, size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    frame = np.full((size, size), 0.2, dtype=np.float32)
    cx = int((bearing + 1.0) * 0.5 * (size - 1))
    r = max(2, size // 8)
    y0, y1 = size // 2 - r, size // 2 + r
    x0, x1 = max(0, cx - r), min(size, cx + r)
    frame[y0:y1, x0:x1] = 0.95
    return frame, frame.copy()


def test_new_visual_blob_raises_novelty_and_bearing() -> None:
    tracker = VisualNoveltyTracker(VisualNoveltyConfig(n_columns=8))
    blank = np.full((32, 32), 0.2, dtype=np.float32)
    first = tracker.observe(blank, blank)
    left, right = _frame_with_blob(bearing=0.7)
    second = tracker.observe(left, right)
    assert second.visual_novelty > first.visual_novelty
    assert second.novelty_bearing > 0.2
    assert second.new_stimulus or second.visual_novelty >= 0.18


def test_repeat_view_decays_novelty() -> None:
    tracker = VisualNoveltyTracker(VisualNoveltyConfig(n_columns=8, repeat_decay=0.5))
    left, right = _frame_with_blob(bearing=-0.5)
    a = tracker.observe(left, right)
    b = tracker.observe(left, right)
    c = tracker.observe(left, right)
    assert a.visual_novelty >= b.visual_novelty >= c.visual_novelty * 0.9


def test_static_view_habituates_novelty_slowly() -> None:
    tracker = VisualNoveltyTracker(
        VisualNoveltyConfig(
            n_columns=8,
            habituation_enabled=True,
            habituation_rate=0.1,
            habituation_strength=0.9,
        )
    )
    left, right = _frame_with_blob(bearing=0.4)
    tracker.observe(left, right)  # first exposure
    early = tracker.observe(left, right)
    for _ in range(40):
        late = tracker.observe(left, right)
    assert late.visual_novelty < early.visual_novelty
    assert late.visual_novelty <= 0.2 * early.visual_novelty + 1e-6


def test_scene_change_recovers_novelty_after_habituation() -> None:
    tracker = VisualNoveltyTracker(
        VisualNoveltyConfig(n_columns=8, habituation_rate=0.2, dishabituation_rate=1.0)
    )
    wall, _ = _frame_with_blob(bearing=-0.6)
    for _ in range(30):
        habituated = tracker.observe(wall, wall)
    # A very different scene should dishabituate and read as more novel again.
    bright = np.full((32, 32), 0.9, dtype=np.float32)
    dark = np.zeros((32, 32), dtype=np.float32)
    recovered = tracker.observe(bright, dark)
    assert recovered.visual_novelty > habituated.visual_novelty


def test_habituation_can_be_disabled() -> None:
    tracker = VisualNoveltyTracker(
        VisualNoveltyConfig(n_columns=8, habituation_enabled=False)
    )
    left, right = _frame_with_blob(bearing=0.4)
    tracker.observe(left, right)
    for _ in range(30):
        out = tracker.observe(left, right)
    # Without habituation, the per-signature decay floors but novelty persists.
    assert out.visual_novelty > 0.0


def test_apply_visual_explore_goal_turns_toward_novelty() -> None:
    goal = np.asarray([0.3, 0.0, 0.5, 0.5, 0.0], dtype=np.float32)
    from flybrain.vision.novelty import VisualNoveltyCues

    cues = VisualNoveltyCues(
        visual_novelty=0.8,
        novelty_bearing=-0.7,
        novelty_confidence=0.9,
        new_stimulus=True,
        unique_patches=3,
    )
    out = apply_visual_explore_goal(goal, cues, wall_hazard=0.2)
    assert float(out[1]) < -0.2
    assert float(out[0]) >= float(goal[0])
