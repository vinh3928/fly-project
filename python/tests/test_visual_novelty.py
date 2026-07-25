"""Eye-driven visual novelty for explore motor (independent 3×3 per eye)."""

from __future__ import annotations

import numpy as np

from flybrain.vision.novelty import (
    VisualNoveltyConfig,
    VisualNoveltyCues,
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


def test_observe_returns_eighteen_independent_patches() -> None:
    tracker = VisualNoveltyTracker(VisualNoveltyConfig())
    blank = np.full((32, 32), 0.2, dtype=np.float32)
    cues = tracker.observe(blank, blank)
    assert len(cues.patches) == 18  # 2 eyes × 3×3
    left = [p for p in cues.patches if p.eye == "left"]
    right = [p for p in cues.patches if p.eye == "right"]
    assert len(left) == 9 and len(right) == 9
    assert {(p.row, p.col) for p in left} == {(r, c) for r in range(3) for c in range(3)}


def test_eyes_are_not_stereo_averaged() -> None:
    """Novelty in the right eye alone must not require the left eye to change."""
    tracker = VisualNoveltyTracker(VisualNoveltyConfig(habituation_enabled=False))
    blank = np.full((32, 32), 0.2, dtype=np.float32)
    tracker.observe(blank, blank)
    left_only = blank.copy()
    right_blob = blank.copy()
    right_blob[20:28, 20:28] = 0.95
    cues = tracker.observe(left_only, right_blob)
    right_peak = max((p for p in cues.patches if p.eye == "right"), key=lambda p: p.visual_novelty)
    left_peak = max((p for p in cues.patches if p.eye == "left"), key=lambda p: p.visual_novelty)
    assert right_peak.visual_novelty > left_peak.visual_novelty
    assert cues.peak is not None and cues.peak.eye == "right"


def test_new_visual_blob_raises_novelty_and_bearing() -> None:
    tracker = VisualNoveltyTracker(VisualNoveltyConfig())
    blank = np.full((32, 32), 0.2, dtype=np.float32)
    first = tracker.observe(blank, blank)
    left, right = _frame_with_blob(bearing=0.7)
    second = tracker.observe(left, right)
    assert second.visual_novelty > first.visual_novelty
    assert second.novelty_bearing > 0.0
    assert second.new_stimulus or second.visual_novelty >= 0.18


def test_scalar_is_peak_not_blend() -> None:
    tracker = VisualNoveltyTracker(VisualNoveltyConfig(habituation_enabled=False))
    blank = np.full((48, 48), 0.2, dtype=np.float32)
    tracker.observe(blank, blank)
    left = blank.copy()
    right = blank.copy()
    # Bright blob only in left eye, lower-right cell.
    left[32:48, 32:48] = 0.95
    cues = tracker.observe(left, right)
    assert cues.peak is not None
    assert cues.visual_novelty == cues.peak.visual_novelty
    assert cues.novelty_bearing == cues.peak.novelty_bearing
    # Other patches keep their own scores; scalar is not a mean of all 18.
    mean_all = float(np.mean([p.visual_novelty for p in cues.patches]))
    assert cues.visual_novelty > mean_all


def test_repeat_view_decays_novelty() -> None:
    tracker = VisualNoveltyTracker(VisualNoveltyConfig(repeat_decay=0.5))
    left, right = _frame_with_blob(bearing=-0.5)
    a = tracker.observe(left, right)
    b = tracker.observe(left, right)
    c = tracker.observe(left, right)
    assert a.visual_novelty >= b.visual_novelty >= c.visual_novelty * 0.9


def test_static_view_habituates_novelty_slowly() -> None:
    tracker = VisualNoveltyTracker(
        VisualNoveltyConfig(
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
        VisualNoveltyConfig(habituation_rate=0.2, dishabituation_rate=1.0)
    )
    wall, _ = _frame_with_blob(bearing=-0.6)
    for _ in range(30):
        habituated = tracker.observe(wall, wall)
    bright = np.full((32, 32), 0.9, dtype=np.float32)
    dark = np.zeros((32, 32), dtype=np.float32)
    recovered = tracker.observe(bright, dark)
    assert recovered.visual_novelty > habituated.visual_novelty


def test_habituation_can_be_disabled() -> None:
    tracker = VisualNoveltyTracker(
        VisualNoveltyConfig(habituation_enabled=False)
    )
    left, right = _frame_with_blob(bearing=0.4)
    tracker.observe(left, right)
    for _ in range(30):
        out = tracker.observe(left, right)
    assert out.visual_novelty > 0.0


def test_apply_visual_explore_goal_turns_toward_novelty() -> None:
    from flybrain.vision.novelty import PatchNovelty

    goal = np.asarray([0.3, 0.0, 0.5, 0.5, 0.0], dtype=np.float32)
    cues = VisualNoveltyCues(
        patches=[
            PatchNovelty(
                eye="left",
                row=1,
                col=0,
                visual_novelty=0.8,
                novelty_bearing=-0.7,
                novelty_confidence=0.9,
                new_stimulus=True,
            )
        ]
    )
    out = apply_visual_explore_goal(goal, cues, wall_hazard=0.2)
    assert float(out[1]) < -0.2
    assert float(out[0]) >= float(goal[0])


def test_color_puddle_registers_as_novel_vs_gray_floor() -> None:
    """Sugar-gold chroma should flip patch signatures vs floor-gray of similar luma."""
    size = 32
    floor = np.full((3, size, size), 0.48, dtype=np.float32)
    tracker = VisualNoveltyTracker(VisualNoveltyConfig(habituation_enabled=False))
    tracker.observe(floor, floor)
    sugar = floor.copy()
    sugar[0, size // 2 :, size // 2 :] = 0.95
    sugar[1, size // 2 :, size // 2 :] = 0.75
    sugar[2, size // 2 :, size // 2 :] = 0.20
    hit = tracker.observe(sugar, sugar)
    assert hit.new_stimulus or hit.visual_novelty >= 0.18

    tracker2 = VisualNoveltyTracker(VisualNoveltyConfig(habituation_enabled=False))
    tracker2.observe(floor, floor)
    wash = floor.copy()
    wash[:, size // 2 :, size // 2 :] = 0.63
    muted = tracker2.observe(wash, wash)
    assert hit.visual_novelty + 1e-6 >= muted.visual_novelty * 0.5


def test_legacy_n_columns_maps_to_grid() -> None:
    cfg = VisualNoveltyConfig.from_dict({"n_columns": 8})
    assert cfg.grid_rows == 3 and cfg.grid_cols == 3
