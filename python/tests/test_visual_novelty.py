"""Hexagonal eye aperture and triangular-facet novelty."""

from __future__ import annotations

import numpy as np

from flybrain.phase4 import HEX_FACET_COUNT, PATCH_VEC_DIM
from flybrain.vision.hex_eye import (
    apply_hex_aperture,
    facet_centroid_bearing,
    hex_aperture_mask,
    hex_triangle_masks,
)
from flybrain.vision.novelty import (
    VisualNoveltyConfig,
    VisualNoveltyCues,
    VisualNoveltyTracker,
    apply_visual_explore_goal,
    stereo_patch_embedding,
)


def _frame_with_blob(*, bearing: float = 0.6, size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    frame = np.full((size, size), 0.2, dtype=np.float32)
    cx = int((bearing + 1.0) * 0.5 * (size - 1))
    r = max(2, size // 8)
    y0, y1 = size // 2 - r, size // 2 + r
    x0, x1 = max(0, cx - r), min(size, cx + r)
    frame[y0:y1, x0:x1] = 0.95
    return frame, frame.copy()


def test_hex_aperture_zeros_corners() -> None:
    mask = hex_aperture_mask(64, 64)
    assert mask.shape == (64, 64)
    assert not mask[0, 0]
    assert not mask[0, 63]
    assert not mask[63, 0]
    assert mask[32, 32]
    frame = np.ones((3, 64, 64), dtype=np.float32)
    masked = apply_hex_aperture(frame)
    assert float(masked[:, 0, 0].sum()) == 0.0
    assert float(masked[:, 32, 32].mean()) > 0.5


def test_six_triangle_masks_cover_hex() -> None:
    masks = hex_triangle_masks(48, 48)
    assert len(masks) == HEX_FACET_COUNT
    union = np.zeros((48, 48), dtype=bool)
    for m in masks:
        assert m.any()
        union |= m
    np.testing.assert_array_equal(union, hex_aperture_mask(48, 48))


def test_stereo_patch_embedding_is_60d() -> None:
    left = np.full((3, 32, 32), 0.4, dtype=np.float32)
    right = np.full((3, 32, 32), 0.5, dtype=np.float32)
    emb = stereo_patch_embedding(left, right)
    assert emb.shape == (PATCH_VEC_DIM,)
    assert PATCH_VEC_DIM == 60


def test_observe_returns_twelve_independent_facets() -> None:
    tracker = VisualNoveltyTracker(VisualNoveltyConfig())
    blank = np.full((32, 32), 0.2, dtype=np.float32)
    cues = tracker.observe(blank, blank)
    assert len(cues.patches) == 12  # 2 eyes × 6 facets
    left = [p for p in cues.patches if p.eye == "left"]
    right = [p for p in cues.patches if p.eye == "right"]
    assert len(left) == 6 and len(right) == 6
    assert {p.col for p in left} == set(range(6))


def test_eyes_are_not_stereo_averaged() -> None:
    tracker = VisualNoveltyTracker(VisualNoveltyConfig(habituation_enabled=False))
    blank = np.full((32, 32), 0.2, dtype=np.float32)
    tracker.observe(blank, blank)
    left_only = blank.copy()
    right_blob = blank.copy()
    right_blob[14:22, 20:28] = 0.95
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
    left[20:30, 30:42] = 0.95
    cues = tracker.observe(left, right)
    assert cues.peak is not None
    assert cues.visual_novelty == cues.peak.visual_novelty
    assert cues.novelty_bearing == cues.peak.novelty_bearing
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
    tracker.observe(left, right)
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
    tracker = VisualNoveltyTracker(VisualNoveltyConfig(habituation_enabled=False))
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
                row=0,
                col=5,
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


def test_facet_bearing_right_sector_is_positive() -> None:
    # Facets near +x should have positive bearing.
    rights = [facet_centroid_bearing(i) for i in range(6)]
    assert max(rights) > 0.2
    assert min(rights) < -0.2


def test_legacy_n_columns_maps_to_hex_facets() -> None:
    cfg = VisualNoveltyConfig.from_dict({"n_columns": 8})
    assert cfg.facet_count == 6
    assert cfg.grid_cols == 6
