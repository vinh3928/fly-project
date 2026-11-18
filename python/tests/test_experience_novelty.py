"""Experience novelty: normalized full-embedding distance to remembered experiences."""

from __future__ import annotations

import numpy as np
import pytest

from flybrain.curiosity import CuriosityConfig, CuriosityDrive, NoveltyConfig, NoveltyTracker
from flybrain.curiosity.experience import (
    ExperienceNovelty,
    ExperienceNoveltyConfig,
    RunningStats,
)
from flybrain.phase4 import TOTAL_EMBEDDING_DIM
from flybrain.phase8 import EXPERIENCE_DIMS
from flybrain.pipeline.live_mind import format_live_step_log


def _small(**kwargs) -> ExperienceNoveltyConfig:
    base = dict(dims=4, memory_size=50, knn=2, recent_exclude=0, history_size=50, warmup=3)
    base.update(kwargs)
    return ExperienceNoveltyConfig(**base)


def test_running_stats_match_numpy() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(40, 3)) * [1.0, 5.0, 0.1] + [2.0, -1.0, 0.0]
    stats = RunningStats(3, min_std=1e-6)
    for row in data:
        stats.update(row)
    assert np.allclose(stats.mean, data.mean(axis=0))
    assert np.allclose(stats.std, data.std(axis=0))
    z = stats.normalize(data[0])
    assert np.allclose(z, (data[0] - data.mean(axis=0)) / data.std(axis=0))


def test_running_stats_floor_spread_on_constant_dims() -> None:
    stats = RunningStats(2, min_std=0.5)
    for _ in range(5):
        stats.update(np.array([1.0, 1.0]))
    assert np.allclose(stats.std, [0.5, 0.5])


def test_first_observations_have_no_distance_or_novelty() -> None:
    exp = ExperienceNovelty(_small())
    first = exp.observe(np.ones(4))
    assert first.distance is None
    assert first.novelty == 0.0
    assert first.memory == 1


def test_warmup_keeps_novelty_zero_until_history_fills() -> None:
    exp = ExperienceNovelty(_small(warmup=5))
    rng = np.random.default_rng(1)
    results = [exp.observe(rng.normal(size=4)) for _ in range(5)]
    # 4 distances recorded (first had no memory) < warmup 5.
    assert all(r.novelty == 0.0 for r in results)


def test_new_experience_scores_high_after_repeats() -> None:
    exp = ExperienceNovelty(_small())
    rng = np.random.default_rng(2)
    familiar = np.array([1.0, 0.0, 0.0, 0.0])
    for _ in range(30):
        exp.observe(familiar + rng.normal(scale=0.01, size=4))
    repeat = exp.observe(familiar + rng.normal(scale=0.01, size=4))
    novel = exp.observe(np.array([0.0, 0.0, 1.0, 1.0]))
    assert novel.distance is not None and repeat.distance is not None
    assert novel.distance > repeat.distance
    assert novel.novelty == pytest.approx(1.0)
    assert novel.novelty > repeat.novelty


def test_normalization_exposes_small_spread_dimensions() -> None:
    """A dimension that barely varies still separates experiences once normalized."""
    exp = ExperienceNovelty(_small(dims=3, knn=1))
    base = np.array([10.0, 10.0, 0.0])
    for i in range(30):
        exp.observe(base + np.array([0.0, 0.0, 0.001 * (i % 2)]))
    shifted = exp.observe(base + np.array([0.0, 0.0, 0.01]))
    assert shifted.novelty > 0.5


def test_recent_memories_are_skipped() -> None:
    exp = ExperienceNovelty(_small(recent_exclude=3))
    for _ in range(4):
        result = exp.observe(np.ones(4))
    assert result.distance is None
    fifth = exp.observe(np.ones(4) * 2.0)
    assert fifth.distance is not None


def test_memory_is_a_rolling_buffer() -> None:
    exp = ExperienceNovelty(_small(memory_size=5))
    for i in range(12):
        exp.observe(np.full(4, float(i)))
    assert exp.memory_count == 5
    exp.reset()
    assert exp.memory_count == 0


def test_only_leading_dims_are_used() -> None:
    """The affect tail beyond ``dims`` must not change the distance."""
    a, b = ExperienceNovelty(_small()), ExperienceNovelty(_small())
    rng = np.random.default_rng(3)
    for _ in range(10):
        head = rng.normal(size=4)
        ra = a.observe(np.concatenate([head, [0.0, 0.0]]))
        rb = b.observe(np.concatenate([head, rng.normal(size=2) * 50.0]))
    assert ra.distance == pytest.approx(rb.distance)


def test_scale_maps_median_to_zero_and_p95_to_one() -> None:
    exp = ExperienceNovelty(_small(warmup=1, history_size=200))
    exp._history.extend(np.linspace(0.0, 1.0, 101).tolist())
    assert exp._scale(0.5) == pytest.approx(0.0)
    assert exp._scale(0.95) == pytest.approx(1.0)
    assert exp._scale(0.725) == pytest.approx(0.5)
    assert exp._scale(0.1) == 0.0


def test_scale_with_flat_history_is_binary() -> None:
    exp = ExperienceNovelty(_small(warmup=1))
    exp._history.extend([0.2] * 10)
    assert exp._scale(0.2) == 0.0
    assert exp._scale(0.3) == 1.0


def test_default_config_uses_fused_latent_only() -> None:
    assert ExperienceNoveltyConfig().dims == EXPERIENCE_DIMS < TOTAL_EMBEDDING_DIM


def test_tracker_scores_experience_novelty() -> None:
    tracker = NoveltyTracker(
        NoveltyConfig(experience_weight=0.6, experience=dict(dims=4, warmup=3, knn=2, recent_exclude=0))
    )
    rng = np.random.default_rng(4)
    for _ in range(30):
        tracker.observe(embedding=np.array([1.0, 0.0, 0.0, 0.0]) + rng.normal(scale=0.01, size=4))
    obs = tracker.observe(embedding=np.array([0.0, 0.0, 1.0, 1.0]))
    assert obs.experience_novelty == pytest.approx(1.0)
    assert obs.embedding_new
    assert obs.first_visit
    assert obs.novelty_score == pytest.approx(0.6)
    assert obs.details["experience_distance"] is not None
    assert tracker.summary()["experience_memory"] == 31


def test_drive_builds_experience_config_from_yaml_dict() -> None:
    cfg = CuriosityConfig.from_dict(
        {"novelty": {"experience_weight": 0.6, "experience": {"memory_size": 7, "knn": 3}}}
    )
    drive = CuriosityDrive(cfg)
    assert drive.tracker.config.experience_weight == 0.6
    assert drive.tracker.experience.config.memory_size == 7
    assert drive.tracker.experience.config.knn == 3


def test_log_line_shows_experience_novelty() -> None:
    line = format_live_step_log(
        5,
        np.zeros(5, dtype=np.float32),
        {"curiosity_gated": 0.12, "experience_novelty": 0.73},
    )
    assert "cur=0.120 exp=0.73" in line
