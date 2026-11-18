"""Phase 8 curiosity / novelty / exploration coverage tests."""

from __future__ import annotations

import numpy as np

from flybrain.curiosity import CuriosityConfig, CuriosityDrive, NoveltyTracker, curiosity_gate
from flybrain.env_explore import FakeExplorationEnv
from flybrain.phase8 import MIN_COVERAGE_CELLS_CURIOUS
from flybrain.pipeline.live_mind import LiveMindConfig, LiveMindPipeline, make_live_env
from flybrain.training.rewards import HomeostaticRewardConfig, compute_total_reward


def test_novelty_first_visit_then_diminishes() -> None:
    tracker = NoveltyTracker()
    pos = np.asarray([0.05, 0.05], dtype=np.float32)
    first = tracker.observe(position=pos, events=["landmark_0"])
    second = tracker.observe(position=pos, events=["landmark_0"])
    assert first.first_visit
    assert first.novelty_score > second.novelty_score
    assert first.event_new
    assert not second.event_new
    assert tracker.exploration_xp > 0.0


def test_curiosity_gate_suppresses_under_threat() -> None:
    cfg = CuriosityConfig()
    open_gate = curiosity_gate(
        signals={"energy": 0.8, "hunger": 0.2},
        affect={"survival_risk": 0.1},
        config=cfg,
    )
    closed = curiosity_gate(
        signals={"energy": 0.2, "hunger": 0.8},
        affect={"survival_risk": 0.9},
        config=cfg,
    )
    assert open_gate > 0.4
    assert closed == 0.0
    # Food-seek band (hunger ≥ 0.45 / energy < 0.55) also closes explore.
    seeking = curiosity_gate(
        signals={"energy": 0.5, "hunger": 0.5},
        affect={"survival_risk": 0.2},
        config=cfg,
    )
    assert seeking == 0.0


def test_curiosity_drive_prediction_and_bias() -> None:
    drive = CuriosityDrive(CuriosityConfig(enabled=True))
    emb0 = np.zeros(272, dtype=np.float32)
    emb1 = np.ones(272, dtype=np.float32) * 0.2
    signals = {"energy": 0.9, "hunger": 0.1}
    affect = {"survival_risk": 0.1, "exploration": 0.2, "novelty": 0.2}
    step0 = drive.step(
        embedding=emb0,
        action=np.asarray([0.5, 0.1, 0.5, 0.5, 0.0], dtype=np.float32),
        signals=signals,
        affect=affect,
        position=np.asarray([0.0, 0.0], dtype=np.float32),
    )
    step1 = drive.step(
        embedding=emb1,
        action=np.asarray([0.6, -0.2, 0.5, 0.5, 0.0], dtype=np.float32),
        signals=signals,
        affect=affect,
        position=np.asarray([0.15, 0.0], dtype=np.float32),
    )
    assert step0.gate > 0.0
    assert step1.prediction_error >= 0.0
    biased = drive.apply_action_bias(
        np.asarray([0.1, 0.0, 0.5, 0.4, 0.0], dtype=np.float32),
        explore_bias=0.8,
    )
    assert biased[0] > 0.1


def test_reward_includes_curiosity() -> None:
    base = compute_total_reward(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    with_c = compute_total_reward(
        0.8,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        config=HomeostaticRewardConfig(curiosity_weight=0.35),
    )
    assert with_c > base


def test_fake_explore_coverage_grows_with_explore_policy() -> None:
    env = FakeExplorationEnv()
    env.reset(seed=3)
    for _ in range(80):
        action = env.explore_policy_action()
        _, _, term, trunc, info = env.step(action)
        if term or trunc:
            break
    assert info["unique_cells"] >= MIN_COVERAGE_CELLS_CURIOUS
    assert info["coverage"] > 0.0
    assert info["exploration_xp"] > 0.0


def test_curious_live_mind_covers_more_than_disabled() -> None:
    steps = 60

    def run(*, enabled: bool) -> dict:
        env = make_live_env(fake_explore=True)
        cfg = LiveMindConfig(
            backend="in_memory",
            max_steps=steps,
            llm_mock=True,
            fly_id="fly_explore",
            mind_interval=1,
            curiosity=CuriosityConfig(enabled=enabled),
            use_vision_head=False,
        )
        pipeline = LiveMindPipeline(env, config=cfg)
        try:
            return pipeline.run(max_steps=steps)
        finally:
            pipeline.close()

    curious = run(enabled=True)
    baseline = run(enabled=False)
    assert curious["unique_cells"] >= baseline["unique_cells"]
    assert curious["curiosity_reward_mean"] >= 0.0
    assert curious["last"]["stm_id"] is not None
    # With curiosity on and sated start, gate should open at least once.
    assert curious["curiosity"]["curiosity_steps"] == steps


def test_hungry_still_prefers_feeding_over_pure_explore() -> None:
    """Gate must close when hungry so food-search behavior remains dominant."""
    closed = curiosity_gate(
        signals={"energy": 0.3, "hunger": 0.7, "sugar": 0.6},
        affect={"survival_risk": 0.2},
        config=CuriosityConfig(hunger_max=0.55, energy_min=0.4),
    )
    assert closed == 0.0
