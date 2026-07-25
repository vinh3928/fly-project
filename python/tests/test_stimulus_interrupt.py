"""Unit tests for stimulus-interrupt (3×3 novelty → STM patch history)."""

from __future__ import annotations

import numpy as np

from flybrain.memory.pgvector import InMemoryShortTermMemory
from flybrain.phase4 import PATCH_VEC_DIM, TOTAL_EMBEDDING_DIM
from flybrain.pipeline.live_mind import LiveMindConfig, LiveMindPipeline
from flybrain.thinking.retrospective import RetrospectiveConfig
from flybrain.thinking.stimulus_interrupt import (
    STOP_GOAL,
    PatchGridHistoryCache,
    StimulusInterruptConfig,
    best_history_similarity,
    best_patch_stm_similarity,
    evaluate_stimulus_interrupt,
    has_candidate_cue,
    part_vector_for_tests,
)
from flybrain.vision.novelty import stereo_patch_embedding


def _gray(val: float, size: int = 32) -> np.ndarray:
    return np.full((size, size), val, dtype=np.float32)


def _rgb_floor(size: int = 32) -> np.ndarray:
    return np.full((3, size, size), 0.48, dtype=np.float32)


def test_candidate_cue_taste_and_visual() -> None:
    cfg = StimulusInterruptConfig(min_visual_novelty=0.18)
    ok, cue = has_candidate_cue(
        signals={"sugar": 0.2, "bitter": 0.0},
        visual_novelty=0.0,
        new_stimulus=False,
        config=cfg,
        stimulus_thresholds={"sugar": 0.15, "bitter": 0.2},
    )
    assert ok and cue == "sugar"
    ok, cue = has_candidate_cue(
        signals={"sugar": 0.0, "bitter": 0.0},
        visual_novelty=0.5,
        new_stimulus=False,
        config=cfg,
    )
    assert ok and cue == "visual_novelty"


def test_empty_patch_stm_arms_on_visual_cue() -> None:
    mem = InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM)
    cfg = StimulusInterruptConfig(history_backend="patch_grid", cooldown_steps=0)
    eye = _gray(0.3)
    decision = evaluate_stimulus_interrupt(
        memory=mem,
        signals={},
        affect={"survival_risk": 0.1},
        step=1,
        cooldown_until_step=0,
        config=cfg,
        visual_novelty=0.5,
        new_stimulus=True,
        left_eye=eye,
        right_eye=eye,
    )
    assert decision.candidate
    assert decision.history_miss
    assert decision.should_arm
    assert decision.reason == "empty_patch_stm"
    assert decision.patch_embedding is not None
    assert decision.patch_embedding.shape[0] == PATCH_VEC_DIM
    mem.close()


def test_stm_patch_match_is_history_hit_not_saved_locally() -> None:
    """Confirmed interrupt patch in STM ⇒ next identical cue is not novel."""
    mem = InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM)
    cfg = StimulusInterruptConfig(
        history_backend="patch_grid",
        patch_match_min=0.92,
        cooldown_steps=0,
    )
    eye = _gray(0.4)
    patch = stereo_patch_embedding(eye, eye)
    mem.insert_embedding(
        embedding=np.zeros(TOTAL_EMBEDDING_DIM, dtype=np.float32),
        signals={"stimulus_interrupt": True},
        affect={},
        fly_id="fly_001",
        patch_vec=patch,
    )
    decision = evaluate_stimulus_interrupt(
        memory=mem,
        signals={},
        affect={"survival_risk": 0.1},
        fly_id="fly_001",
        step=5,
        cooldown_until_step=0,
        config=cfg,
        visual_novelty=0.6,
        new_stimulus=True,
        left_eye=eye,
        right_eye=eye,
    )
    assert decision.candidate
    assert not decision.history_miss
    assert not decision.should_arm
    assert decision.reason == "history_hit"
    assert decision.best_similarity >= 0.92
    mem.close()


def test_color_puddle_novel_vs_saved_floor_arms() -> None:
    mem = InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM)
    cfg = StimulusInterruptConfig(
        history_backend="patch_grid",
        patch_match_min=0.98,  # tight: small chroma shift must miss
        cooldown_steps=0,
    )
    floor = _rgb_floor()
    mem.insert_embedding(
        embedding=np.zeros(TOTAL_EMBEDDING_DIM, dtype=np.float32),
        signals={"stimulus_interrupt": True},
        affect={},
        fly_id="fly_001",
        patch_vec=stereo_patch_embedding(floor, floor),
    )
    # Entirely different scene (bright gold wash) vs gray floor.
    sugar = np.zeros_like(floor)
    sugar[0, :, :] = 0.95
    sugar[1, :, :] = 0.75
    sugar[2, :, :] = 0.15
    decision = evaluate_stimulus_interrupt(
        memory=mem,
        signals={},
        affect={"survival_risk": 0.1},
        fly_id="fly_001",
        step=2,
        cooldown_until_step=0,
        config=cfg,
        visual_novelty=0.5,
        new_stimulus=True,
        left_eye=sugar,
        right_eye=sugar,
    )
    assert decision.should_arm
    assert decision.reason == "novel_vs_patches"
    assert decision.history_miss
    assert decision.best_similarity < 0.98
    mem.close()


def test_single_novel_square_arms_despite_familiar_grid() -> None:
    """One unfamiliar 3×3 square must arm even though ~17 squares match STM.

    Under whole-vector cosine this washed out to ~0.99 (history_hit); per-square
    matching lets the lone novel square drive the miss.
    """
    mem = InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM)
    cfg = StimulusInterruptConfig(
        history_backend="patch_grid",
        patch_match_min=0.95,
        cooldown_steps=0,
    )
    floor = _rgb_floor()  # uniform gray floor, (3, 32, 32)
    mem.insert_embedding(
        embedding=np.zeros(TOTAL_EMBEDDING_DIM, dtype=np.float32),
        signals={"stimulus_interrupt": True},
        affect={},
        fly_id="fly_001",
        patch_vec=stereo_patch_embedding(floor, floor),
    )
    # Same floor, but the top-left 3×3 square is a strong gold wash.
    novel = floor.copy()
    novel[0, 0:10, 0:10] = 0.95
    novel[1, 0:10, 0:10] = 0.75
    novel[2, 0:10, 0:10] = 0.10
    decision = evaluate_stimulus_interrupt(
        memory=mem,
        signals={},
        affect={"survival_risk": 0.1},
        fly_id="fly_001",
        step=3,
        cooldown_until_step=0,
        config=cfg,
        visual_novelty=0.5,
        new_stimulus=True,
        left_eye=novel,
        right_eye=novel,
    )
    assert decision.should_arm
    assert decision.reason == "novel_vs_patches"
    assert decision.history_miss
    assert decision.best_similarity < 0.95
    assert decision.patch_worst_cell == 0  # top-left square of the left eye
    mem.close()


def test_clear_interrupt_persists_patch_to_stm() -> None:
    from flybrain.env import FakeFlyEnv

    env = FakeFlyEnv()
    mem = InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM)
    pipe = LiveMindPipeline(
        env,
        config=LiveMindConfig(
            backend="in_memory",
            llm_mock=True,
            enable_motor_layer=True,
            mind_interval=40,
            stimulus_interrupt=StimulusInterruptConfig(
                enabled=True,
                history_backend="patch_grid",
                cooldown_steps=0,
            ),
        ),
        memory=mem,
    )
    eye = _gray(0.55)
    patch = stereo_patch_embedding(eye, eye)
    pipe._interrupt_pending = True
    pipe._interrupt_reason = "novel_vs_patches"
    pipe._interrupt_patch_vec = patch
    pipe._step = 7
    pipe._clear_stimulus_interrupt()
    assert pipe._interrupt_pending is False
    sim, empty, _worst = best_patch_stm_similarity(mem, patch, fly_id="fly_001")
    assert not empty
    assert sim >= 0.99
    # Second identical cue must history-hit.
    d = evaluate_stimulus_interrupt(
        memory=mem,
        signals={},
        affect={"survival_risk": 0.1},
        fly_id="fly_001",
        step=8,
        cooldown_until_step=0,
        config=StimulusInterruptConfig(history_backend="patch_grid", cooldown_steps=0),
        visual_novelty=0.5,
        new_stimulus=True,
        left_eye=eye,
        right_eye=eye,
    )
    assert d.reason == "history_hit"
    assert not d.should_arm
    pipe.close()


def test_taste_cue_bypasses_familiar_patches() -> None:
    mem = InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM)
    cfg = StimulusInterruptConfig(history_backend="patch_grid", cooldown_steps=0)
    eye = _gray(0.5)
    mem.insert_embedding(
        embedding=np.zeros(TOTAL_EMBEDDING_DIM, dtype=np.float32),
        signals={},
        affect={},
        fly_id="fly_001",
        patch_vec=stereo_patch_embedding(eye, eye),
    )
    decision = evaluate_stimulus_interrupt(
        memory=mem,
        signals={"sugar": 0.5, "bitter": 0.0},
        affect={"survival_risk": 0.1},
        fly_id="fly_001",
        step=1,
        cooldown_until_step=0,
        config=cfg,
        stimulus_thresholds={"sugar": 0.15, "bitter": 0.2},
        retrospective=RetrospectiveConfig(),
        left_eye=eye,
        right_eye=eye,
    )
    assert decision.cue == "sugar"
    assert decision.should_arm
    mem.close()


def test_stm_backend_still_works() -> None:
    mem = InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM)
    parts = part_vector_for_tests(eye=1.0, sensors=0.5)
    mem.insert_embedding(
        embedding=parts.embedding,
        signals={},
        affect={},
        fly_id="fly_001",
        left_eye_vec=parts.left_eye_vec,
        right_eye_vec=parts.right_eye_vec,
        sensors_vec=parts.sensors_vec,
    )
    min_sim, scores, empty = best_history_similarity(mem, parts, fly_id="fly_001")
    assert not empty and min_sim >= 0.99
    cfg = StimulusInterruptConfig(
        history_backend="stm",
        history_relative=False,
        history_match_min=0.55,
        cooldown_steps=0,
    )
    decision = evaluate_stimulus_interrupt(
        memory=mem,
        parts=parts,
        signals={},
        affect={"survival_risk": 0.1},
        fly_id="fly_001",
        step=1,
        cooldown_until_step=0,
        config=cfg,
        visual_novelty=0.5,
        new_stimulus=True,
    )
    assert decision.reason == "history_hit"
    mem.close()


def test_evaluate_crisis_skips_interrupt() -> None:
    mem = InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM)
    cfg = StimulusInterruptConfig(history_backend="patch_grid", cooldown_steps=0)
    eye = _gray(0.2)
    decision = evaluate_stimulus_interrupt(
        memory=mem,
        signals={"sugar": 0.5, "bitter": 0.9},
        affect={"survival_risk": 0.1},
        step=1,
        cooldown_until_step=0,
        config=cfg,
        stimulus_thresholds={"sugar": 0.15, "bitter": 0.2},
        left_eye=eye,
        right_eye=eye,
    )
    assert decision.crisis and not decision.should_arm
    mem.close()


def test_pipeline_hard_stop_while_pending() -> None:
    from flybrain.env import FakeFlyEnv

    env = FakeFlyEnv()
    mem = InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM)
    pipe = LiveMindPipeline(
        env,
        config=LiveMindConfig(
            backend="in_memory",
            llm_mock=True,
            enable_motor_layer=True,
            mind_interval=40,
            stimulus_interrupt=StimulusInterruptConfig(enabled=True, cooldown_steps=0),
        ),
        memory=mem,
    )
    obs, _ = env.reset()
    pipe._last_mind_info = {"intent": {"future": {"chosen_name": "explore"}}, "affect": {}}
    pipe._step = 1
    pipe._interrupt_pending = True
    pipe._interrupt_reason = "test"
    action, info = pipe.process_motor_only(obs)
    assert info.get("stimulus_interrupt_pending") is True
    assert abs(float(action[0])) <= 0.08
    pipe.close()


def test_config_from_yaml_loads_stimulus_interrupt() -> None:
    from flybrain.pipeline.live_mind import load_mind_config

    raw = load_mind_config()
    cfg = LiveMindConfig.from_yaml(raw)
    assert cfg.stimulus_interrupt is not None
    assert cfg.stimulus_interrupt.history_backend == "patch_grid"
    assert cfg.stimulus_interrupt.patch_match_min == 0.92


def test_patch_grid_history_cache_api_compat() -> None:
    cache = PatchGridHistoryCache.from_config(StimulusInterruptConfig())
    cache.reset()
