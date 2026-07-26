"""Unit tests for stimulus-interrupt (cue → joint STM history)."""

from __future__ import annotations

import numpy as np

from flybrain.memory.pgvector import InMemoryShortTermMemory
from flybrain.phase4 import (
    INTERRUPT_JOINT_VEC_DIM,
    PATCH_VEC_DIM,
    SENSORS_VEC_DIM,
    SMELL_VEC_DIM,
    TOTAL_EMBEDDING_DIM,
)
from flybrain.pipeline.live_mind import LiveMindConfig, LiveMindPipeline
from flybrain.signals.builder import ANTENNA_DIM, SignalBuilder, antenna_smell_vec
from flybrain.thinking.retrospective import RetrospectiveConfig
from flybrain.thinking.stimulus_interrupt import (
    STOP_GOAL,
    PassiveSmellGate,
    PatchGridHistoryCache,
    StimulusInterruptConfig,
    best_history_similarity,
    build_interrupt_joint_vec,
    cosine_similarity,
    evaluate_stimulus_interrupt,
    has_candidate_cue,
    part_vector_for_tests,
)
from flybrain.vision.novelty import stereo_patch_embedding


def _gray(val: float, size: int = 32) -> np.ndarray:
    return np.full((size, size), val, dtype=np.float32)


def _rgb_floor(size: int = 32) -> np.ndarray:
    return np.full((3, size, size), 0.48, dtype=np.float32)


def _smell(*, odor_l: float = 0.0, odor_r: float = 0.0, sugar_l: float = 0.0) -> np.ndarray:
    """Build a 16-d antenna smell vector (Odor=channel 7)."""
    v = np.zeros(SMELL_VEC_DIM, dtype=np.float32)
    v[0] = sugar_l
    v[7] = odor_l
    v[8 + 7] = odor_r
    return v


def _sensors(val: float = 0.5) -> np.ndarray:
    s = np.zeros(SENSORS_VEC_DIM, dtype=np.float32)
    s[0] = val
    return s


def _insert_joint(
    mem: InMemoryShortTermMemory,
    *,
    patch: np.ndarray,
    smell: np.ndarray,
    sensors: np.ndarray,
) -> np.ndarray:
    joint = build_interrupt_joint_vec(patch, smell, sensors)
    mem.insert_embedding(
        embedding=np.zeros(TOTAL_EMBEDDING_DIM, dtype=np.float32),
        signals={"stimulus_interrupt": True},
        affect={},
        fly_id="fly_001",
        sensors_vec=sensors,
        patch_vec=patch,
        smell_vec=smell,
        joint_vec=joint,
    )
    return joint


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


def test_build_interrupt_joint_vec_shape_and_unit() -> None:
    patch = np.ones(PATCH_VEC_DIM, dtype=np.float32)
    smell = _smell(odor_l=0.4, odor_r=0.3)
    sensors = _sensors(0.7)
    joint = build_interrupt_joint_vec(patch, smell, sensors)
    assert joint.shape == (INTERRUPT_JOINT_VEC_DIM,)
    assert abs(float(np.linalg.norm(joint)) - 1.0) < 1e-5


def test_empty_joint_stm_arms_on_visual_cue() -> None:
    mem = InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM)
    cfg = StimulusInterruptConfig(history_backend="joint", cooldown_steps=0)
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
        smell_vec=_smell(),
        sensors_vec=_sensors(0.2),
    )
    assert decision.candidate
    assert decision.history_miss
    assert decision.should_arm
    assert decision.reason == "empty_joint_stm"
    assert decision.patch_embedding is not None
    assert decision.joint_embedding is not None
    assert decision.joint_embedding.shape[0] == INTERRUPT_JOINT_VEC_DIM
    mem.close()


def test_smell_novelty_alone_arms() -> None:
    """STM-first smell miss opens cue without visual/taste; empty joint arms."""
    mem = InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM)
    cfg = StimulusInterruptConfig(
        history_backend="joint",
        smell_enabled=True,
        smell_match_min=0.92,
        cooldown_steps=0,
    )
    eye = _gray(0.4)
    decision = evaluate_stimulus_interrupt(
        memory=mem,
        signals={},
        affect={"survival_risk": 0.1},
        fly_id="fly_001",
        step=1,
        cooldown_until_step=0,
        config=cfg,
        visual_novelty=0.0,
        new_stimulus=False,
        left_eye=eye,
        right_eye=eye,
        smell_vec=_smell(odor_l=0.7, odor_r=0.65),
        sensors_vec=_sensors(0.3),
    )
    assert decision.candidate
    assert decision.cue == "smell_novelty"
    assert decision.should_arm
    assert decision.reason == "empty_joint_stm"
    mem.close()


def test_same_joint_is_history_hit() -> None:
    mem = InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM)
    cfg = StimulusInterruptConfig(
        history_backend="joint",
        patch_match_min=0.92,
        cooldown_steps=0,
    )
    eye = _gray(0.4)
    patch = stereo_patch_embedding(eye, eye)
    smell = _smell(odor_l=0.5, odor_r=0.45)
    sensors = _sensors(0.4)
    _insert_joint(mem, patch=patch, smell=smell, sensors=sensors)
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
        smell_vec=smell.copy(),
        sensors_vec=sensors.copy(),
    )
    assert decision.candidate
    assert not decision.history_miss
    assert not decision.should_arm
    assert decision.reason == "history_hit"
    assert decision.best_similarity >= 0.92
    assert "patches" in decision.modality_scores
    mem.close()


def test_visual_cue_ignores_novel_smell_on_l1() -> None:
    """L1 compares only the cue column: visual cue + same patches ⇒ history_hit."""
    mem = InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM)
    cfg = StimulusInterruptConfig(
        history_backend="joint",
        patch_match_min=0.92,
        smell_match_min=0.92,
        cooldown_steps=0,
    )
    eye = _gray(0.4)
    patch = stereo_patch_embedding(eye, eye)
    sensors = _sensors(0.5)
    _insert_joint(
        mem,
        patch=patch,
        smell=_smell(odor_l=0.1, odor_r=0.1, sugar_l=0.8),
        sensors=sensors,
    )
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
        left_eye=eye,
        right_eye=eye,
        smell_vec=_smell(odor_l=0.9, odor_r=0.85, sugar_l=0.0),
        sensors_vec=sensors.copy(),
    )
    assert decision.cue in ("new_stimulus", "visual_novelty")
    assert decision.reason == "history_hit"
    assert not decision.should_arm
    mem.close()


def test_smell_cue_novel_smell_arms_via_l1_column() -> None:
    """Smell cue → L1 compares smell_vec only among joint neighbors."""
    mem = InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM)
    cfg = StimulusInterruptConfig(
        history_backend="joint",
        smell_enabled=True,
        smell_match_min=0.92,
        cooldown_steps=0,
    )
    eye = _gray(0.4)
    patch = stereo_patch_embedding(eye, eye)
    sensors = _sensors(0.5)
    _insert_joint(
        mem,
        patch=patch,
        smell=_smell(sugar_l=0.9),
        sensors=sensors,
    )
    decision = evaluate_stimulus_interrupt(
        memory=mem,
        signals={},
        affect={"survival_risk": 0.1},
        fly_id="fly_001",
        step=2,
        cooldown_until_step=0,
        config=cfg,
        visual_novelty=0.0,
        new_stimulus=False,
        left_eye=eye,
        right_eye=eye,
        smell_vec=_smell(odor_l=0.95, odor_r=0.9),
        sensors_vec=sensors.copy(),
    )
    assert decision.cue == "smell_novelty"
    assert decision.should_arm
    assert decision.reason == "novel_vs_smell"
    assert decision.history_miss
    assert "smell" in decision.novel_modalities
    mem.close()


def test_familiar_vision_novel_smell_arms() -> None:
    """Backward-compat name: smell-alone path arms when smell column misses."""
    test_smell_cue_novel_smell_arms_via_l1_column()


def test_per_block_norm_smell_change_moves_joint() -> None:
    """Orthogonal smell directions must move joint cosine enough to miss at default thr."""
    patch = np.ones(PATCH_VEC_DIM, dtype=np.float32)
    sensors = _sensors(0.6)
    # Sugar-dominant vs odor-dominant ⇒ low smell cosine after per-block L2.
    j1 = build_interrupt_joint_vec(patch, _smell(sugar_l=0.9), sensors)
    j2 = build_interrupt_joint_vec(patch, _smell(odor_l=0.95, odor_r=0.9), sensors)
    sim = cosine_similarity(j1, j2)
    assert sim < 0.92
    # Equal modality weight: smell-only change still pulls joint below threshold.
    # (2 + 0) / 3 ≈ 0.67 when smell blocks are orthogonal.
    assert sim < 0.75


def test_zero_stored_joint_skipped_arms() -> None:
    """Old rows without joint_vec are skipped → empty_joint_stm arms."""
    mem = InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM)
    cfg = StimulusInterruptConfig(history_backend="joint", cooldown_steps=0)
    eye = _gray(0.38)
    patch = stereo_patch_embedding(eye, eye)
    mem.insert_embedding(
        embedding=np.zeros(TOTAL_EMBEDDING_DIM, dtype=np.float32),
        signals={"stimulus_interrupt": True},
        affect={},
        fly_id="fly_001",
        patch_vec=patch,
        smell_vec=_smell(odor_l=0.5, odor_r=0.5),
        # joint_vec defaults to zeros → skipped by similar_by
    )
    decision = evaluate_stimulus_interrupt(
        memory=mem,
        signals={},
        affect={"survival_risk": 0.1},
        fly_id="fly_001",
        step=4,
        cooldown_until_step=0,
        config=cfg,
        visual_novelty=0.5,
        new_stimulus=True,
        left_eye=eye,
        right_eye=eye,
        smell_vec=_smell(odor_l=0.5, odor_r=0.5),
        sensors_vec=_sensors(0.2),
    )
    assert decision.should_arm
    assert decision.reason == "empty_joint_stm"
    mem.close()


def test_clear_interrupt_persists_joint_to_stm() -> None:
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
                history_backend="joint",
                cooldown_steps=0,
            ),
        ),
        memory=mem,
    )
    eye = _gray(0.55)
    patch = stereo_patch_embedding(eye, eye)
    smell = _smell(odor_l=0.4, odor_r=0.35)
    sensors = _sensors(0.33)
    joint = build_interrupt_joint_vec(patch, smell, sensors)
    pipe._interrupt_pending = True
    pipe._interrupt_reason = "novel_vs_joint"
    pipe._interrupt_patch_vec = patch
    pipe._interrupt_smell_vec = smell
    pipe._interrupt_sensors_vec = sensors
    pipe._interrupt_joint_vec = joint
    pipe._step = 7
    pipe._clear_stimulus_interrupt()
    assert pipe._interrupt_pending is False
    recent = mem.recent(fly_id="fly_001", limit=1)
    assert recent
    assert float(np.linalg.norm(recent[0].joint_vec)) > 0.1
    assert float(np.max(np.abs(recent[0].smell_vec - smell))) < 1e-5
    # Second identical cue must history-hit.
    d = evaluate_stimulus_interrupt(
        memory=mem,
        signals={},
        affect={"survival_risk": 0.1},
        fly_id="fly_001",
        step=8,
        cooldown_until_step=0,
        config=StimulusInterruptConfig(history_backend="joint", cooldown_steps=0),
        visual_novelty=0.5,
        new_stimulus=True,
        left_eye=eye,
        right_eye=eye,
        smell_vec=smell,
        sensors_vec=sensors,
    )
    assert d.reason == "history_hit"
    assert not d.should_arm
    pipe.close()


def test_color_puddle_novel_vs_saved_floor_arms() -> None:
    mem = InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM)
    cfg = StimulusInterruptConfig(
        history_backend="joint",
        patch_match_min=0.98,
        cooldown_steps=0,
    )
    floor = _rgb_floor()
    sensors = _sensors(0.4)
    _insert_joint(
        mem,
        patch=stereo_patch_embedding(floor, floor),
        smell=_smell(),
        sensors=sensors,
    )
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
        smell_vec=_smell(),
        sensors_vec=sensors.copy(),
    )
    assert decision.should_arm
    assert decision.reason == "novel_vs_patches"
    assert decision.history_miss
    assert decision.best_similarity < 0.98
    mem.close()


def test_antenna_smell_vec_from_observation() -> None:
    obs = np.zeros(197, dtype=np.float32)
    start = 72 + 30 + 48
    obs[start + 7] = 0.6
    obs[start + 8 + 7] = 0.4
    smell = antenna_smell_vec(obs)
    assert smell.shape == (SMELL_VEC_DIM,)
    assert abs(float(smell[7]) - 0.6) < 1e-6
    assert abs(float(smell[15]) - 0.4) < 1e-6
    sensory = SignalBuilder().build(obs)
    assert sensory.smell_vec.shape == (SMELL_VEC_DIM,)
    assert abs(float(sensory.summary["odor_left"]) - 0.6) < 1e-6
    assert abs(float(sensory.summary["odor_right"]) - 0.4) < 1e-6
    assert abs(float(sensory.summary["odor"]) - 0.6) < 1e-6
    assert ANTENNA_DIM == SMELL_VEC_DIM


def test_taste_cue_bypasses_familiar_joint() -> None:
    mem = InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM)
    cfg = StimulusInterruptConfig(history_backend="joint", cooldown_steps=0)
    eye = _gray(0.5)
    patch = stereo_patch_embedding(eye, eye)
    smell = _smell()
    sensors = _sensors(0.2)
    _insert_joint(mem, patch=patch, smell=smell, sensors=sensors)
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
        smell_vec=smell,
        sensors_vec=sensors,
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
    cfg = StimulusInterruptConfig(history_backend="joint", cooldown_steps=0)
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
        sensors_vec=_sensors(),
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
    pipe._interrupt_remaining = 5
    pipe._interrupt_reason = "test"
    action, info = pipe.process_motor_only(obs)
    assert info.get("stimulus_interrupt_pending") is True
    assert info.get("interrupt_remaining") == 5
    assert abs(float(action[0])) <= 0.08
    pipe.close()


def test_deep_thought_freeze_stops_motor_then_clears() -> None:
    from flybrain.env import FakeFlyEnv

    env = FakeFlyEnv()
    mem = InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM)
    cfg = StimulusInterruptConfig(
        enabled=True,
        history_backend="joint",
        cooldown_steps=0,
        deep_thought_steps=3,
        deep_thought_gain=10.0,
    )
    pipe = LiveMindPipeline(
        env,
        config=LiveMindConfig(
            backend="in_memory",
            llm_mock=True,
            enable_motor_layer=True,
            mind_interval=40,
            stimulus_interrupt=cfg,
            max_steps=12,
            console_log=False,
        ),
        memory=mem,
    )
    eye = _gray(0.55)
    patch = stereo_patch_embedding(eye, eye)
    smell = _smell(odor_l=0.4, odor_r=0.35)
    sensors = _sensors(0.33)
    joint = build_interrupt_joint_vec(patch, smell, sensors)
    pipe._interrupt_pending = True
    pipe._interrupt_remaining = 3
    pipe._interrupt_reason = "novel_vs_smell"
    pipe._interrupt_patch_vec = patch
    pipe._interrupt_smell_vec = smell
    pipe._interrupt_sensors_vec = sensors
    pipe._interrupt_joint_vec = joint
    pipe._last_mind_info = {"intent": {"future": {"chosen_name": "explore"}}, "affect": {}}
    obs, _ = env.reset()
    for i in range(3):
        pipe._step = i + 1
        action, info = pipe.process_motor_only(obs)
        assert info.get("stimulus_interrupt_pending") is True
        assert abs(float(action[0])) <= 0.08
        assert abs(float(action[1])) <= 0.08
        pipe._tick_interrupt_freeze()
    assert pipe._interrupt_pending is False
    recent = mem.recent(fly_id="fly_001", limit=1)
    assert recent
    assert float(np.linalg.norm(recent[0].joint_vec)) > 0.1
    pipe.close()


def test_deep_thought_mind_tick_runs_devil_without_motor() -> None:
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
                deep_thought_steps=8,
                deep_thought_gain=10.0,
                cooldown_steps=0,
            ),
        ),
        memory=mem,
    )
    obs, _ = env.reset()
    pipe._interrupt_pending = True
    pipe._interrupt_remaining = 8
    pipe._interrupt_reason = "novel_vs_smell"
    pipe._interrupt_meta = {"cue": "smell_novelty"}
    action, info = pipe._process_deep_thought(obs)
    assert info.get("deep_thought") is True
    assert info.get("mind_tick") is True
    assert info.get("stimulus_interrupt_pending") is True
    assert info.get("devil_critique")
    assert info.get("judgement")  # deep ticks now judge
    assert "judgement_confirm_count" in info
    assert abs(float(action[0])) <= 0.08
    assert abs(float(action[1])) <= 0.08
    # Freeze must not clear on the mind tick itself.
    assert pipe._interrupt_pending is True
    assert pipe._interrupt_remaining == 8
    pipe.close()


def test_heighten_sensory_for_deep_thought_scales_senses() -> None:
    from flybrain.thinking.stimulus_interrupt import heighten_sensory_for_deep_thought

    builder = SignalBuilder(vision_size=8)
    obs = np.zeros(197, dtype=np.float32)
    # Antenna sugar / odor non-zero so smell scales visibly.
    start = 72 + 30 + 48  # joints + contacts + foot taste
    obs[start + 0] = 0.05  # left sugar
    obs[start + 7] = 0.08  # left odor
    eye = np.full((8, 8), 0.05, dtype=np.float32)
    sensory = builder.build(obs, left_eye=eye, right_eye=eye)
    orig_odor = float(sensory.summary["odor"])
    orig_smell = float(np.max(sensory.smell_vec))
    orig_eye = float(np.max(sensory.left_eye))
    boosted = heighten_sensory_for_deep_thought(sensory, gain=10.0)
    assert abs(float(boosted.summary["odor"]) - orig_odor * 10.0) < 1e-5
    assert abs(float(np.max(boosted.smell_vec)) - orig_smell * 10.0) < 1e-5
    assert abs(float(np.max(boosted.left_eye)) - min(1.0, orig_eye * 10.0)) < 1e-5
    # Original unchanged.
    assert float(sensory.summary["odor"]) == orig_odor
    assert float(np.max(sensory.left_eye)) == orig_eye


def test_config_from_yaml_loads_stimulus_interrupt() -> None:
    from flybrain.pipeline.live_mind import load_mind_config

    raw = load_mind_config()
    cfg = LiveMindConfig.from_yaml(raw)
    assert cfg.stimulus_interrupt is not None
    assert cfg.stimulus_interrupt.history_backend == "joint"
    assert cfg.stimulus_interrupt.patch_match_min == 0.92
    assert cfg.stimulus_interrupt.smell_enabled is True
    assert cfg.stimulus_interrupt.smell_match_min == 0.92
    assert cfg.stimulus_interrupt.joint_neighbors == 8
    assert cfg.stimulus_interrupt.deep_thought_steps == 80
    assert cfg.stimulus_interrupt.deep_thought_gain == 10.0
    assert cfg.stimulus_interrupt.deep_thought_mind_interval == 4
    assert cfg.stimulus_interrupt.conditioning_hit_threshold == 3
    assert cfg.stimulus_interrupt.judgement_confirmations == 2
    assert cfg.stimulus_interrupt.passive_smell_enabled is True
    assert cfg.stimulus_interrupt.passive_smell_interval == 4
    assert cfg.stimulus_interrupt.passive_smell_change_min == 0.05
    assert cfg.stimulus_interrupt.passive_smell_refresh_ms == 500.0


def test_deep_thought_uses_faster_mind_interval() -> None:
    """While frozen, mind ticks on deep_thought_mind_interval (4), not 40."""
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
                deep_thought_steps=12,
                deep_thought_mind_interval=4,
                deep_thought_gain=10.0,
                cooldown_steps=0,
                judgement_confirmations=99,  # don't early-unlock in this interval test
            ),
            max_steps=16,
            console_log=False,
        ),
        memory=mem,
    )
    obs, _ = env.reset()
    pipe._interrupt_pending = True
    pipe._interrupt_remaining = 12
    pipe._interrupt_reason = "novel_vs_smell"
    pipe._last_mind_info = {"intent": {"future": {"chosen_name": "explore"}}}
    deep_ticks = 0
    for step in range(12):
        pipe._step = step
        think_interval = (
            pipe.stimulus_interrupt.deep_thought_mind_interval
            if pipe._interrupt_pending
            else pipe.config.mind_interval
        )
        if step % think_interval == 0:
            action, info = pipe._process_deep_thought(obs)
            deep_ticks += 1
            assert info.get("deep_thought") is True
            assert info.get("mind_tick") is True
            assert abs(float(action[0])) <= 0.08
        else:
            action, info = pipe.process_motor_only(obs)
            assert info.get("mind_tick") is False
            assert abs(float(action[0])) <= 0.08
        pipe._tick_interrupt_freeze()
    # steps 0,4,8 → 3 deep ticks in a 12-step freeze (not just 0 when interval=40)
    assert deep_ticks == 3
    assert pipe._interrupt_pending is False
    pipe.close()


def test_passive_smell_gate_samples_on_interval_only() -> None:
    gate = PassiveSmellGate(interval=4, change_min=0.05, refresh_ms=500.0)
    sampled = [
        gate.observe(_smell(odor_l=0.5), step=step, now_ms=0.0).sampled for step in range(8)
    ]
    assert sampled == [True, False, False, False, True, False, False, False]


def test_passive_smell_gate_closes_when_smell_is_stable() -> None:
    """Stable antenna within the refresh window ⇒ no STM probe."""
    gate = PassiveSmellGate(interval=4, change_min=0.05, refresh_ms=500.0)
    smell = _smell(odor_l=0.5, odor_r=0.5)
    first = gate.observe(smell, step=0, now_ms=0.0)
    assert first.active and first.changed
    # Same reading, next sample tick, well inside the 500 ms refresh window.
    stable = gate.observe(smell, step=4, now_ms=100.0)
    assert stable.sampled is True
    assert stable.changed is False
    assert stable.active is False
    assert stable.reason == ""
    # Off-tick steps do not sample and stay closed.
    assert gate.observe(smell, step=5, now_ms=120.0).active is False


def test_passive_smell_gate_opens_on_change() -> None:
    gate = PassiveSmellGate(interval=4, change_min=0.05, refresh_ms=500.0)
    gate.observe(_smell(odor_l=0.5), step=0, now_ms=0.0)
    changed = gate.observe(_smell(odor_l=0.9), step=4, now_ms=50.0)
    assert changed.changed is True
    assert changed.active is True
    assert changed.reason == "smell_change"
    assert changed.delta > 0.05


def test_passive_smell_gate_opens_on_vision_novelty_off_tick() -> None:
    gate = PassiveSmellGate(interval=4, change_min=0.05, refresh_ms=500.0)
    smell = _smell(odor_l=0.5)
    gate.observe(smell, step=0, now_ms=0.0)
    seen = gate.observe(smell, step=5, now_ms=60.0, visual_novel=True)
    assert seen.sampled is False
    assert seen.active is True
    assert seen.reason == "vision_novelty"


def test_passive_smell_gate_refresh_timer_reopens() -> None:
    gate = PassiveSmellGate(interval=4, change_min=0.05, refresh_ms=500.0)
    smell = _smell(odor_l=0.5)
    gate.observe(smell, step=0, now_ms=0.0)
    assert gate.observe(smell, step=4, now_ms=200.0).active is False
    due = gate.observe(smell, step=8, now_ms=500.0)
    assert due.active is True
    assert due.reason == "timer"


def test_passive_smell_gate_accumulates_slow_drift() -> None:
    """Baseline updates only on change, so sub-threshold drift still trips."""
    gate = PassiveSmellGate(interval=4, change_min=0.05, refresh_ms=1e9)
    gate.observe(_smell(odor_l=0.50), step=0, now_ms=0.0)
    assert gate.observe(_smell(odor_l=0.52), step=4, now_ms=10.0).changed is False
    assert gate.observe(_smell(odor_l=0.54), step=8, now_ms=20.0).changed is False
    assert gate.observe(_smell(odor_l=0.56), step=12, now_ms=30.0).changed is True


def test_passive_smell_gate_ignores_empty_antenna() -> None:
    gate = PassiveSmellGate(interval=4, change_min=0.05, refresh_ms=1e9)
    first = gate.observe(np.zeros(SMELL_VEC_DIM, dtype=np.float32), step=0, now_ms=0.0)
    assert first.changed is False


def test_closed_passive_gate_skips_all_stm_queries() -> None:
    """A no-cue step with the gate closed must not touch STM at all."""

    class _CountingMemory(InMemoryShortTermMemory):
        def __init__(self) -> None:
            super().__init__(dim=TOTAL_EMBEDDING_DIM)
            self.queries = 0

        def similar_by(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            self.queries += 1
            return super().similar_by(*args, **kwargs)

    mem = _CountingMemory()
    cfg = StimulusInterruptConfig(history_backend="joint", cooldown_steps=0)
    gate = PassiveSmellGate.from_config(cfg)
    smell = _smell(odor_l=0.5, odor_r=0.5)
    eye = _gray(0.4)
    kwargs = dict(
        memory=mem,
        signals={},
        affect={"survival_risk": 0.1},
        fly_id="fly_001",
        cooldown_until_step=0,
        config=cfg,
        visual_novelty=0.0,
        new_stimulus=False,
        left_eye=eye,
        right_eye=eye,
        smell_vec=smell,
        sensors_vec=_sensors(0.3),
        passive_smell=gate,
    )
    first = evaluate_stimulus_interrupt(step=0, now_ms=0.0, **kwargs)
    assert first.smell_probe_ran is True
    probes_after_first = mem.queries
    assert probes_after_first > 0

    closed = evaluate_stimulus_interrupt(step=4, now_ms=100.0, **kwargs)
    assert closed.smell_probe_ran is False
    assert closed.passive_smell_reason == ""
    assert closed.candidate is False
    assert closed.reason == "no_cue"
    assert mem.queries == probes_after_first
    mem.close()


def test_pipeline_owns_passive_smell_gate() -> None:
    from flybrain.env import FakeFlyEnv

    env = FakeFlyEnv()
    mem = InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM)
    pipe = LiveMindPipeline(
        env,
        config=LiveMindConfig(
            backend="in_memory",
            llm_mock=True,
            stimulus_interrupt=StimulusInterruptConfig(passive_smell_interval=4),
            console_log=False,
        ),
        memory=mem,
    )
    assert pipe._passive_smell.interval == 4
    pipe._passive_smell.observe(_smell(odor_l=0.5), step=0, now_ms=0.0)
    assert pipe._passive_smell._last is not None
    pipe._passive_smell.reset()
    assert pipe._passive_smell._last is None
    pipe.close()


def test_patch_grid_history_cache_api_compat() -> None:
    cache = PatchGridHistoryCache.from_config(StimulusInterruptConfig())
    cache.reset()


def test_stop_goal_is_zero_locomotion() -> None:
    assert STOP_GOAL.shape == (5,)
    assert float(STOP_GOAL[0]) == 0.0
    assert float(STOP_GOAL[1]) == 0.0


def test_deep_thought_conditioning_and_early_unlock_on_confirming_accept() -> None:
    """Two confirming accept judgements clear freeze early and resume motor."""
    from unittest.mock import patch

    from flybrain.env import FakeFlyEnv
    from flybrain.thinking.judgement import DevilProposal, JudgementResult
    from flybrain.thinking.personality import PersonalityTraits

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
                deep_thought_steps=80,
                deep_thought_mind_interval=4,
                deep_thought_gain=10.0,
                cooldown_steps=0,
                conditioning_hit_threshold=2,
                judgement_confirmations=2,
            ),
            console_log=False,
        ),
        memory=mem,
    )
    obs, _ = env.reset()
    pipe._interrupt_pending = True
    pipe._interrupt_remaining = 80
    pipe._interrupt_reason = "novel_vs_smell"
    pipe._interrupt_meta = {"cue": "smell_novelty"}
    pipe.conditioning.begin_freeze()

    odor_prop = DevilProposal(
        risk="odor",
        forward_scale=0.5,
        turn_delta=0.25,
        confidence=0.9,
        reason="odor plume",
        ran=True,
        source="test",
    )

    def _force_accept(proposal, **kwargs):
        return JudgementResult(
            decision="accept",
            reason="test_force_accept",
            blend_weight=0.4,
            proposal=proposal,
            traits_snapshot=PersonalityTraits().as_dict(),
            scores={"accept": 1.0},
        )

    pipe.judgement.evaluate = _force_accept  # type: ignore[method-assign]

    with patch(
        "flybrain.thinking.service.propose_critique",
        return_value=odor_prop,
    ):
        action1, info1 = pipe._process_deep_thought(obs)
        assert info1.get("judgement", {}).get("decision") == "accept"
        assert info1.get("judgement_confirm_count") == 1
        assert pipe._interrupt_pending is True
        assert abs(float(action1[0])) <= 0.08
        assert "odor" in (info1.get("conditioning") or {}).get("memory_block", "")

        action2, info2 = pipe._process_deep_thought(obs)

    assert info2.get("stimulus_interrupt_early_unlock") is True
    assert pipe._interrupt_pending is False
    assert info2.get("stimulus_interrupt") is False
    assert abs(float(action2[0])) > 0.05 or abs(float(action2[1])) > 0.05
    assert "odor=" in str(info2.get("refined_thought") or "")
    assert "Judged:" in str(info2.get("refined_thought") or "")
    pipe.close()


def test_deep_thought_acknowledge_suppresses_conditioning_memory() -> None:
    from unittest.mock import patch

    from flybrain.env import FakeFlyEnv
    from flybrain.thinking.judgement import DevilProposal, JudgementResult
    from flybrain.thinking.personality import PersonalityTraits

    env = FakeFlyEnv()
    mem = InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM)
    pipe = LiveMindPipeline(
        env,
        config=LiveMindConfig(
            backend="in_memory",
            llm_mock=True,
            enable_motor_layer=True,
            stimulus_interrupt=StimulusInterruptConfig(
                enabled=True,
                deep_thought_steps=80,
                judgement_confirmations=99,
                conditioning_hit_threshold=1,
                cooldown_steps=0,
            ),
            console_log=False,
        ),
        memory=mem,
    )
    obs, _ = env.reset()
    pipe._interrupt_pending = True
    pipe._interrupt_remaining = 80
    pipe._interrupt_reason = "novel_vs_wall"
    pipe._interrupt_meta = {"cue": "visual_novelty"}
    pipe.conditioning.begin_freeze()

    wall_prop = DevilProposal(
        risk="wall",
        forward_scale=0.4,
        turn_delta=0.3,
        confidence=0.8,
        reason="wall ahead",
        ran=True,
        source="test",
    )

    def _force_ack(proposal, **kwargs):
        return JudgementResult(
            decision="acknowledge",
            reason="test_force_ack",
            blend_weight=0.0,
            proposal=proposal,
            traits_snapshot=PersonalityTraits().as_dict(),
            scores={"acknowledge": 1.0},
        )

    pipe.judgement.evaluate = _force_ack  # type: ignore[method-assign]
    with patch("flybrain.thinking.service.propose_critique", return_value=wall_prop):
        _, info1 = pipe._process_deep_thought(obs)
        assert info1.get("judgement", {}).get("decision") == "acknowledge"
        assert pipe.conditioning.is_acknowledged("wall")
        assert "wall" not in pipe.conditioning.memory_block()

        _, info2 = pipe._process_deep_thought(obs)

    block = (info2.get("conditioning") or {}).get("memory_block", "")
    assert "wall" not in block
    assert pipe._interrupt_pending is True
    pipe.close()


def test_deep_thought_mismatch_does_not_early_unlock() -> None:
    from unittest.mock import patch

    from flybrain.env import FakeFlyEnv
    from flybrain.thinking.judgement import DevilProposal, JudgementResult
    from flybrain.thinking.personality import PersonalityTraits

    env = FakeFlyEnv()
    mem = InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM)
    pipe = LiveMindPipeline(
        env,
        config=LiveMindConfig(
            backend="in_memory",
            llm_mock=True,
            enable_motor_layer=True,
            stimulus_interrupt=StimulusInterruptConfig(
                enabled=True,
                deep_thought_steps=80,
                judgement_confirmations=2,
                cooldown_steps=0,
            ),
            console_log=False,
        ),
        memory=mem,
    )
    obs, _ = env.reset()
    pipe._interrupt_pending = True
    pipe._interrupt_remaining = 80
    pipe._interrupt_meta = {"cue": "smell_novelty"}
    pipe.conditioning.begin_freeze()

    odor_prop = DevilProposal(
        risk="odor",
        forward_scale=0.5,
        turn_delta=0.2,
        confidence=0.9,
        reason="test",
        ran=True,
        source="test",
    )
    decisions = iter(["accept", "acknowledge"])

    def _alternating(proposal, **kwargs):
        dec = next(decisions)
        return JudgementResult(
            decision=dec,  # type: ignore[arg-type]
            reason="test_alt",
            blend_weight=0.4 if dec == "accept" else 0.0,
            proposal=proposal,
            traits_snapshot=PersonalityTraits().as_dict(),
            scores={dec: 1.0},
        )

    pipe.judgement.evaluate = _alternating  # type: ignore[method-assign]
    with patch("flybrain.thinking.service.propose_critique", return_value=odor_prop):
        for _ in range(2):
            _, info = pipe._process_deep_thought(obs)
    assert info.get("stimulus_interrupt_early_unlock") is not True
    assert pipe._interrupt_pending is True
    assert pipe.conditioning.confirmation_count() == 1
    pipe.close()
