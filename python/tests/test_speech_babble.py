"""Speech babble sampler + gate + executor."""

from __future__ import annotations

import numpy as np

from flybrain.speech.articulatory import ARTICULATORY_DIM
from flybrain.speech.babble import (
    PRIMITIVE_AH,
    PRIMITIVE_MM,
    SpeechBabbleConfig,
    babble_gate,
    expand_controls,
    sample_babble_utterance,
)
from flybrain.speech.executor import SpeechExecutor


def test_sample_babble_utterance_shape() -> None:
    rng = np.random.default_rng(0)
    cfg = SpeechBabbleConfig(min_steps=2, max_steps=5, duration_range=(4, 8))
    name, steps, tags = sample_babble_utterance(rng, config=cfg)
    assert name.startswith("candidate_utter_")
    assert "babble" in tags and "speech" in tags
    assert 2 <= len(steps) <= 5
    for step in steps:
        assert step.control.shape == (ARTICULATORY_DIM,)
        assert 4 <= int(step.duration_steps) <= 8
    expanded = expand_controls(steps)
    assert len(expanded) == sum(int(s.duration_steps) for s in steps)
    assert expanded[0].shape == (ARTICULATORY_DIM,)


def test_sample_babble_biases_toward_weighted_word() -> None:
    cfg = SpeechBabbleConfig(
        min_steps=2,
        max_steps=4,
        duration_range=(2, 2),
        word_bias_explore=0.0,
    )
    # Heavy "ma" should dominate when explore is off.
    hits = 0
    trials = 40
    for seed in range(trials):
        rng = np.random.default_rng(seed)
        _name, steps, tags = sample_babble_utterance(
            rng,
            config=cfg,
            word_weights=[("ma", 0.9), ("ss", 0.05)],
        )
        assert "word_bias" in tags
        assert any(str(t).startswith("word:") for t in tags)
        prims = [s.primitive for s in steps]
        if prims[:2] == [PRIMITIVE_MM, PRIMITIVE_AH]:
            hits += 1
    assert hits >= 30, f"expected weight bias toward ma, got hits={hits}/{trials}"


def test_sample_babble_explore_can_ignore_weights() -> None:
    cfg = SpeechBabbleConfig(
        min_steps=2,
        max_steps=3,
        duration_range=(2, 2),
        word_bias_explore=1.0,
    )
    rng = np.random.default_rng(1)
    _name, _steps, tags = sample_babble_utterance(
        rng,
        config=cfg,
        word_weights=[("ma", 1.0)],
    )
    assert "word_bias" not in tags


def test_speech_babble_gate() -> None:
    cfg = SpeechBabbleConfig(
        enabled=True,
        min_novelty=0.25,
        max_survival_risk=0.45,
        max_hunger=0.55,
        cooldown_steps=160,
    )
    assert babble_gate(
        config=cfg,
        step=400,
        last_babble_step=0,
        executor_active=False,
        novelty=0.5,
        survival_risk=0.1,
        hunger=0.2,
    ).ok
    assert not babble_gate(
        config=cfg,
        step=10,
        last_babble_step=0,
        executor_active=False,
        novelty=0.5,
        survival_risk=0.1,
        hunger=0.2,
    ).ok
    assert not babble_gate(
        config=cfg,
        step=400,
        last_babble_step=0,
        executor_active=True,
        novelty=0.5,
        survival_risk=0.1,
        hunger=0.2,
    ).ok


def test_speech_executor_playback() -> None:
    rng = np.random.default_rng(3)
    _, steps, _ = sample_babble_utterance(
        rng, config=SpeechBabbleConfig(min_steps=2, max_steps=2, duration_range=(2, 2))
    )
    ex = SpeechExecutor()
    ex.start(steps, utterance_id="u1", name="candidate_utter_x")
    frames = 0
    while ex.active:
        frame = ex.tick()
        assert frame.active or frame.finished
        assert frame.control.shape == (ARTICULATORY_DIM,)
        frames += 1
        if frames > 20:
            break
    assert frames >= 2
    stats = ex.episode_stats()
    assert stats["utterance_id"] == "u1"
