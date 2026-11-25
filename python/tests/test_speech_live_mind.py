"""Fake-env speech cortex: hear / babble upsert with letters + image meta."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flybrain.env import FakeFlyEnv
from flybrain.memory.speech_cortex import SpeechCortexConfig
from flybrain.pipeline.live_mind import LiveMindConfig, LiveMindPipeline
from flybrain.speech.babble import SpeechBabbleConfig


def test_fake_env_audio_in_step_info() -> None:
    env = FakeFlyEnv()
    env.reset(seed=0)
    info = {}
    for _ in range(6):
        _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert "audio_bands" in info
    assert len(info["audio_bands"]) == 16
    assert float(info["audio_rms"]) > 0.0
    env.set_mouth_articulatory([0.8, 0.7, 0.4, 0.4, 0.9, 0.0, 0.8])
    assert float(env.mouth_articulatory[0]) == pytest.approx(0.8, abs=1e-5)


def test_live_mind_hear_and_babble_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    env = FakeFlyEnv()
    log_path = tmp_path / "speech_live.log"
    cfg = LiveMindConfig(
        backend="in_memory",
        max_steps=5,
        llm_mock=True,
        fly_id="fly_speech",
        enable_motor_layer=True,
        mind_interval=8,
        console_log=True,
        log_path=str(log_path),
        speech_cortex=SpeechCortexConfig(
            enabled=True,
            backend="in_memory",
            speech_interval=4,
            hear_rms_min=0.05,
            hear_cooldown_steps=0,
            babble=SpeechBabbleConfig(
                enabled=True,
                min_novelty=0.0,
                max_survival_risk=1.0,
                max_hunger=1.0,
                cooldown_steps=0,
                min_steps=2,
                max_steps=3,
                duration_range=(2, 2),
            ),
        ),
    )
    pipeline = LiveMindPipeline(env, config=cfg)
    try:
        assert pipeline.speech_cortex is not None
        # Hear path: inject audio features as if Unity/FakeFlyEnv just stepped.
        bands = np.clip(np.linspace(0.2, 0.9, 16), 0.0, 1.0).astype(np.float32)
        pipeline._cache_env_audio(
            {"audio_bands": bands.tolist(), "audio_rms": 0.4, "audio_voiced": 1.0}
        )
        pipeline._step = 10
        assert pipeline._maybe_hear_utterance({"cue": "env_audio"})
        hears = [
            r
            for r in pipeline.speech_cortex.list_utterances(fly_id="fly_speech")
            if r.source == "hear"
        ]
        assert len(hears) == 1
        assert hears[0].alphabet
        assert hears[0].image is not None
        assert len(hears[0].spectrogram[0]) == 16

        # Babble path: low-load gate with forced signals.
        pipeline._step = 20
        assert pipeline._maybe_babble_speech(
            {"visual_novelty": 0.9, "hunger": 0.1, "tendency_curiosity": 0.8},
            {"survival_risk": 0.05},
        )
        assert pipeline.speech_executor.active
        pipeline._tick_speech_executor({})
        babbles = [
            r
            for r in pipeline.speech_cortex.list_utterances(fly_id="fly_speech")
            if r.source == "babble"
        ]
        assert len(babbles) == 1
        assert babbles[0].alphabet
        assert babbles[0].articulatory
        assert babbles[0].image is not None

        out = capsys.readouterr()
        assert "speech_babble step=20" in out.out
        assert "speech_babble step=20" in out.err
        assert "speech_hear step=10" in out.out
        log_text = log_path.read_text(encoding="utf-8")
        assert "speech_babble step=20" in log_text
        assert "name=" in log_text
        assert "speech_hear step=10" in log_text
        assert pipeline._last_speech_hear_alphabet
        assert pipeline._last_speech_babble_alphabet
    finally:
        pipeline.close()


def test_speech_channel_send_status_tracks_alphabets() -> None:
    from flybrain.debug.speech_channel import SpeechChannel

    ch = SpeechChannel()
    ch.send_status(hear="ABC", babble="XY")
    assert ch.last_hear_alphabet == "ABC"
    assert ch.last_babble_alphabet == "XY"
    ch.send_status(hear="Z")
    assert ch.last_hear_alphabet == "Z"
    assert ch.last_babble_alphabet == "XY"


def test_live_mind_babble_reinforces_wordlike_alphabet() -> None:
    env = FakeFlyEnv()
    cfg = LiveMindConfig(
        backend="in_memory",
        max_steps=3,
        llm_mock=True,
        fly_id="fly_word",
        enable_motor_layer=True,
        console_log=False,
        speech_cortex=SpeechCortexConfig(
            enabled=True,
            backend="in_memory",
            word_min_len=2,
            word_weight_delta=0.15,
            babble=SpeechBabbleConfig(
                enabled=True,
                min_novelty=0.0,
                max_survival_risk=1.0,
                max_hunger=1.0,
                cooldown_steps=0,
                min_steps=2,
                max_steps=3,
                duration_range=(2, 2),
            ),
        ),
    )
    pipeline = LiveMindPipeline(env, config=cfg)
    try:
        from flybrain.memory.speech_cortex import UtteranceProgram, reinforce_wordlike_babble

        # Force a word-like alphabet the same way live_mind does after babble upsert.
        program = UtteranceProgram(
            utterance_id="forced",
            alphabet="ba",
            status="candidate",
            source="babble",
            fly_id="fly_word",
        )
        assert pipeline.speech_cortex is not None
        pipeline.speech_cortex.upsert_utterance(program)
        stored, word = reinforce_wordlike_babble(
            pipeline.speech_cortex,
            program,
            min_len=cfg.speech_cortex.word_min_len,
            delta=cfg.speech_cortex.word_weight_delta,
        )
        assert word is not None and word.word == "ba"
        assert word.weight == pytest.approx(0.15)
        assert stored.score == pytest.approx(0.15)
        got = pipeline.speech_cortex.get_word(fly_id="fly_word", word="ba")
        assert got is not None and got.hits == 1
    finally:
        pipeline.close()


def test_live_mind_push_speech_status_to_env_channel() -> None:
    env = FakeFlyEnv()

    class _Ch:
        def __init__(self) -> None:
            self.hear = ""
            self.babble = ""

        def send_status(self, hear=None, babble=None) -> None:
            if hear is not None:
                self.hear = str(hear)
            if babble is not None:
                self.babble = str(babble)

    ch = _Ch()
    env.speech_channel = ch  # type: ignore[attr-defined]
    cfg = LiveMindConfig(
        backend="in_memory",
        max_steps=2,
        llm_mock=True,
        fly_id="fly_speech_status",
        enable_motor_layer=True,
        speech_cortex=SpeechCortexConfig(enabled=False, backend="in_memory"),
    )
    pipeline = LiveMindPipeline(env, config=cfg)
    try:
        pipeline._push_speech_status(hear="HI", babble="BY")
        assert ch.hear == "HI"
        assert ch.babble == "BY"
        pipeline._push_speech_status(babble="ZZ")
        assert ch.hear == "HI"
        assert ch.babble == "ZZ"
    finally:
        pipeline.close()


def test_speech_parallel_tick_every_four_steps() -> None:
    env = FakeFlyEnv()
    cfg = LiveMindConfig(
        backend="in_memory",
        max_steps=5,
        llm_mock=True,
        fly_id="fly_speech_iv",
        enable_motor_layer=True,
        speech_cortex=SpeechCortexConfig(
            enabled=True,
            backend="in_memory",
            speech_interval=4,
            hear_rms_min=0.05,
            hear_cooldown_steps=0,
            babble=SpeechBabbleConfig(enabled=False),
        ),
    )
    pipeline = LiveMindPipeline(env, config=cfg)
    try:
        bands = np.clip(np.linspace(0.3, 0.95, 16), 0.0, 1.0).astype(np.float32)
        pipeline._cache_env_audio(
            {"audio_bands": bands.tolist(), "audio_rms": 0.5, "audio_voiced": 1.0}
        )
        # Off-cadence: no hear decision.
        pipeline._step = 1
        info_off: dict = {}
        pipeline._speech_parallel_tick(info_off)
        assert not info_off.get("speech_tick")
        assert not info_off.get("speech_hear")
        assert pipeline.speech_cortex.list_utterances(fly_id="fly_speech_iv") == []

        # On-cadence (step % 4 == 0): hear fires.
        pipeline._step = 8
        info_on: dict = {"visual_novelty": 0.5, "hunger": 0.1}
        pipeline._speech_parallel_tick(info_on)
        assert info_on.get("speech_tick") is True
        assert info_on.get("speech_hear") is True
        assert info_on.get("speech_alphabet")
        rows = pipeline.speech_cortex.list_utterances(fly_id="fly_speech_iv")
        assert len(rows) == 1 and rows[0].source == "hear"
    finally:
        pipeline.close()
