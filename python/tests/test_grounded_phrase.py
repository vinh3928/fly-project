"""Grounded phrase store + image/sound match + LLM label parsing."""

from __future__ import annotations

import time

import numpy as np
import pytest

from flybrain.env import FakeFlyEnv
from flybrain.memory.grounded_phrase import (
    GroundedPhrase,
    GroundedPhraseConfig,
    GroundedSentence,
    InMemoryGroundedPhraseStore,
    cosine_sim,
    match_grounded_phrase,
    match_grounded_sentence,
    match_top_grounded_phrases,
)
from flybrain.memory.speech_cortex import SpeechCortexConfig, looks_like_word
from flybrain.pipeline.live_mind import LiveMindConfig, LiveMindPipeline
from flybrain.thinking.phrase_labeler import (
    PhraseGroundingService,
    heuristic_phrase_label,
    parse_phrase_label,
)
from flybrain.thinking.sentence_composer import (
    SentenceComposeService,
    heuristic_sentence,
    parse_sentence_label,
)


def test_cosine_and_match_prefers_similar_patch_and_audio() -> None:
    store = InMemoryGroundedPhraseStore()
    a = store.upsert_phrase(
        GroundedPhrase(
            phrase_id="p1",
            phrase="sugar pad",
            fly_id="fly_a",
            patch_vec=[1.0, 0.0, 0.0],
            spectrogram=[[0.9, 0.1, 0.0]],
            confidence=0.8,
            score=0.8,
        )
    )
    store.upsert_phrase(
        GroundedPhrase(
            phrase_id="p2",
            phrase="wind",
            fly_id="fly_a",
            patch_vec=[0.0, 1.0, 0.0],
            spectrogram=[[0.0, 0.9, 0.1]],
            confidence=0.7,
            score=0.7,
        )
    )
    cfg = GroundedPhraseConfig(
        enabled=True,
        match_patch_min=0.5,
        match_audio_min=0.5,
        match_combined_min=0.5,
    )
    hit = match_grounded_phrase(
        store,
        fly_id="fly_a",
        patch_vec=[0.95, 0.05, 0.0],
        spectrogram=[[0.85, 0.15, 0.0]],
        config=cfg,
    )
    assert hit is not None
    assert hit.phrase.phrase_id == a.phrase_id
    assert cosine_sim([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)


def test_heuristic_and_parse_phrase_label() -> None:
    h = heuristic_phrase_label(alphabet="ma", rms=0.5, voiced=0.2, cue="")
    assert h.phrase
    parsed = parse_phrase_label(
        '{"phrase": "bee buzz", "confidence": 0.77}',
        fallback=h,
    )
    assert parsed.phrase == "bee buzz"
    assert parsed.confidence == pytest.approx(0.77)


def test_looks_like_multiword_phrase() -> None:
    assert looks_like_word("sugar pad")
    assert looks_like_word("bee buzz")
    assert not looks_like_word("mmm mmm")


def test_phrase_service_mock_stores_label() -> None:
    store = InMemoryGroundedPhraseStore()
    cfg = GroundedPhraseConfig(enabled=True, mock=True, cooldown_steps=0, min_confidence=0.2)
    svc = PhraseGroundingService(cfg, store, client=None)
    draft = GroundedPhrase(
        phrase_id="d1",
        phrase="",
        alphabet="ma",
        fly_id="fly_a",
        patch_vec=[0.2, 0.3],
        spectrogram=[[0.4, 0.5]],
        rms=0.4,
        voiced=0.2,
    )
    assert svc.maybe_submit(
        step=10,
        draft=draft,
        left_eye=np.zeros((8, 8), dtype=np.float32),
        right_eye=np.zeros((8, 8), dtype=np.float32),
        bands=[0.4, 0.5],
        cue="env_audio",
    )
    stored = None
    for _ in range(50):
        stored = svc.poll()
        if stored is not None:
            break
        time.sleep(0.01)
    assert stored is not None
    assert stored.phrase
    assert store.get_phrase(stored.phrase_id) is not None
    svc.close()


def test_live_mind_phrase_grounding_on_hear() -> None:
    env = FakeFlyEnv()
    cfg = LiveMindConfig(
        backend="in_memory",
        max_steps=3,
        llm_mock=True,
        fly_id="fly_phrase",
        enable_motor_layer=True,
        console_log=False,
        speech_cortex=SpeechCortexConfig(
            enabled=True,
            backend="in_memory",
            hear_rms_min=0.05,
            hear_cooldown_steps=0,
            speech_interval=1,
        ),
        phrase_grounding=GroundedPhraseConfig(
            enabled=True,
            backend="in_memory",
            mock=True,
            cooldown_steps=0,
            min_confidence=0.2,
            match_combined_min=0.99,  # force miss → label path
            reinforce_speech_words=True,
            word_weight_delta=0.1,
        ),
    )
    pipeline = LiveMindPipeline(env, config=cfg)
    try:
        assert pipeline.phrase_grounding is not None
        pipeline._cache_env_audio(
            {
                "audio_bands": list(np.linspace(0.2, 0.9, 16)),
                "audio_rms": 0.5,
                "audio_voiced": 0.3,
            }
        )
        pipeline._step = 4
        assert pipeline._maybe_hear_utterance({"cue": "env_audio"})
        stored = None
        for _ in range(80):
            pipeline._poll_phrase_grounding()
            if pipeline.phrase_grounding.last_stored is not None:
                stored = pipeline.phrase_grounding.last_stored
                break
            time.sleep(0.01)
        assert stored is not None and stored.phrase
        words = pipeline.speech_cortex.list_words(fly_id="fly_phrase")  # type: ignore[union-attr]
        assert words, "expected speech Word reinforce from grounded phrase"
    finally:
        pipeline.close()


def test_heuristic_and_parse_sentence() -> None:
    h = heuristic_sentence(["sugar pad", "approach"], {"hunger": 0.7})
    assert "sugar pad" in h.sentence.lower()
    assert "want" in h.sentence.lower()
    parsed = parse_sentence_label(
        '{"sentence": "I hear wind and bee.", "confidence": 0.8}',
        fallback=h,
    )
    assert parsed.sentence == "I hear wind and bee."
    assert parsed.confidence == pytest.approx(0.8)


def test_match_top_phrases_and_sentence_recall() -> None:
    store = InMemoryGroundedPhraseStore()
    patch = [1.0, 0.0, 0.0]
    audio = [[0.9, 0.1, 0.0]]
    p1 = store.upsert_phrase(
        GroundedPhrase(
            phrase_id="p1",
            phrase="sugar pad",
            fly_id="fly_s",
            patch_vec=patch,
            spectrogram=audio,
            confidence=0.8,
            score=0.8,
        )
    )
    p2 = store.upsert_phrase(
        GroundedPhrase(
            phrase_id="p2",
            phrase="approach",
            fly_id="fly_s",
            patch_vec=[0.95, 0.05, 0.0],
            spectrogram=[[0.85, 0.15, 0.0]],
            confidence=0.7,
            score=0.7,
        )
    )
    cfg = GroundedPhraseConfig(
        enabled=True,
        sentence_compose=True,
        sentence_match_combined_min=0.5,
        sentence_top_k=4,
        match_patch_min=0.5,
        match_audio_min=0.5,
        match_combined_min=0.5,
    )
    top = match_top_grounded_phrases(
        store,
        fly_id="fly_s",
        patch_vec=patch,
        spectrogram=audio,
        config=cfg,
    )
    assert len(top) >= 2
    assert {m.phrase.phrase_id for m in top} >= {p1.phrase_id, p2.phrase_id}

    sent = store.upsert_sentence(
        GroundedSentence(
            sentence_id="s1",
            sentence="I want the sugar pad.",
            phrases=["sugar pad", "approach"],
            phrase_ids=[p1.phrase_id, p2.phrase_id],
            fly_id="fly_s",
            drive={"hunger": 0.7},
            patch_vec=patch,
            spectrogram=audio,
            confidence=0.6,
            score=0.6,
        )
    )
    hit = match_grounded_sentence(
        store,
        fly_id="fly_s",
        patch_vec=patch,
        spectrogram=audio,
        config=cfg,
    )
    assert hit is not None
    assert hit.sentence.sentence_id == sent.sentence_id


def test_sentence_compose_service_mock_stores() -> None:
    store = InMemoryGroundedPhraseStore()
    cfg = GroundedPhraseConfig(
        enabled=True,
        mock=True,
        sentence_compose=True,
        sentence_min_phrases=2,
        sentence_cooldown_steps=0,
        sentence_min_confidence=0.2,
    )
    svc = SentenceComposeService(cfg, store, client=None)
    draft = GroundedSentence(
        sentence_id="d1",
        sentence="",
        phrases=["sugar pad", "approach"],
        phrase_ids=["p1", "p2"],
        fly_id="fly_s",
        drive={"hunger": 0.8},
        patch_vec=[0.2, 0.3],
        spectrogram=[[0.4, 0.5]],
    )
    assert svc.maybe_submit(step=1, draft=draft)
    stored = None
    for _ in range(50):
        stored = svc.poll()
        if stored is not None:
            break
        time.sleep(0.01)
    assert stored is not None
    assert "sugar" in stored.sentence.lower()
    assert store.get_sentence(stored.sentence_id) is not None
    svc.close()


def test_live_mind_sentence_compose_on_phrase_matches() -> None:
    env = FakeFlyEnv()
    cfg = LiveMindConfig(
        backend="in_memory",
        max_steps=3,
        llm_mock=True,
        fly_id="fly_sent",
        enable_motor_layer=True,
        console_log=False,
        speech_cortex=SpeechCortexConfig(
            enabled=True,
            backend="in_memory",
            hear_rms_min=0.05,
            hear_cooldown_steps=0,
            speech_interval=1,
        ),
        phrase_grounding=GroundedPhraseConfig(
            enabled=True,
            backend="in_memory",
            mock=True,
            cooldown_steps=0,
            min_confidence=0.2,
            match_patch_min=0.5,
            match_audio_min=0.5,
            match_combined_min=0.5,
            sentence_compose=True,
            sentence_min_phrases=2,
            sentence_top_k=4,
            sentence_match_combined_min=0.4,
            sentence_cooldown_steps=0,
            sentence_min_confidence=0.2,
            reinforce_speech_words=True,
        ),
    )
    pipeline = LiveMindPipeline(env, config=cfg)
    try:
        assert pipeline.sentence_compose is not None
        assert pipeline.phrase_store is not None
        patch = [1.0, 0.0, 0.0]
        audio = [[0.9, 0.1, 0.0]]
        pipeline.phrase_store.upsert_phrase(
            GroundedPhrase(
                phrase_id="ps1",
                phrase="sugar pad",
                fly_id="fly_sent",
                patch_vec=patch,
                spectrogram=audio,
                confidence=0.8,
                score=0.8,
            )
        )
        pipeline.phrase_store.upsert_phrase(
            GroundedPhrase(
                phrase_id="ps2",
                phrase="approach",
                fly_id="fly_sent",
                patch_vec=[0.95, 0.05, 0.0],
                spectrogram=[[0.85, 0.15, 0.0]],
                confidence=0.7,
                score=0.7,
            )
        )
        info: dict = {"hunger": 0.75, "visual_novelty": 0.2}
        # Seed eye embedding so match uses similar patch.
        pipeline._speech_eye_image = lambda: type(  # type: ignore[method-assign]
            "Eye", (), {"patch_vec": patch, "image_path": None}
        )()
        pipeline._cache_env_audio(
            {
                "audio_bands": list(audio[0]),
                "audio_rms": 0.5,
                "audio_voiced": 0.3,
            }
        )
        pipeline._step = 5
        pipeline._ground_or_recall_phrase(
            alphabet="ma",
            spectrogram=audio,
            rms=0.5,
            voiced=0.3,
            utterance_id="u1",
            place_id=None,
            cue="env_audio",
            source="hear",
            info=info,
        )
        stored = None
        for _ in range(80):
            pipeline._poll_phrase_grounding(info)
            if pipeline.sentence_compose.last_stored is not None:
                stored = pipeline.sentence_compose.last_stored
                break
            time.sleep(0.01)
        assert stored is not None and stored.sentence
        assert "sugar" in stored.sentence.lower() or "approach" in stored.sentence.lower()
        assert info.get("sentence_composed") or info.get("sentence_recall")
    finally:
        pipeline.close()
