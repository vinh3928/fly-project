"""Speech cortex store: in-memory upsert + alphabet + image bind."""

from __future__ import annotations

import numpy as np
import pytest

from flybrain.memory.speech_cortex import (
    InMemorySpeechCortex,
    SpeechContext,
    SpeechCortexConfig,
    SpeechImage,
    UtteranceProgram,
    build_babble_utterance,
    build_hear_utterance,
    looks_like_word,
    open_speech_cortex,
    reinforce_wordlike_babble,
)
from flybrain.speech.articulatory import clip_articulatory, clip_audio_bands


def test_speech_cortex_config_from_dict() -> None:
    cfg = SpeechCortexConfig.from_dict(
        {
            "enabled": True,
            "backend": "neo4j",
            "uri": "bolt://127.0.0.1:7689",
            "hear_rms_min": 0.1,
            "speech_interval": 4,
            "babble": {"min_steps": 2, "cooldown_steps": 50},
        }
    )
    assert cfg.enabled is True
    assert cfg.uri.endswith(":7689")
    assert cfg.hear_rms_min == 0.1
    assert cfg.speech_interval == 4
    assert cfg.babble.min_steps == 2
    assert cfg.babble.cooldown_steps == 50


def test_open_speech_cortex_in_memory() -> None:
    store = open_speech_cortex(backend="in_memory")
    assert isinstance(store, InMemorySpeechCortex)
    store.close()


def test_in_memory_babble_upsert_has_alphabet_and_image() -> None:
    store = InMemorySpeechCortex()
    ctrl = clip_articulatory([0.9, 0.8, 0.4, 0.35, 0.95, 0.0, 0.9])
    img = SpeechImage(
        image_id="img-1",
        path_or_b64_thumb="",
        place_id="place_a",
        step=12,
        patch_vec=[0.1, 0.2, 0.3],
    )
    program = build_babble_utterance(
        fly_id="fly_a",
        articulatory=[ctrl, ctrl],
        image=img,
        context=SpeechContext(cue="speech_babble", place_id="place_a", step=12),
    )
    stored = store.upsert_utterance(program)
    assert stored.utterance_id
    assert stored.source == "babble"
    assert stored.status == "candidate"
    assert stored.alphabet
    assert stored.image is not None
    assert stored.image.place_id == "place_a"
    assert stored.image.patch_vec[:3] == [0.1, 0.2, 0.3]
    listed = store.list_utterances(fly_id="fly_a", status="candidate")
    assert len(listed) == 1
    assert listed[0].alphabet == stored.alphabet


def test_hear_utterance_spectral_letters() -> None:
    store = InMemorySpeechCortex()
    bands = clip_audio_bands(np.linspace(0.2, 0.95, 16))
    program = build_hear_utterance(
        fly_id="fly_a",
        spectrogram=[bands],
        rms=0.5,
        voiced=1.0,
        image=SpeechImage(image_id="img-h", step=3, patch_vec=[1.0]),
        context=SpeechContext(cue="env_audio", step=3),
    )
    stored = store.upsert_utterance(program)
    assert stored.source == "hear"
    assert stored.alphabet
    assert stored.image is not None
    assert len(stored.spectrogram) == 1
    assert len(stored.spectrogram[0]) == 16


def test_promote_and_retire() -> None:
    store = InMemorySpeechCortex()
    ctrl = clip_articulatory([0.85, 0.7, 0.4, 0.35, 0.9, 0.0, 0.85])
    row = store.upsert_utterance(build_babble_utterance(fly_id="fly_a", articulatory=[ctrl]))
    promoted = store.promote_utterance(row.utterance_id, alphabet="ab")
    assert promoted is not None
    assert promoted.status == "promoted"
    assert promoted.alphabet == "ab"
    retired = store.retire_utterance(row.utterance_id)
    assert retired is not None
    assert retired.status == "retired"


def test_looks_like_word() -> None:
    assert looks_like_word("ma")
    assert looks_like_word("ba")
    assert looks_like_word("mona")
    assert looks_like_word("MA!")  # non-letters stripped
    assert not looks_like_word("#")
    assert not looks_like_word("a")
    assert not looks_like_word("mmm")
    assert not looks_like_word("sss")
    assert not looks_like_word("ae")  # vowels only
    assert not looks_like_word("st")  # consonants only
    assert not looks_like_word("m", min_len=2)


def test_reinforce_wordlike_babble_increases_weight() -> None:
    store = InMemorySpeechCortex()
    program = UtteranceProgram(
        utterance_id="u-word",
        alphabet="ma",
        status="candidate",
        source="babble",
        fly_id="fly_a",
        score=0.0,
    )
    store.upsert_utterance(program)
    stored, word = reinforce_wordlike_babble(store, program, delta=0.1)
    assert word is not None
    assert word.word == "ma"
    assert word.weight == pytest.approx(0.1)
    assert word.hits == 1
    assert stored.score == pytest.approx(0.1)
    stored2, word2 = reinforce_wordlike_babble(store, stored, delta=0.1)
    assert word2 is not None
    assert word2.weight == pytest.approx(0.2)
    assert word2.hits == 2
    assert stored2.score == pytest.approx(0.2)
    listed = store.list_words(fly_id="fly_a")
    assert listed and listed[0].word == "ma"


def test_reinforce_skips_non_wordlike_and_hear() -> None:
    store = InMemorySpeechCortex()
    babble = UtteranceProgram(
        utterance_id="u-noise",
        alphabet="mmm",
        status="candidate",
        source="babble",
        fly_id="fly_a",
    )
    store.upsert_utterance(babble)
    row, word = reinforce_wordlike_babble(store, babble, delta=0.2)
    assert word is None
    assert row.score == 0.0
    assert store.list_words(fly_id="fly_a") == []

    hear = UtteranceProgram(
        utterance_id="u-hear",
        alphabet="ma",
        status="candidate",
        source="hear",
        fly_id="fly_a",
    )
    store.upsert_utterance(hear)
    _, word_hear = reinforce_wordlike_babble(store, hear, delta=0.2)
    assert word_hear is None
