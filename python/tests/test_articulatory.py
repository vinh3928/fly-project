"""Deterministic articulatory / spectral → phoneme → letter table."""

from __future__ import annotations

import numpy as np

from flybrain.speech.articulatory import (
    ARTICULATORY_DIM,
    AUDIO_BAND_DIM,
    articulatory_to_letters,
    articulatory_to_phoneme,
    clip_articulatory,
    clip_audio_bands,
    phonemes_to_letters,
    spectral_to_letters,
    spectral_to_phonemes,
)


def test_clip_articulatory_shape_and_ranges() -> None:
    out = clip_articulatory([2.0, -1.0, 0.5, 0.5, 0.9, 3.0, 1.5])
    assert out.shape == (ARTICULATORY_DIM,)
    assert float(out[0]) == 1.0
    assert float(out[1]) == 0.0
    assert float(out[5]) == 1.0
    assert float(out[6]) == 1.0
    assert clip_articulatory(None).shape == (ARTICULATORY_DIM,)


def test_articulatory_ah_maps_to_vowel_letter() -> None:
    # Open jaw + lips + voice + amp → vowel (a/o/e/…).
    ctrl = clip_articulatory([0.9, 0.8, 0.4, 0.35, 0.95, 0.0, 0.9])
    ph = articulatory_to_phoneme(ctrl)
    assert ph in ("a", "e", "i", "o", "u")
    letters = articulatory_to_letters([ctrl, ctrl])
    assert letters and all(c.isalpha() or c == "#" for c in letters)


def test_silence_maps_to_hash() -> None:
    quiet = clip_articulatory([0.2, 0.2, 0.3, 0.3, 0.0, 0.0, 0.0])
    assert articulatory_to_phoneme(quiet) == "#"
    # Hash phonemes are dropped from the spelling string (unknown/silence).
    assert phonemes_to_letters(["#", "a", "#"]) == "a"
    assert phonemes_to_letters(["#", "#"]) == "#"


def test_spectral_to_letters_nonempty_on_energy() -> None:
    bands = clip_audio_bands(np.linspace(0.1, 0.9, AUDIO_BAND_DIM))
    letters = spectral_to_letters([bands], rms=0.4, voiced=1.0)
    assert isinstance(letters, str) and len(letters) >= 1
    ph = spectral_to_phonemes([bands], rms=0.4, voiced=1.0)
    assert ph
