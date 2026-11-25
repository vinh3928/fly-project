"""Articulatory controls + deterministic phoneme/letter tables (no ML ASR/TTS).

Control vector (7-d):
  jaw, lip_open, tongue_front, tongue_height, voicing, pitch, amplitude
"""

from __future__ import annotations

from typing import Any

import numpy as np

ARTICULATORY_DIM = 7
AUDIO_BAND_DIM = 16

# Coarse phoneme inventory used for babble + hear mapping.
PHONEMES: tuple[str, ...] = (
    "a",
    "e",
    "i",
    "o",
    "u",
    "m",
    "b",
    "p",
    "s",
    "t",
    "n",
    "l",
    "#",  # silence / unknown
)

PHONEME_TO_LETTER: dict[str, str] = {
    "a": "a",
    "e": "e",
    "i": "i",
    "o": "o",
    "u": "u",
    "m": "m",
    "b": "b",
    "p": "p",
    "s": "s",
    "t": "t",
    "n": "n",
    "l": "l",
    "#": "#",
}


def clip_articulatory(vec: np.ndarray | list[float] | None) -> np.ndarray:
    """Clip a 7-d articulatory command into valid ranges."""
    raw = np.asarray(vec if vec is not None else np.zeros(ARTICULATORY_DIM), dtype=np.float32)
    out = np.zeros(ARTICULATORY_DIM, dtype=np.float32)
    n = min(ARTICULATORY_DIM, int(raw.size))
    out[:n] = raw[:n]
    # jaw, lip, tongue_front, tongue_height, voicing, amplitude ∈ [0, 1]
    for i in (0, 1, 2, 3, 4, 6):
        out[i] = float(np.clip(out[i], 0.0, 1.0))
    # pitch ∈ [-1, 1]
    out[5] = float(np.clip(out[5], -1.0, 1.0))
    return out


def _bin3(x: float) -> int:
    return int(np.clip(int(float(x) * 3.0), 0, 2))


def articulatory_to_phoneme(ctrl: np.ndarray) -> str:
    """Map one articulatory frame to a coarse phoneme id."""
    c = clip_articulatory(ctrl)
    amp = float(c[6])
    voice = float(c[4])
    if amp < 0.08 or voice < 0.12:
        return "#"
    jaw = _bin3(float(c[0]))
    lip = _bin3(float(c[1]))
    tf = _bin3(float(c[2]))
    th = _bin3(float(c[3]))
    # Vowels: open tract + voicing.
    if lip >= 1 and jaw >= 1:
        if th >= 2 and tf >= 2:
            return "i"
        if th >= 2 and tf <= 0:
            return "u"
        if jaw >= 2:
            return "a"
        if tf >= 2:
            return "e"
        return "o"
    # Consonants: closed / constricted.
    if lip <= 0 and jaw <= 1:
        if voice >= 0.55:
            return "m" if th <= 1 else "b"
        return "p"
    if tf >= 2 and th <= 1:
        return "s" if voice < 0.45 else "t"
    if tf <= 0:
        return "n" if voice >= 0.4 else "l"
    return "a" if voice >= 0.5 else "#"


def articulatory_to_phonemes(sequence: list[np.ndarray] | np.ndarray) -> list[str]:
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    out: list[str] = []
    for row in arr:
        ph = articulatory_to_phoneme(row)
        if not out or out[-1] != ph:
            out.append(ph)
    return out or ["#"]


def articulatory_to_letters(sequence: list[np.ndarray] | np.ndarray) -> str:
    phonemes = articulatory_to_phonemes(sequence)
    chars = [PHONEME_TO_LETTER.get(p, "#") for p in phonemes]
    return "".join(c for c in chars if c != "#") or "#"


def _prototype_bands() -> dict[str, np.ndarray]:
    """Hand-built 16-band spectral prototypes for env-only hearing."""
    proto: dict[str, np.ndarray] = {}
    for i, name in enumerate(PHONEMES):
        v = np.zeros(AUDIO_BAND_DIM, dtype=np.float32)
        if name == "#":
            proto[name] = v
            continue
        center = 1 + (i % (AUDIO_BAND_DIM - 2))
        v[center] = 1.0
        if center > 0:
            v[center - 1] = 0.45
        if center + 1 < AUDIO_BAND_DIM:
            v[center + 1] = 0.45
        # Vowels emphasize lower bands; fricatives higher.
        if name in ("a", "e", "o", "u", "i"):
            v[:4] += 0.35
        if name in ("s", "t"):
            v[-4:] += 0.55
        if name in ("m", "b", "p", "n"):
            v[2:6] += 0.4
        n = float(np.linalg.norm(v)) or 1.0
        proto[name] = (v / n).astype(np.float32)
    return proto


_PROTOTYPES = _prototype_bands()


def clip_audio_bands(bands: np.ndarray | list[float] | None) -> np.ndarray:
    raw = np.asarray(bands if bands is not None else np.zeros(AUDIO_BAND_DIM), dtype=np.float32)
    out = np.zeros(AUDIO_BAND_DIM, dtype=np.float32)
    n = min(AUDIO_BAND_DIM, int(raw.size))
    out[:n] = np.clip(raw[:n], 0.0, 1.0)
    return out


def spectral_to_phoneme(bands: np.ndarray, *, rms: float = 0.0, voiced: float = 0.0) -> str:
    b = clip_audio_bands(bands)
    energy = float(np.mean(b)) if rms <= 0.0 else float(rms)
    if energy < 0.05:
        return "#"
    n = float(np.linalg.norm(b)) or 1.0
    unit = b / n
    best = "#"
    best_sim = -1.0
    for name, proto in _PROTOTYPES.items():
        if name == "#":
            continue
        sim = float(np.dot(unit, proto))
        if voiced < 0.2 and name in ("a", "e", "i", "o", "u", "m", "n", "l", "b"):
            sim *= 0.85
        if sim > best_sim:
            best_sim = sim
            best = name
    return best if best_sim >= 0.15 else "#"


def spectral_to_phonemes(
    frames: list[np.ndarray] | np.ndarray,
    *,
    rms: float | list[float] = 0.0,
    voiced: float | list[float] = 0.0,
) -> list[str]:
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if np.isscalar(rms):
        rms_list = [float(rms)] * int(arr.shape[0])
    else:
        rms_list = [float(x) for x in list(rms)]
        while len(rms_list) < int(arr.shape[0]):
            rms_list.append(rms_list[-1] if rms_list else 0.0)
    if np.isscalar(voiced):
        voiced_list = [float(voiced)] * int(arr.shape[0])
    else:
        voiced_list = [float(x) for x in list(voiced)]
        while len(voiced_list) < int(arr.shape[0]):
            voiced_list.append(voiced_list[-1] if voiced_list else 0.0)
    out: list[str] = []
    for i, row in enumerate(arr):
        ph = spectral_to_phoneme(row, rms=rms_list[i], voiced=voiced_list[i])
        if not out or out[-1] != ph:
            out.append(ph)
    return out or ["#"]


def spectral_to_letters(
    frames: list[np.ndarray] | np.ndarray,
    *,
    rms: float | list[float] = 0.0,
    voiced: float | list[float] = 0.0,
) -> str:
    phonemes = spectral_to_phonemes(frames, rms=rms, voiced=voiced)
    chars = [PHONEME_TO_LETTER.get(p, "#") for p in phonemes]
    return "".join(c for c in chars if c != "#") or "#"


def phonemes_to_letters(phonemes: list[str]) -> str:
    chars = [PHONEME_TO_LETTER.get(str(p), "#") for p in phonemes]
    return "".join(c for c in chars if c != "#") or "#"


def articulatory_dict(ctrl: np.ndarray) -> dict[str, float]:
    c = clip_articulatory(ctrl)
    keys = (
        "jaw",
        "lip_open",
        "tongue_front",
        "tongue_height",
        "voicing",
        "pitch",
        "amplitude",
    )
    return {k: float(c[i]) for i, k in enumerate(keys)}


def parse_articulatory_sequence(raw: Any) -> list[np.ndarray]:
    if raw is None:
        return []
    if isinstance(raw, (str, bytes)):
        import json

        raw = json.loads(raw)
    seq = []
    for row in list(raw or []):
        seq.append(clip_articulatory(row))
    return seq
