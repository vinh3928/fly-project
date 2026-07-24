"""Unity eye-frame unpacking for live-mind wall_hazard."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from flybrain.unity_env import to_grayscale_eye, unpack_unity_observations
from flybrain.vision.recurrent_head import heuristic_cues_from_frames


def test_to_grayscale_eye_chw_and_hwc() -> None:
    chw = np.random.rand(1, 64, 64).astype(np.float32)
    out = to_grayscale_eye(chw)
    assert out.shape == (1, 64, 64)
    hwc = np.random.rand(32, 32, 3).astype(np.float32)
    out2 = to_grayscale_eye(hwc, size=64)
    assert out2.shape == (1, 64, 64)


def test_unpack_unity_observations_named_eyes() -> None:
    vector = np.linspace(-1, 1, 197, dtype=np.float32)
    left = np.full((1, 64, 64), 0.2, dtype=np.float32)
    right = np.full((64, 64, 1), 0.8, dtype=np.float32)
    specs = [
        SimpleNamespace(shape=(197,), name="VectorSensor_size197"),
        SimpleNamespace(shape=(1, 64, 64), name="LeftEye"),
        SimpleNamespace(shape=(64, 64, 1), name="RightEye"),
    ]
    v, l, r = unpack_unity_observations([vector, left, right], specs)
    assert v.shape == (197,)
    assert l is not None and l.shape == (1, 64, 64)
    assert r is not None and r.shape == (1, 64, 64)
    assert float(l.mean()) < float(r.mean())


def test_heuristic_detects_light_near_wall() -> None:
    # Uniform bright fill in both eyes (Unity-like near wall).
    frame = np.full((1, 64, 64), 0.72, dtype=np.float32)
    cues = heuristic_cues_from_frames(frame, frame)
    assert cues.wall_hazard >= 0.45


def test_heuristic_still_detects_dark_synthetic_bars() -> None:
    frame = np.full((64, 64), 0.4, dtype=np.float32)
    frame[:, :8] = 0.05
    left = frame[None, ...]
    right = frame[None, ...]
    cues = heuristic_cues_from_frames(left, right)
    assert cues.wall_hazard >= 0.5
