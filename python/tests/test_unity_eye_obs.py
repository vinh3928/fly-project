"""Unity eye-frame unpacking for live-mind wall_hazard."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from flybrain.unity_env import to_eye_frame, to_grayscale_eye, unpack_unity_observations
from flybrain.vision.recurrent_head import heuristic_cues_from_frames


def test_to_grayscale_eye_chw_and_hwc() -> None:
    chw = np.random.rand(1, 64, 64).astype(np.float32)
    out = to_grayscale_eye(chw)
    assert out.shape == (1, 64, 64)
    hwc = np.random.rand(32, 32, 3).astype(np.float32)
    out2 = to_grayscale_eye(hwc, size=64)
    assert out2.shape == (1, 64, 64)


def test_to_eye_frame_preserves_rgb() -> None:
    hwc = np.zeros((32, 32, 3), dtype=np.float32)
    hwc[..., 0] = 0.95
    hwc[..., 1] = 0.75
    hwc[..., 2] = 0.2
    rgb = to_eye_frame(hwc, size=64, prefer_color=True)
    assert rgb.shape == (3, 64, 64)
    assert float(rgb[0].mean()) > float(rgb[2].mean())
    gray = to_eye_frame(hwc, size=64, prefer_color=False)
    assert gray.shape == (1, 64, 64)


def test_unpack_unity_observations_named_eyes() -> None:
    vector = np.linspace(-1, 1, 197, dtype=np.float32)
    left = np.full((1, 64, 64), 0.2, dtype=np.float32)
    right = np.full((64, 64, 1), 0.8, dtype=np.float32)
    specs = [
        SimpleNamespace(shape=(197,), name="VectorSensor_size197"),
        SimpleNamespace(shape=(1, 64, 64), name="LeftEye"),
        SimpleNamespace(shape=(64, 64, 1), name="RightEye"),
    ]
    v, left_eye, right_eye = unpack_unity_observations([vector, left, right], specs)
    assert v.shape == (197,)
    assert left_eye is not None and left_eye.shape[0] in (1, 3) and left_eye.shape[-2:] == (64, 64)
    assert (
        right_eye is not None and right_eye.shape[0] in (1, 3) and right_eye.shape[-2:] == (64, 64)
    )
    assert float(left_eye.mean()) < float(right_eye.mean())


def test_unpack_unity_observations_preserves_rgb_eyes() -> None:
    vector = np.zeros(197, dtype=np.float32)
    left = np.zeros((64, 64, 3), dtype=np.float32)
    left[..., 0] = 0.9
    right = np.zeros((3, 64, 64), dtype=np.float32)
    right[2] = 0.8
    specs = [
        SimpleNamespace(shape=(197,), name="VectorSensor_size197"),
        SimpleNamespace(shape=(64, 64, 3), name="LeftEye"),
        SimpleNamespace(shape=(3, 64, 64), name="RightEye"),
    ]
    _, left_eye, right_eye = unpack_unity_observations(
        [vector, left, right], specs, prefer_color=True
    )
    assert left_eye is not None and left_eye.shape == (3, 64, 64)
    assert right_eye is not None and right_eye.shape == (3, 64, 64)
    assert float(left_eye[0].mean()) > 0.8
    assert float(right_eye[2].mean()) > 0.7


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


def test_heuristic_accepts_rgb_frames() -> None:
    frame = np.full((3, 64, 64), 0.72, dtype=np.float32)
    cues = heuristic_cues_from_frames(frame, frame)
    assert cues.wall_hazard >= 0.45
