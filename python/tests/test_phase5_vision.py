from __future__ import annotations

import numpy as np
import torch

from flybrain.env_visual_nav import FakeVisualNavEnv
from flybrain.phase5 import AGENT_OBS_HAS_TARGET_XYZ, SUCCESS_RADIUS, VISION_SIZE
from flybrain.unity_env import DEFAULT_OBS_DIM
from flybrain.encoders import VisionEncoder
from flybrain.vision import (
    NUM_CUES,
    RecurrentVisionHead,
    heuristic_cues_from_frames,
    make_synthetic_batch,
    make_synthetic_sample,
)


def test_synthetic_sample_shapes_and_labels() -> None:
    rng = np.random.default_rng(0)
    sample = make_synthetic_sample(rng, size=VISION_SIZE)
    assert sample.left.shape == (1, VISION_SIZE, VISION_SIZE)
    assert sample.right.shape == (1, VISION_SIZE, VISION_SIZE)
    assert -1.0 <= sample.target_bearing <= 1.0
    assert 0.0 <= sample.wall_hazard <= 1.0
    batch = make_synthetic_batch(4, seed=1)
    assert batch["left"].shape == (4, 1, VISION_SIZE, VISION_SIZE)


def test_recurrent_vision_head_forward() -> None:
    head = RecurrentVisionHead()
    left = torch.randn(2, 1, VISION_SIZE, VISION_SIZE)
    right = torch.randn(2, 1, VISION_SIZE, VISION_SIZE)
    cues, hidden = head(left, right, None)
    assert cues.shape == (2, NUM_CUES)
    assert hidden.shape[0] == 1
    assert hidden.shape[1] == 2


def test_heuristic_wall_hazard_ignores_textured_grass_like_field() -> None:
    """Outdoor meadow clutter must not read as a near wall the way blank chamber walls do."""
    from flybrain.thinking.composer import WALL_HAZARD_REACT_MIN, compose_thought
    from flybrain.thinking.two_layer_future import DEFAULT_STIMULUS_THRESHOLDS, _stimulus_peaks

    rng = np.random.default_rng(11)
    grass = rng.normal(loc=0.55, scale=0.18, size=(64, 64)).astype(np.float32)
    grass = np.clip(grass, 0.0, 1.0)
    grass_cues = heuristic_cues_from_frames(grass, grass)

    blank_wall = np.full((64, 64), 0.75, dtype=np.float32)
    wall_cues = heuristic_cues_from_frames(blank_wall, blank_wall)

    dark = np.full((64, 64), 0.55, dtype=np.float32)
    dark[:, :8] = 0.05
    dark[:, -8:] = 0.05
    dark_cues = heuristic_cues_from_frames(dark, dark)

    assert grass_cues.wall_hazard < WALL_HAZARD_REACT_MIN
    assert wall_cues.wall_hazard >= 0.45 or dark_cues.wall_hazard >= 0.45
    assert grass_cues.wall_hazard < max(wall_cues.wall_hazard, dark_cues.wall_hazard)
    assert DEFAULT_STIMULUS_THRESHOLDS["wall"] == WALL_HAZARD_REACT_MIN == 0.45
    # Mild grass-range hazard must not peak as wall or slow the composer.
    mild = {"wall_hazard": 0.35, "energy": 1.0, "hunger": 0.0}
    assert "wall" not in _stimulus_peaks(mild, {})
    fast = compose_thought(
        signals=mild,
        affect={"survival_risk": 0.0, "exploration": 0.5, "joy": 0.2, "anger": 0.0, "altruism": 0.0},
        reflection="",
        present_future={"present": "", "future": ""},
        devil_critique="",
    )
    blocked = compose_thought(
        signals={**mild, "wall_hazard": 0.8},
        affect={"survival_risk": 0.0, "exploration": 0.5, "joy": 0.2, "anger": 0.0, "altruism": 0.0},
        reflection="",
        present_future={"present": "", "future": ""},
        devil_critique="",
    )
    assert float(fast.action[0]) > float(blocked.action[0])


def test_heuristic_cues_from_bright_target() -> None:
    rng = np.random.default_rng(2)
    sample = make_synthetic_sample(rng)
    # Force strong right-bearing sample
    from flybrain.vision.synthetic import render_synthetic_frame

    left = render_synthetic_frame(bearing=0.8, distance=0.3, wall_left=0.1, wall_right=0.1)
    right = render_synthetic_frame(bearing=0.8, distance=0.3, wall_left=0.1, wall_right=0.1)
    cues = heuristic_cues_from_frames(left, right)
    assert cues.target_bearing > 0.2
    assert cues.target_confidence > 0.0


def test_fake_visual_nav_no_target_xyz_in_obs() -> None:
    assert AGENT_OBS_HAS_TARGET_XYZ is False
    env = FakeVisualNavEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (DEFAULT_OBS_DIM,)
    assert info["has_target_xyz_in_obs"] is False
    # Ensure target coords are not embedded in obs by checking dim contract
    assert obs.shape[0] == DEFAULT_OBS_DIM
    action = np.array([0.5, 0.0, 0.5, 0.5, 0.0], dtype=np.float32)
    obs2, reward, term, trunc, info2 = env.step(action)
    assert obs2.shape == (DEFAULT_OBS_DIM,)
    assert "_privileged_target" in info2
    assert "has_target_xyz_in_obs" in info2 and info2["has_target_xyz_in_obs"] is False
    assert env.left_eye.shape == (1, VISION_SIZE, VISION_SIZE)


def test_fake_visual_nav_can_approach_with_heuristic() -> None:
    env = FakeVisualNavEnv()
    obs, info = env.reset(seed=7)
    start = info["distance_to_target"]
    for _ in range(100):
        action = env.vision_policy_action()
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            break
    assert info["has_target_xyz_in_obs"] is False
    assert obs.shape[0] == DEFAULT_OBS_DIM
    assert info["distance_to_target"] <= start - 0.05 or info.get("success", False)


def test_vision_encoder_accepts_rgb_when_built_for_grayscale() -> None:
    enc = VisionEncoder(in_channels=1, latent_dim=32)
    rgb = torch.rand(2, 3, 64, 64)
    out = enc(rgb)
    assert out.shape == (2, 32)


def test_vision_encoder_accepts_gray_when_built_for_rgb() -> None:
    enc = VisionEncoder(in_channels=3, latent_dim=32)
    gray = torch.rand(2, 1, 64, 64)
    out = enc(gray)
    assert out.shape == (2, 32)


def test_recurrent_head_infer_cues_rgb() -> None:
    head = RecurrentVisionHead()
    rgb = np.random.rand(3, 64, 64).astype(np.float32)
    cues = head.infer_cues(rgb, rgb)
    assert 0.0 <= cues.wall_hazard <= 1.0
