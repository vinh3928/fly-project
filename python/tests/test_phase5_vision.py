from __future__ import annotations

import numpy as np
import torch

from flybrain.env_visual_nav import FakeVisualNavEnv
from flybrain.phase5 import AGENT_OBS_HAS_TARGET_XYZ, SUCCESS_RADIUS, VISION_SIZE
from flybrain.unity_env import DEFAULT_OBS_DIM
from flybrain.vision import (
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
    assert cues.shape == (2, 5)
    assert hidden.shape[0] == 1
    assert hidden.shape[1] == 2


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
