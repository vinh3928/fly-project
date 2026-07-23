"""Synthetic stereo frames for offline visual-nav training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flybrain.phase5 import VISION_SIZE


@dataclass
class SyntheticNavSample:
    left: np.ndarray  # (1, H, W)
    right: np.ndarray
    target_bearing: float  # -1 .. 1 (left .. right)
    target_confidence: float
    wall_hazard: float
    forward_bias: float
    turn_bias: float


def render_synthetic_frame(
    *,
    bearing: float,
    distance: float,
    wall_left: float,
    wall_right: float,
    size: int = VISION_SIZE,
    stereo_offset: float = 0.0,
) -> np.ndarray:
    """Render a grayscale HxW frame with a bright target blob and dark wall bars."""
    frame = np.full((size, size), 0.35, dtype=np.float32)
    # Floor gradient
    yy = np.linspace(0.25, 0.55, size, dtype=np.float32)
    frame += yy[:, None] * 0.15

    # Wall bars intensity ~ proximity
    left_w = int(np.clip(wall_left * size * 0.35, 0, size // 3))
    right_w = int(np.clip(wall_right * size * 0.35, 0, size // 3))
    if left_w > 0:
        frame[:, :left_w] = np.minimum(frame[:, :left_w], 0.05)
    if right_w > 0:
        frame[:, size - right_w :] = np.minimum(frame[:, size - right_w :], 0.05)

    # Target blob: closer => larger / brighter
    cx = 0.5 + 0.45 * np.clip(bearing + stereo_offset, -1.0, 1.0)
    cy = 0.55 - 0.25 * np.clip(1.0 - distance, 0.0, 1.0)
    radius = max(1.5, (0.08 + 0.22 * (1.0 - distance)) * size)
    ys, xs = np.ogrid[:size, :size]
    dist2 = (xs - cx * size) ** 2 + (ys - cy * size) ** 2
    mask = dist2 <= radius**2
    frame[mask] = np.maximum(frame[mask], 0.95)

    return np.clip(frame, 0.0, 1.0)[None, ...].astype(np.float32)


def make_synthetic_sample(
    rng: np.random.Generator,
    *,
    size: int = VISION_SIZE,
) -> SyntheticNavSample:
    bearing = float(rng.uniform(-1.0, 1.0))
    distance = float(rng.uniform(0.05, 1.0))
    wall_left = float(rng.uniform(0.0, 1.0))
    wall_right = float(rng.uniform(0.0, 1.0))
    wall_hazard = float(np.clip(max(wall_left, wall_right), 0.0, 1.0))
    confidence = float(np.clip(1.1 - distance, 0.0, 1.0))
    forward_bias = float(np.clip(0.7 * confidence - 0.8 * wall_hazard, -1.0, 1.0))
    turn_bias = float(np.clip(bearing * confidence - 0.3 * (wall_right - wall_left), -1.0, 1.0))

    left = render_synthetic_frame(
        bearing=bearing,
        distance=distance,
        wall_left=wall_left,
        wall_right=wall_right,
        size=size,
        stereo_offset=-0.04,
    )
    right = render_synthetic_frame(
        bearing=bearing,
        distance=distance,
        wall_left=wall_left,
        wall_right=wall_right,
        size=size,
        stereo_offset=0.04,
    )
    return SyntheticNavSample(
        left=left,
        right=right,
        target_bearing=bearing,
        target_confidence=confidence,
        wall_hazard=wall_hazard,
        forward_bias=forward_bias,
        turn_bias=turn_bias,
    )


def make_synthetic_batch(
    batch_size: int,
    *,
    seed: int = 0,
    size: int = VISION_SIZE,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    samples = [make_synthetic_sample(rng, size=size) for _ in range(batch_size)]
    return {
        "left": np.stack([s.left for s in samples], axis=0),
        "right": np.stack([s.right for s in samples], axis=0),
        "target_bearing": np.asarray([s.target_bearing for s in samples], dtype=np.float32),
        "target_confidence": np.asarray([s.target_confidence for s in samples], dtype=np.float32),
        "wall_hazard": np.asarray([s.wall_hazard for s in samples], dtype=np.float32),
        "forward_bias": np.asarray([s.forward_bias for s in samples], dtype=np.float32),
        "turn_bias": np.asarray([s.turn_bias for s in samples], dtype=np.float32),
    }
