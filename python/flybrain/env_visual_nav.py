"""Fake visual navigation env: synthetic eyes, no privileged target XYZ in obs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flybrain.env import FakeFlyConfig, FakeFlyEnv
from flybrain.phase5 import EVAL_EPISODE_STEPS, SUCCESS_RADIUS, VISION_SIZE
from flybrain.unity_env import DEFAULT_ACTION_DIM, DEFAULT_OBS_DIM
from flybrain.vision.synthetic import render_synthetic_frame


@dataclass
class FakeVisualNavConfig(FakeFlyConfig):
    arena_half_extent: float = 0.45
    success_radius: float = SUCCESS_RADIUS
    max_steps: int = EVAL_EPISODE_STEPS
    vision_size: int = VISION_SIZE
    observation_dim: int = DEFAULT_OBS_DIM
    action_dim: int = DEFAULT_ACTION_DIM


class FakeVisualNavEnv(FakeFlyEnv):
    """2D pose in a square arena; agent obs is vector-only (no target coords).

    Privileged target/wall metrics live only in ``info`` for evaluation.
    """

    def __init__(self, config: FakeVisualNavConfig | None = None) -> None:
        self.nav_config = config or FakeVisualNavConfig()
        super().__init__(self.nav_config)
        self._pos = np.zeros(2, dtype=np.float32)
        self._yaw = 0.0
        self._target = np.array([0.3, 0.2], dtype=np.float32)
        self._left_eye = np.zeros(
            (1, self.nav_config.vision_size, self.nav_config.vision_size), np.float32
        )
        self._right_eye = np.zeros_like(self._left_eye)
        self._wall_hits = 0
        self._start_dist = 0.0
        self._dist = 0.0
        self._bearing = 0.0

    @property
    def left_eye(self) -> np.ndarray:
        return self._left_eye

    @property
    def right_eye(self) -> np.ndarray:
        return self._right_eye

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self._energy = 1.0
        extent = self.nav_config.arena_half_extent * 0.5
        self._pos = self._rng.uniform(-extent, extent, size=2).astype(np.float32)
        self._yaw = float(self._rng.uniform(-np.pi, np.pi))
        for _ in range(32):
            self._target = self._rng.uniform(
                -self.nav_config.arena_half_extent * 0.8,
                self.nav_config.arena_half_extent * 0.8,
                size=2,
            ).astype(np.float32)
            if np.linalg.norm(self._target - self._pos) > 0.2:
                break
        self._wall_hits = 0
        self._update_vision()
        self._dist = float(np.linalg.norm(self._target - self._pos))
        self._start_dist = self._dist
        info = {
            **info,
            "distance_to_target": self._dist,
            "wall_hits": self._wall_hits,
            "has_target_xyz_in_obs": False,
            "agent_obs_dim": int(obs.shape[0]),
        }
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        forward = float(np.clip(action[0] if action.size > 0 else 0.0, -1.0, 1.0))
        turn = float(np.clip(action[1] if action.size > 1 else 0.0, -1.0, 1.0))
        self._yaw += turn * 0.45
        direction = np.array([np.sin(self._yaw), np.cos(self._yaw)], dtype=np.float32)
        self._pos = self._pos + direction * (forward * 0.05)

        hit = False
        half = self.nav_config.arena_half_extent
        for i in range(2):
            if self._pos[i] > half:
                self._pos[i] = half
                hit = True
            elif self._pos[i] < -half:
                self._pos[i] = -half
                hit = True
        if hit:
            self._wall_hits += 1

        # Keep a stable vector obs without draining energy (nav-focused episodes).
        self._step += 1
        self._energy = 1.0
        self._obs = np.clip(
            self._obs + self._rng.normal(0.0, 0.002, size=self._obs.shape), -1.0, 1.0
        ).astype(np.float32)
        self._obs[0] = self._energy
        obs = self._obs.copy()

        self._update_vision()
        prev = self._dist
        self._dist = float(np.linalg.norm(self._target - self._pos))
        approach = prev - self._dist
        reward = float(approach * 5.0 - (0.5 if hit else 0.0) - 0.01 * abs(forward))
        success = self._dist <= self.nav_config.success_radius
        if success:
            reward += 1.0
        terminated = bool(success)
        truncated = self._step >= self.nav_config.max_steps

        info = {
            "effort": float(np.mean(np.abs(action))),
            "distance_to_target": self._dist,
            "start_distance": self._start_dist,
            "wall_hits": self._wall_hits,
            "success": success,
            "has_target_xyz_in_obs": False,
            "agent_obs_dim": int(obs.shape[0]),
            "pos": self._pos.copy(),
            "yaw": self._yaw,
            "_privileged_target": self._target.copy(),
            "_privileged_bearing": self._bearing,
        }
        return obs, reward, terminated, truncated, info

    def vision_policy_action(self) -> np.ndarray:
        """Greedy action from rendered eyes via centroid heuristic (no target XYZ)."""
        from flybrain.vision.recurrent_head import heuristic_cues_from_frames

        cues = heuristic_cues_from_frames(self._left_eye, self._right_eye)
        # Prefer turning toward bright mass; move forward when confident and clear.
        forward = float(
            np.clip(0.55 + 0.4 * cues.target_confidence - 0.9 * cues.wall_hazard, -0.2, 1.0)
        )
        turn = float(np.clip(cues.target_bearing * max(0.35, cues.target_confidence), -1.0, 1.0))
        return np.asarray([forward, turn, 0.5, 0.5, 0.0], dtype=np.float32)

    def _update_vision(self) -> None:
        rel = self._target - self._pos
        dist = float(np.linalg.norm(rel) + 1e-6)
        world_angle = float(np.arctan2(rel[0], rel[1]))
        bearing_rad = world_angle - self._yaw
        bearing = float(
            np.clip(np.arctan2(np.sin(bearing_rad), np.cos(bearing_rad)) / np.pi, -1.0, 1.0)
        )
        self._bearing = bearing
        half = self.nav_config.arena_half_extent
        forward = np.array([np.sin(self._yaw), np.cos(self._yaw)], dtype=np.float32)
        right = np.array([forward[1], -forward[0]], dtype=np.float32)
        wall_left = self._wall_proximity(-right, half) * 0.65
        wall_right = self._wall_proximity(right, half) * 0.65
        norm_dist = float(np.clip(dist / (half * 2.0), 0.0, 1.0))
        self._left_eye = render_synthetic_frame(
            bearing=bearing,
            distance=norm_dist,
            wall_left=wall_left,
            wall_right=wall_right,
            size=self.nav_config.vision_size,
            stereo_offset=-0.04,
        )
        self._right_eye = render_synthetic_frame(
            bearing=bearing,
            distance=norm_dist,
            wall_left=wall_left,
            wall_right=wall_right,
            size=self.nav_config.vision_size,
            stereo_offset=0.04,
        )

    def _wall_proximity(self, direction: np.ndarray, half: float) -> float:
        t_candidates = []
        for axis in range(2):
            d = float(direction[axis])
            if abs(d) < 1e-6:
                continue
            if d > 0:
                t_candidates.append((half - float(self._pos[axis])) / d)
            else:
                t_candidates.append((-half - float(self._pos[axis])) / d)
        t = min((t for t in t_candidates if t > 0), default=half * 2)
        return float(np.clip(1.0 - t / (half * 1.5), 0.0, 1.0))
