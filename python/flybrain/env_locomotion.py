"""Fake locomotion env: goal-conditioned walking on a 2D plane (Phase 9)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flybrain.env import FakeFlyConfig, FakeFlyEnv
from flybrain.motor.rewards import LocomotionRewardConfig, compute_locomotion_reward
from flybrain.phase9 import (
    BODY_OBS_DIM,
    LOCOMOTION_EPISODE_STEPS,
    MOTOR_GOAL_DIM,
    MOTOR_TRAINING_OBS_DIM,
)
from flybrain.unity_env import DEFAULT_ACTION_DIM


@dataclass
class FakeLocomotionConfig(FakeFlyConfig):
    arena_half_extent: float = 0.5
    max_steps: int = LOCOMOTION_EPISODE_STEPS
    observation_dim: int = BODY_OBS_DIM
    action_dim: int = DEFAULT_ACTION_DIM
    include_goal_in_obs: bool = False
    step_scale: float = 0.04
    turn_scale: float = 0.35


class FakeLocomotionEnv(FakeFlyEnv):
    """Pose integrates motor actions; external mind goals drive tracking reward."""

    def __init__(self, config: FakeLocomotionConfig | None = None) -> None:
        self.loco_config = config or FakeLocomotionConfig()
        # Gym space uses body obs; goal may be appended in _obs_vector.
        obs_dim = (
            MOTOR_TRAINING_OBS_DIM
            if self.loco_config.include_goal_in_obs
            else self.loco_config.observation_dim
        )
        cfg = FakeFlyConfig(
            observation_dim=obs_dim,
            action_dim=self.loco_config.action_dim,
            max_steps=self.loco_config.max_steps,
            seed=self.loco_config.seed,
        )
        super().__init__(cfg)
        self._pos = np.zeros(2, dtype=np.float32)
        self._yaw = 0.0
        self._goal = np.asarray([0.0, 0.0, 0.5, 0.5, 0.0], dtype=np.float32)
        self._start_pos = self._pos.copy()
        self._prev_pos = self._pos.copy()
        self._prev_yaw = 0.0
        self._fell = False
        self._tracking_errors: list[float] = []

    def set_goal(self, goal: np.ndarray) -> None:
        g = np.asarray(goal, dtype=np.float32).reshape(-1)
        out = np.zeros(MOTOR_GOAL_DIM, dtype=np.float32)
        out[: min(MOTOR_GOAL_DIM, g.size)] = g[:MOTOR_GOAL_DIM]
        out[0] = float(np.clip(out[0], -1.0, 1.0))
        out[1] = float(np.clip(out[1], -1.0, 1.0))
        out[2:] = np.clip(out[2:], 0.0, 1.0)
        self._goal = out

    def agent_position(self) -> np.ndarray:
        return self._pos.copy()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        _, info = super().reset(seed=seed, options=options)
        half = self.loco_config.arena_half_extent * 0.25
        self._pos = self._rng.uniform(-half, half, size=2).astype(np.float32)
        self._yaw = float(self._rng.uniform(-np.pi, np.pi))
        self._start_pos = self._pos.copy()
        self._prev_pos = self._pos.copy()
        self._prev_yaw = self._yaw
        self._fell = False
        self._tracking_errors = []
        # Default stand goal; caller / motor layer overrides.
        self._goal = np.asarray([0.0, 0.0, 0.5, 0.5, 0.0], dtype=np.float32)
        if options and "goal" in options:
            self.set_goal(np.asarray(options["goal"], dtype=np.float32))
        obs = self._write_obs()
        info = {
            **info,
            "goal": self._goal.copy(),
            "position": self._pos.copy(),
            "yaw": self._yaw,
            "forward_displacement": 0.0,
            "tracking_error": 0.0,
            "fell": False,
        }
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        forward = float(np.clip(action[0] if action.size > 0 else 0.0, -1.0, 1.0))
        turn = float(np.clip(action[1] if action.size > 1 else 0.0, -1.0, 1.0))

        self._prev_pos = self._pos.copy()
        self._prev_yaw = self._yaw
        self._yaw += turn * self.loco_config.turn_scale
        direction = np.array([np.sin(self._yaw), np.cos(self._yaw)], dtype=np.float32)
        self._pos = self._pos + direction * (forward * self.loco_config.step_scale)
        half = self.loco_config.arena_half_extent
        self._pos = np.clip(self._pos, -half, half)

        # Crude fall if commanded extreme mismatch while "unstable".
        self._fell = bool(abs(forward) > 0.95 and abs(turn) > 0.95 and self._rng.random() < 0.01)

        delta = self._pos - self._prev_pos
        body_forward = np.array([np.sin(self._prev_yaw), np.cos(self._prev_yaw)], dtype=np.float32)
        forward_progress = float(np.dot(delta, body_forward))
        yaw_progress = float(self._yaw - self._prev_yaw)
        tracking_error = float(np.mean(np.abs(action[:MOTOR_GOAL_DIM] - self._goal)))
        self._tracking_errors.append(tracking_error)

        reward = compute_locomotion_reward(
            goal=self._goal,
            action=action,
            forward_progress=forward_progress,
            yaw_progress=yaw_progress,
            upright=0.0 if self._fell else 1.0,
            fell=self._fell,
            config=LocomotionRewardConfig(),
        )

        self._step += 1
        obs = self._write_obs(action=action)
        disp = float(np.linalg.norm(self._pos - self._start_pos))
        terminated = self._fell
        truncated = self._step >= self.loco_config.max_steps
        info = {
            "goal": self._goal.copy(),
            "position": self._pos.copy(),
            "yaw": self._yaw,
            "forward_progress": forward_progress,
            "forward_displacement": disp,
            "tracking_error": tracking_error,
            "mean_tracking_error": float(np.mean(self._tracking_errors)),
            "fell": self._fell,
            "motor_action": action.copy(),
        }
        return obs, float(reward), terminated, truncated, info

    def _write_obs(self, action: np.ndarray | None = None) -> np.ndarray:
        body = np.zeros(BODY_OBS_DIM, dtype=np.float32)
        body[0:2] = self._pos
        body[2] = np.sin(self._yaw)
        body[3] = np.cos(self._yaw)
        body[176] = 1.0  # energy
        body[182] = 0.0  # hunger
        if action is not None and action.size >= 2:
            body[4] = action[0]
            body[5] = action[1]
        if self.loco_config.include_goal_in_obs:
            obs = np.concatenate([body, self._goal]).astype(np.float32)
        else:
            obs = body
        self._obs = obs
        return obs.copy()
