"""Unity ML-Agents environment wrapper for the fly chamber player."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None
    spaces = None

BEHAVIOR_NAME = "FlyAgent"
DEFAULT_OBS_DIM = 197
DEFAULT_ACTION_DIM = 5


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_build_path() -> Path | None:
    """Return the first existing standalone player path, if any."""
    builds = repo_root() / "builds"
    candidates = [
        builds / "FlyChamber.app",
        builds / "FlyChamber.exe",
        builds / "FlyChamber.x86_64",
        builds / "FlyChamber",
    ]
    for path in candidates:
        if path.exists():
            return path
    env_path = os.environ.get("FLY_UNITY_BUILD")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
    return None


@dataclass
class UnityFlyConfig:
    file_name: str | None = None
    worker_id: int = 0
    base_port: int = 5005
    no_graphics: bool = True
    seed: int = 42
    timeout_wait: int = 120
    behavior_name: str = BEHAVIOR_NAME


class UnityFlyEnv(gym.Env if gym is not None else object):
    """Gymnasium wrapper around ``mlagents_envs.UnityEnvironment``.

    Pass ``file_name=None`` to connect to a Unity Editor play session
    (press Play in the editor first, with the ML-Agents port free).
    Pass a player path (or rely on ``builds/FlyChamber.*``) to launch the binary.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: UnityFlyConfig | None = None) -> None:
        if gym is None:
            raise ImportError("gymnasium is required for UnityFlyEnv")
        try:
            from mlagents_envs.environment import UnityEnvironment
            from mlagents_envs.side_channel.engine_configuration_channel import (
                EngineConfigurationChannel,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "mlagents_envs is required. Install with: pip install -e '.[mlagents]'"
            ) from exc

        self.config = config or UnityFlyConfig()
        file_name = self.config.file_name
        if file_name is None:
            build = default_build_path()
            file_name = str(build) if build is not None else None

        self._engine_channel = EngineConfigurationChannel()
        self._env = UnityEnvironment(
            file_name=file_name,
            worker_id=self.config.worker_id,
            base_port=self.config.base_port,
            seed=self.config.seed,
            no_graphics=self.config.no_graphics if file_name else False,
            timeout_wait=self.config.timeout_wait,
            side_channels=[self._engine_channel],
        )
        if file_name is not None:
            self._engine_channel.set_configuration_parameters(
                time_scale=20.0,
                width=84,
                height=84,
                quality_level=1,
            )

        self._behavior = self.config.behavior_name
        self._env.reset()
        self._ensure_behavior()
        spec = self._env.behavior_specs[self._behavior]
        obs_size = int(spec.observation_specs[0].shape[0])
        act_size = int(spec.action_spec.continuous_size)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_size,), dtype=np.float32
        )
        self._steps = 0

    def _ensure_behavior(self) -> None:
        names = list(self._env.behavior_specs.keys())
        if not names:
            raise RuntimeError("Unity environment reported no ML-Agents behaviors")
        if self._behavior not in names:
            # Editor/build may register "FlyAgent?team=0" — match prefix.
            match = next((n for n in names if n.startswith(self._behavior)), names[0])
            self._behavior = match

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        if seed is not None:
            # UnityEnvironment seed is fixed at construction; still track requested seed.
            pass
        self._env.reset()
        self._ensure_behavior()
        self._steps = 0
        obs = self._current_obs()
        return obs, {"seed": seed if seed is not None else self.config.seed, "behavior": self._behavior}

    def step(self, action):
        from mlagents_envs.base_env import ActionTuple

        action = np.asarray(action, dtype=np.float32).reshape(1, -1)
        self._env.set_actions(self._behavior, ActionTuple(continuous=action))
        self._env.step()
        self._steps += 1
        decision, terminal = self._env.get_steps(self._behavior)

        if len(terminal) > 0:
            obs = np.asarray(terminal.obs[0][0], dtype=np.float32)
            reward = float(terminal.reward[0])
            return obs, reward, True, False, {"steps": self._steps}

        if len(decision) == 0:
            raise RuntimeError("Unity returned no decision or terminal steps")

        obs = np.asarray(decision.obs[0][0], dtype=np.float32)
        reward = float(decision.reward[0])
        return obs, reward, False, False, {"steps": self._steps}

    def _current_obs(self) -> np.ndarray:
        decision, terminal = self._env.get_steps(self._behavior)
        if len(terminal) > 0:
            return np.asarray(terminal.obs[0][0], dtype=np.float32)
        if len(decision) > 0:
            return np.asarray(decision.obs[0][0], dtype=np.float32)
        # First reset can leave an empty queue until one Academy step.
        self._env.step()
        decision, terminal = self._env.get_steps(self._behavior)
        if len(terminal) > 0:
            return np.asarray(terminal.obs[0][0], dtype=np.float32)
        if len(decision) > 0:
            return np.asarray(decision.obs[0][0], dtype=np.float32)
        raise RuntimeError("No observations available after Unity reset")

    def close(self) -> None:
        if getattr(self, "_env", None) is not None:
            self._env.close()
            self._env = None  # type: ignore[assignment]

    def __enter__(self) -> UnityFlyEnv:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
