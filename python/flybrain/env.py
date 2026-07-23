from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flybrain.unity_env import DEFAULT_ACTION_DIM, DEFAULT_OBS_DIM, default_build_path

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None
    spaces = None


@dataclass
class FakeFlyConfig:
    observation_dim: int = DEFAULT_OBS_DIM
    action_dim: int = DEFAULT_ACTION_DIM
    max_steps: int = 200
    seed: int = 42


class FakeFlyEnv(gym.Env if gym is not None else object):
    """Gymnasium-compatible stub used before the Unity binary is available."""

    metadata = {"render_modes": []}

    def __init__(self, config: FakeFlyConfig | None = None) -> None:
        if gym is None:
            raise ImportError("gymnasium is required for FakeFlyEnv")
        self.config = config or FakeFlyConfig()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.config.observation_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.config.action_dim,), dtype=np.float32
        )
        self._rng = np.random.default_rng(self.config.seed)
        self._step = 0
        self._energy = 1.0
        self._obs = np.zeros(self.config.observation_dim, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step = 0
        self._energy = 1.0
        self._obs = self._rng.normal(0.0, 0.1, size=self.config.observation_dim).astype(np.float32)
        self._write_internal()
        return self._obs.copy(), {"seed": seed if seed is not None else self.config.seed}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        self._step += 1
        effort = float(np.mean(np.abs(action)))
        self._energy = max(0.0, self._energy - 0.002 - 0.01 * effort)
        delta = 0.05 * action.mean()
        noise = self._rng.normal(0.0, 0.01, size=self._obs.shape)
        self._obs = np.clip(self._obs + noise, -1.0, 1.0)
        self._write_internal()
        reward = -0.05 * effort + (0.01 if action[0] > 0 else 0.0)
        terminated = self._energy <= 0.0
        truncated = self._step >= self.config.max_steps
        info = {"effort": effort, "delta": delta, "energy": self._energy}
        return self._obs.copy(), float(reward), terminated, truncated, info

    def _write_internal(self) -> None:
        """Pack homeostasis into the Unity SensorManager layout (energy at index 176)."""
        # joints(72)+contacts(30)+foot_taste(48)+antennae(16)+proboscis(10) = 176
        internal = 176
        if self._obs.shape[0] <= internal + 7:
            self._obs[0] = self._energy
            return
        self._obs[internal] = self._energy
        self._obs[internal + 1] = 1.0  # hydration
        self._obs[internal + 2] = 0.0  # fatigue
        self._obs[internal + 3] = 0.0  # damage
        self._obs[internal + 4] = 0.5  # temperature normalized
        self._obs[internal + 5] = 0.0  # toxin
        self._obs[internal + 6] = max(0.0, 1.0 - self._energy)  # hunger
        self._obs[internal + 7] = 0.5  # arousal


def make_env(*, prefer_unity: bool = False):
    """Return UnityFlyEnv when requested and a build exists; otherwise FakeFlyEnv."""
    if prefer_unity:
        from flybrain.unity_env import UnityFlyConfig, UnityFlyEnv

        build = default_build_path()
        if build is None:
            raise FileNotFoundError(
                "No Unity build under builds/. Run ./scripts/build_unity_player.sh first."
            )
        return UnityFlyEnv(UnityFlyConfig(file_name=str(build)))
    return FakeFlyEnv()
