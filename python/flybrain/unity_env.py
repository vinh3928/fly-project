"""Unity ML-Agents environment wrapper for the fly chamber player."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

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
DEFAULT_VISION_SIZE = 64


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


def to_grayscale_eye(
    arr: np.ndarray,
    *,
    size: int = DEFAULT_VISION_SIZE,
) -> np.ndarray:
    """Normalize a visual observation to shape (1, H, W) float32 in [0, 1]."""
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3:
        # CHW (C,H,W) or HWC (H,W,C)
        if x.shape[0] in (1, 3) and x.shape[0] < x.shape[-1]:
            if x.shape[0] == 1:
                x = x[0]
            else:
                x = x.mean(axis=0)
        else:
            if x.shape[-1] == 1:
                x = x[..., 0]
            elif x.shape[-1] >= 3:
                x = x[..., :3].mean(axis=-1)
            else:
                x = x.mean(axis=-1)
    elif x.ndim == 1:
        n = int(np.sqrt(x.size))
        if n * n == x.size:
            x = x.reshape(n, n)
        else:
            x = np.zeros((size, size), dtype=np.float32)
    if x.ndim != 2:
        x = np.zeros((size, size), dtype=np.float32)
    if x.shape != (size, size):
        # Nearest resize without depending on cv2/PIL.
        ys = (np.linspace(0, x.shape[0] - 1, size)).astype(np.int32)
        xs = (np.linspace(0, x.shape[1] - 1, size)).astype(np.int32)
        x = x[ys][:, xs]
    if float(x.max()) > 1.5:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0).astype(np.float32)[None, ...]


def unpack_unity_observations(
    obs_list: Sequence[Any],
    specs: Sequence[Any],
    *,
    vector_dim: int = DEFAULT_OBS_DIM,
    vision_size: int = DEFAULT_VISION_SIZE,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Split ML-Agents step observations into vector + optional stereo eyes.

    Camera sensors registered as ``LeftEye`` / ``RightEye`` are preferred by name;
    otherwise the first two non-vector visual tensors are used.
    """
    vector: np.ndarray | None = None
    left: np.ndarray | None = None
    right: np.ndarray | None = None
    unnamed_visuals: list[np.ndarray] = []

    for i, raw in enumerate(obs_list):
        arr = np.asarray(raw, dtype=np.float32)
        # Batched (n_agents, ...) → first agent
        if arr.ndim >= 1 and specs and i < len(specs):
            pass
        if arr.ndim >= 1 and arr.shape[0] == 1 and arr.ndim >= 2 and arr.shape[1] != vector_dim:
            # Could be (1, H, W) already — keep
            pass
        # DecisionSteps passes obs[i] already indexed per agent in our caller via obs[0]
        name = ""
        shape: tuple[int, ...] = tuple(arr.shape)
        if i < len(specs):
            spec = specs[i]
            name = str(getattr(spec, "name", "") or "").lower()
            shape = tuple(getattr(spec, "shape", shape))

        flat = arr.reshape(-1)
        is_vector = (
            shape == (vector_dim,)
            or flat.size == vector_dim
            or (arr.ndim == 1 and flat.size >= vector_dim * 0.9 and flat.size <= vector_dim * 1.1)
        )
        # Prefer exact vector size; visual tensors are 2D/3D.
        if is_vector and arr.ndim <= 2 and (arr.ndim == 1 or arr.shape[-1] == vector_dim or flat.size == vector_dim):
            if flat.size >= vector_dim:
                vector = flat[:vector_dim].copy()
            else:
                vector = flat.copy()
            continue

        if arr.ndim >= 2 or (len(shape) >= 2 and shape[0] != vector_dim):
            eye = to_grayscale_eye(arr, size=vision_size)
            if "left" in name:
                left = eye
            elif "right" in name:
                right = eye
            else:
                unnamed_visuals.append(eye)

    for eye in unnamed_visuals:
        if left is None:
            left = eye
        elif right is None:
            right = eye
        else:
            break

    if vector is None:
        # Fallback: largest flat vector-like tensor
        best = None
        best_n = -1
        for raw in obs_list:
            flat = np.asarray(raw, dtype=np.float32).reshape(-1)
            score = flat.size
            if flat.size == vector_dim:
                score += 10_000
            if score > best_n:
                best_n = score
                best = flat
        if best is None:
            raise RuntimeError("Unity step had no observation tensors")
        vector = best[:vector_dim] if best.size >= vector_dim else best

    return vector.astype(np.float32), left, right


@dataclass
class UnityFlyConfig:
    file_name: str | None = None
    worker_id: int = 0
    base_port: int = 5005
    no_graphics: bool = True
    seed: int = 42
    timeout_wait: int = 120
    behavior_name: str = BEHAVIOR_NAME
    # When True, keep file_name=None so ML-Agents attaches to the open Editor
    # Play session. Do not fall back to builds/FlyChamber.*.
    editor_connect: bool = False
    vision_size: int = DEFAULT_VISION_SIZE


class UnityFlyEnv(gym.Env if gym is not None else object):
    """Gymnasium wrapper around ``mlagents_envs.UnityEnvironment``.

    Pass ``editor_connect=True`` (and ``file_name=None``) to connect to a Unity
    Editor play session (press Play first). Pass a player path to launch a build.

    When the Agent exposes LeftEye/RightEye camera sensors, frames are available
    via :attr:`left_eye` / :attr:`right_eye` for live-mind wall_hazard cues.
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
        if file_name is None and not self.config.editor_connect:
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
        self._left_eye: np.ndarray | None = None
        self._right_eye: np.ndarray | None = None
        self._env.reset()
        self._ensure_behavior()
        spec = self._env.behavior_specs[self._behavior]
        obs_size = DEFAULT_OBS_DIM
        for o in spec.observation_specs:
            if tuple(o.shape) == (DEFAULT_OBS_DIM,) or (
                len(o.shape) == 1 and int(o.shape[0]) == DEFAULT_OBS_DIM
            ):
                obs_size = int(o.shape[0])
                break
            if len(o.shape) == 1 and int(o.shape[0]) > obs_size:
                obs_size = int(o.shape[0])
        act_size = int(spec.action_spec.continuous_size)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_size,), dtype=np.float32
        )
        self._steps = 0

    @property
    def left_eye(self) -> np.ndarray | None:
        return self._left_eye

    @property
    def right_eye(self) -> np.ndarray | None:
        return self._right_eye

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
        return obs, {
            "seed": seed if seed is not None else self.config.seed,
            "behavior": self._behavior,
            "has_eyes": self._left_eye is not None and self._right_eye is not None,
        }

    def step(self, action):
        from mlagents_envs.base_env import ActionTuple

        action = np.asarray(action, dtype=np.float32).reshape(1, -1)
        self._env.set_actions(self._behavior, ActionTuple(continuous=action))
        self._env.step()
        self._steps += 1
        decision, terminal = self._env.get_steps(self._behavior)

        if len(terminal) > 0:
            obs = self._unpack_obs(terminal)
            reward = float(terminal.reward[0])
            return (
                obs,
                reward,
                True,
                False,
                {
                    "steps": self._steps,
                    "has_eyes": self._left_eye is not None and self._right_eye is not None,
                },
            )

        if len(decision) == 0:
            raise RuntimeError("Unity returned no decision or terminal steps")

        obs = self._unpack_obs(decision)
        reward = float(decision.reward[0])
        return (
            obs,
            reward,
            False,
            False,
            {
                "steps": self._steps,
                "has_eyes": self._left_eye is not None and self._right_eye is not None,
            },
        )

    def _current_obs(self) -> np.ndarray:
        decision, terminal = self._env.get_steps(self._behavior)
        if len(terminal) > 0:
            return self._unpack_obs(terminal)
        if len(decision) > 0:
            return self._unpack_obs(decision)
        # First reset can leave an empty queue until one Academy step.
        self._env.step()
        decision, terminal = self._env.get_steps(self._behavior)
        if len(terminal) > 0:
            return self._unpack_obs(terminal)
        if len(decision) > 0:
            return self._unpack_obs(decision)
        raise RuntimeError("No observations available after Unity reset")

    def _unpack_obs(self, step_result) -> np.ndarray:
        specs = self._env.behavior_specs[self._behavior].observation_specs
        per_agent = [np.asarray(o[0], dtype=np.float32) for o in step_result.obs]
        vector, left, right = unpack_unity_observations(
            per_agent,
            specs,
            vector_dim=DEFAULT_OBS_DIM,
            vision_size=int(self.config.vision_size),
        )
        self._left_eye = left
        self._right_eye = right
        return vector

    def close(self) -> None:
        if getattr(self, "_env", None) is not None:
            self._env.close()
            self._env = None  # type: ignore[assignment]

    def __enter__(self) -> UnityFlyEnv:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
