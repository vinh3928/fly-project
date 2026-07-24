"""Architecture B motor layer: mind goals → motor actions (every physics/decision tick)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from flybrain.phase9 import MOTOR_ACTION_DIM, MOTOR_GOAL_DIM


def clip_goal(goal: np.ndarray) -> np.ndarray:
    g = np.asarray(goal, dtype=np.float32).reshape(-1)
    out = np.zeros(MOTOR_GOAL_DIM, dtype=np.float32)
    n = min(MOTOR_GOAL_DIM, g.size)
    out[:n] = g[:n]
    out[0] = float(np.clip(out[0], -1.0, 1.0))
    out[1] = float(np.clip(out[1], -1.0, 1.0))
    out[2] = float(np.clip(out[2], 0.0, 1.0))
    out[3] = float(np.clip(out[3], 0.0, 1.0))
    out[4] = float(np.clip(out[4], 0.0, 1.0))
    return out


def ensure_locomotion_floor(
    goal: np.ndarray,
    *,
    min_forward: float = 0.4,
    stop_forward_eps: float = 0.08,
) -> np.ndarray:
    """Keep non-stop goals walking so mind+motor produce visible gait."""
    g = clip_goal(goal)
    # Explicit stop / near-zero command: leave alone.
    if abs(float(g[0])) <= stop_forward_eps and abs(float(g[1])) <= stop_forward_eps and float(g[4]) < 0.4:
        return g
    # Feeding with proboscis: allow slow approach but keep some crawl.
    if float(g[4]) >= 0.5:
        g[0] = float(np.clip(max(float(g[0]), min_forward * 0.4), -1.0, 1.0))
        return g
    # Turning or roaming: enforce a forward floor so the fly does not spin in place.
    g[0] = float(np.clip(max(float(g[0]), min_forward), -1.0, 1.0))
    return g


def goal_from_intent(intent: dict[str, Any] | None, action: np.ndarray | None = None) -> np.ndarray:
    """Build a motor goal from thinking intent / high-level action.

    Architecture B prefers the Layer-1/2 discrete plan (`future.chosen_action`) so
    ``plan=turn_left`` actually drives a left turn instead of a timid composer blend.
    """
    if intent:
        future = intent.get("future") or {}
        chosen = future.get("chosen_action")
        if chosen is not None:
            return clip_goal(np.asarray(chosen, dtype=np.float32))
    if action is not None:
        return clip_goal(action)
    if not intent:
        return clip_goal(np.zeros(MOTOR_GOAL_DIM, dtype=np.float32))
    if "action" in intent and intent["action"] is not None:
        return clip_goal(np.asarray(intent["action"], dtype=np.float32))
    return clip_goal(
        np.asarray(
            [
                float(intent.get("forward", 0.0)),
                float(intent.get("turn", 0.0)),
                float(intent.get("height", 0.5)),
                float(intent.get("frequency", 0.5)),
                float(intent.get("proboscis", 0.0)),
            ],
            dtype=np.float32,
        )
    )


@dataclass
class MotorLayerConfig:
    mode: str = "heuristic"  # heuristic | learned
    smoothing: float = 0.55
    max_delta: float = 0.75
    min_forward: float = 0.4
    checkpoint_path: str | None = None
    obs_dim: int = 197
    goal_dim: int = MOTOR_GOAL_DIM
    action_dim: int = MOTOR_ACTION_DIM

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> MotorLayerConfig:
        raw = dict(data or {})
        fields = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in fields})


class HeuristicMotorTracker:
    """Rate-limited smoother that tracks mind goals into motor commands."""

    def __init__(self, *, smoothing: float = 0.35, max_delta: float = 0.45) -> None:
        self.smoothing = float(np.clip(smoothing, 0.05, 1.0))
        self.max_delta = float(max(0.05, max_delta))
        self._command = np.asarray([0.0, 0.0, 0.5, 0.5, 0.0], dtype=np.float32)

    def reset(self, goal: np.ndarray | None = None) -> None:
        self._command = (
            clip_goal(goal)
            if goal is not None
            else np.asarray([0.0, 0.0, 0.5, 0.5, 0.0], dtype=np.float32)
        )

    @property
    def command(self) -> np.ndarray:
        return self._command.copy()

    def act(self, goal: np.ndarray, *, observation: np.ndarray | None = None) -> np.ndarray:
        _ = observation
        target = clip_goal(goal)
        blended = (1.0 - self.smoothing) * self._command + self.smoothing * target
        delta = np.clip(blended - self._command, -self.max_delta, self.max_delta)
        self._command = clip_goal(self._command + delta)
        return self._command.copy()


class MotorPolicyNet(nn.Module):
    """Small MLP: concat(obs_summary, goal) → motor action."""

    def __init__(
        self,
        obs_dim: int = 197,
        goal_dim: int = 5,
        action_dim: int = 5,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        # Use a compact obs projection (first N + mean pool) for speed/stability.
        self.obs_proj = nn.Linear(min(64, obs_dim), 64)
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.net = nn.Sequential(
            nn.Linear(64 + goal_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),
        )

    def _encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [B, obs_dim]
        take = min(64, obs.shape[-1])
        head = obs[..., :take]
        if take < 64:
            pad = torch.zeros(*obs.shape[:-1], 64 - take, device=obs.device, dtype=obs.dtype)
            head = torch.cat([head, pad], dim=-1)
        return torch.tanh(self.obs_proj(head))

    def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        z = self._encode_obs(obs)
        # Map tanh outs: dims 2..4 should be [0,1]; shift after.
        raw = self.net(torch.cat([z, goal], dim=-1))
        forward = raw[..., 0:1]
        turn = raw[..., 1:2]
        height = 0.5 * (raw[..., 2:3] + 1.0)
        frequency = 0.5 * (raw[..., 3:4] + 1.0)
        proboscis = 0.5 * (raw[..., 4:5] + 1.0)
        return torch.cat([forward, turn, height, frequency, proboscis], dim=-1)


class LearnedMotorTracker:
    def __init__(self, policy: MotorPolicyNet, *, device: str = "cpu") -> None:
        self.policy = policy.to(device)
        self.policy.eval()
        self.device = device
        self._command = np.asarray([0.0, 0.0, 0.5, 0.5, 0.0], dtype=np.float32)

    def reset(self, goal: np.ndarray | None = None) -> None:
        self._command = clip_goal(goal) if goal is not None else self._command

    @property
    def command(self) -> np.ndarray:
        return self._command.copy()

    def act(self, goal: np.ndarray, *, observation: np.ndarray | None = None) -> np.ndarray:
        obs = np.zeros(self.policy.obs_dim, dtype=np.float32)
        if observation is not None:
            flat = np.asarray(observation, dtype=np.float32).reshape(-1)
            obs[: min(obs.size, flat.size)] = flat[: obs.size]
        g = clip_goal(goal)
        with torch.no_grad():
            o_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            g_t = torch.as_tensor(g, dtype=torch.float32, device=self.device).unsqueeze(0)
            a = self.policy(o_t, g_t).squeeze(0).cpu().numpy().astype(np.float32)
        self._command = clip_goal(a)
        return self._command.copy()


class MotorLayer:
    """Facade used by live mind: hold last mind goal, emit motor action each tick."""

    def __init__(self, config: MotorLayerConfig | None = None) -> None:
        self.config = config or MotorLayerConfig()
        self.goal = np.asarray([0.0, 0.0, 0.5, 0.5, 0.0], dtype=np.float32)
        self._tracker: HeuristicMotorTracker | LearnedMotorTracker
        if self.config.mode == "learned":
            policy = MotorPolicyNet(
                obs_dim=self.config.obs_dim,
                goal_dim=self.config.goal_dim,
                action_dim=self.config.action_dim,
            )
            ckpt = self.config.checkpoint_path
            if ckpt:
                path = Path(ckpt)
                if path.exists():
                    state = torch.load(path, map_location="cpu", weights_only=True)
                    if isinstance(state, dict) and "model" in state:
                        payload = state["model"]
                    else:
                        payload = state
                    policy.load_state_dict(payload)
            self._tracker = LearnedMotorTracker(policy)
        else:
            self._tracker = HeuristicMotorTracker(
                smoothing=self.config.smoothing,
                max_delta=self.config.max_delta,
            )

    def reset(self, goal: np.ndarray | None = None) -> None:
        if goal is not None:
            self.goal = ensure_locomotion_floor(goal, min_forward=self.config.min_forward)
        self._tracker.reset(self.goal)

    def set_goal(self, goal: np.ndarray) -> None:
        self.goal = ensure_locomotion_floor(goal, min_forward=self.config.min_forward)

    def set_goal_from_intent(
        self, intent: dict[str, Any] | None, action: np.ndarray | None = None
    ) -> np.ndarray:
        self.set_goal(goal_from_intent(intent, action))
        return self.goal.copy()

    def act(self, observation: np.ndarray | None = None) -> np.ndarray:
        return self._tracker.act(self.goal, observation=observation)

    def tracking_error(self) -> float:
        return float(np.mean(np.abs(self._tracker.command - self.goal)))
