"""Live curiosity drive: novelty + prediction error, gated by homeostasis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from flybrain.curiosity.module import CuriosityModule
from flybrain.curiosity.novelty import NoveltyConfig, NoveltyObservation, NoveltyTracker
from flybrain.phase8 import (
    CURIOSITY_REWARD_CAP,
    ENERGY_MIN_FOR_EXPLORE,
    EXPLORE_ACTION_BOOST,
    EXPLORE_CANDIDATE_BONUS,
    HUNGER_GATE,
    NOVELTY_WEIGHT,
    PREDICTION_WEIGHT,
    SURVIVAL_RISK_GATE,
)


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


@dataclass
class CuriosityConfig:
    enabled: bool = True
    curiosity_weight: float = 0.25
    novelty_weight: float = NOVELTY_WEIGHT
    prediction_weight: float = PREDICTION_WEIGHT
    reward_cap: float = CURIOSITY_REWARD_CAP
    survival_risk_max: float = SURVIVAL_RISK_GATE
    hunger_max: float = HUNGER_GATE
    energy_min: float = ENERGY_MIN_FOR_EXPLORE
    explore_action_boost: float = EXPLORE_ACTION_BOOST
    explore_candidate_bonus: float = EXPLORE_CANDIDATE_BONUS
    latent_dim: int = 256
    action_dim: int = 5
    novelty: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CuriosityConfig:
        raw = dict(data or {})
        novelty = dict(raw.pop("novelty", {}) or {})
        fields = set(cls.__dataclass_fields__)
        kwargs = {k: v for k, v in raw.items() if k in fields}
        return cls(novelty=novelty, **kwargs)


def curiosity_gate(
    *,
    signals: dict[str, Any],
    affect: dict[str, Any],
    config: CuriosityConfig | None = None,
) -> float:
    """Return [0, 1] allowance for curiosity. 0 = fully suppressed."""
    cfg = config or CuriosityConfig()
    if not cfg.enabled:
        return 0.0
    survival = float(affect.get("survival_risk", 0.0))
    hunger = float(signals.get("hunger", affect.get("sad", 0.0)))
    energy = float(signals.get("energy", 1.0 - hunger))
    if survival >= cfg.survival_risk_max:
        return 0.0
    if hunger >= cfg.hunger_max:
        return 0.0
    if energy < cfg.energy_min:
        return 0.0
    # Smooth fade near thresholds
    survival_ok = 1.0 - survival / max(1e-6, cfg.survival_risk_max)
    hunger_ok = 1.0 - hunger / max(1e-6, cfg.hunger_max)
    energy_ok = (energy - cfg.energy_min) / max(1e-6, 1.0 - cfg.energy_min)
    return _clamp01(min(survival_ok, hunger_ok, energy_ok))


@dataclass
class CuriosityStepResult:
    curiosity: float
    gated_curiosity: float
    gate: float
    prediction_error: float
    novelty: NoveltyObservation
    explore_bias: float


class CuriosityDrive:
    """Combines NoveltyTracker + CuriosityModule for the live mind loop."""

    def __init__(
        self,
        config: CuriosityConfig | None = None,
        *,
        module: CuriosityModule | None = None,
        tracker: NoveltyTracker | None = None,
    ) -> None:
        self.config = config or CuriosityConfig()
        novelty_cfg = NoveltyConfig(
            **{
                k: v
                for k, v in (self.config.novelty or {}).items()
                if k in NoveltyConfig.__dataclass_fields__
            }
        )
        self.tracker = tracker or NoveltyTracker(novelty_cfg)
        self.module = module or CuriosityModule(
            latent_dim=self.config.latent_dim,
            action_dim=self.config.action_dim,
            reward_cap=self.config.reward_cap,
        )
        self._prev_latent: np.ndarray | None = None
        self._prev_action: np.ndarray | None = None
        self._reward_sum: float = 0.0
        self._steps: int = 0

    def reset(self) -> None:
        self.tracker.reset()
        self._prev_latent = None
        self._prev_action = None
        self._reward_sum = 0.0
        self._steps = 0
        self.module.error_ema.fill_(0.0)

    def _latent(self, embedding: np.ndarray) -> np.ndarray:
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        out = np.zeros(self.config.latent_dim, dtype=np.float32)
        n = min(self.config.latent_dim, vec.shape[0])
        out[:n] = vec[:n]
        return out

    def step(
        self,
        *,
        embedding: np.ndarray,
        action: np.ndarray | None = None,
        signals: dict[str, Any],
        affect: dict[str, Any],
        position: np.ndarray | None = None,
        chem_vector: np.ndarray | None = None,
        events: list[str] | None = None,
    ) -> CuriosityStepResult:
        latent = self._latent(embedding)
        prediction_error = 0.0
        pred_reward = 0.0
        if self._prev_latent is not None and self._prev_action is not None:
            with torch.no_grad():
                z = torch.as_tensor(self._prev_latent, dtype=torch.float32).unsqueeze(0)
                a = torch.as_tensor(self._prev_action, dtype=torch.float32).unsqueeze(0)
                z_next = torch.as_tensor(latent, dtype=torch.float32).unsqueeze(0)
                reward_t, loss_t = self.module.intrinsic_reward(z, a, z_next)
                pred_reward = float(reward_t.mean().item())
                prediction_error = float(loss_t.item())

        novelty = self.tracker.observe(
            position=position,
            embedding=embedding,
            chem_vector=chem_vector,
            events=events,
        )
        raw = (
            self.config.novelty_weight * novelty.novelty_score
            + self.config.prediction_weight * pred_reward
        )
        curiosity = _clamp01(min(self.config.reward_cap, raw))
        gate = curiosity_gate(signals=signals, affect=affect, config=self.config)
        gated = curiosity * gate
        explore_bias = gate * (
            0.55 * novelty.novelty_score + 0.45 * gated + 0.05 * min(1.0, novelty.streak / 5.0)
        )
        explore_bias = _clamp01(explore_bias)

        if action is not None:
            act = np.asarray(action, dtype=np.float32).reshape(-1)
            padded = np.zeros(self.config.action_dim, dtype=np.float32)
            padded[: min(self.config.action_dim, act.size)] = act[: self.config.action_dim]
            self._prev_action = padded
        self._prev_latent = latent
        self._reward_sum += gated
        self._steps += 1

        return CuriosityStepResult(
            curiosity=curiosity,
            gated_curiosity=gated,
            gate=gate,
            prediction_error=prediction_error,
            novelty=novelty,
            explore_bias=explore_bias,
        )

    def apply_action_bias(self, action: np.ndarray, explore_bias: float) -> np.ndarray:
        """Nudge high-level action toward roaming when curiosity is allowed."""
        if explore_bias <= 1e-6 or not self.config.enabled:
            return action
        out = np.asarray(action, dtype=np.float32).copy()
        boost = float(self.config.explore_action_boost) * float(explore_bias)
        out[0] = float(np.clip(out[0] + 0.55 * boost, -1.0, 1.0))
        # Prefer a gentle turn so coverage grows instead of straight-line camping.
        turn_nudge = 0.35 * boost * (1.0 if abs(float(out[1])) < 0.15 else np.sign(out[1]))
        out[1] = float(np.clip(out[1] + turn_nudge, -1.0, 1.0))
        out[3] = float(np.clip(out[3] + 0.25 * boost, 0.0, 1.0))
        return out

    def summary(self) -> dict[str, Any]:
        mean = self._reward_sum / max(1, self._steps)
        return {
            **self.tracker.summary(),
            "curiosity_reward_mean": float(mean),
            "curiosity_reward_sum": float(self._reward_sum),
            "curiosity_steps": int(self._steps),
            "enabled": bool(self.config.enabled),
        }
