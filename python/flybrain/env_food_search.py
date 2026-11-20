"""Fake food-search env: sugar/bitter patches, homeostasis, no food XYZ in obs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flybrain.env import FakeFlyConfig, FakeFlyEnv
from flybrain.phase6 import (
    BITTER_TOXIN_RATE,
    ENERGY_DRAIN_PER_STEP,
    EVAL_EPISODE_STEPS,
    PATCH_RADIUS,
    SUGAR_CONSUME_RATE,
)
from flybrain.signals.builder import PROBOSCIS_CONTACT_INDEX, PROBOSCIS_EXTENSION_INDEX
from flybrain.unity_env import DEFAULT_ACTION_DIM, DEFAULT_OBS_DIM

# Unity SensorManager layout offsets into the 197-vector.
_OFF_FOOT_TASTE = 72 + 30  # 102
_OFF_PROBOSCIS = 72 + 30 + 48 + 16  # 166
_OFF_INTERNAL = _OFF_PROBOSCIS + 10  # 176


@dataclass
class FakeFoodSearchConfig(FakeFlyConfig):
    arena_half_extent: float = 0.45
    patch_radius: float = PATCH_RADIUS
    max_steps: int = EVAL_EPISODE_STEPS
    observation_dim: int = DEFAULT_OBS_DIM
    action_dim: int = DEFAULT_ACTION_DIM
    energy_drain: float = ENERGY_DRAIN_PER_STEP
    sugar_consume_rate: float = SUGAR_CONSUME_RATE
    bitter_toxin_rate: float = BITTER_TOXIN_RATE
    initial_energy: float = 0.35


class FakeFoodSearchEnv(FakeFlyEnv):
    """Arena with sugar + bitter patches; chem/internal cues only in vector obs."""

    def __init__(self, config: FakeFoodSearchConfig | None = None) -> None:
        self.food_config = config or FakeFoodSearchConfig()
        super().__init__(self.food_config)
        self._pos = np.zeros(2, dtype=np.float32)
        self._yaw = 0.0
        self._sugar = np.array([0.25, 0.2], dtype=np.float32)
        self._bitter = np.array([-0.25, -0.2], dtype=np.float32)
        self._energy = float(self.food_config.initial_energy)
        self._hunger = 1.0 - self._energy
        self._damage = 0.0
        self._toxin = 0.0
        self._sugar_consumed = 0.0
        self._bitter_contacts = 0
        self._start_sugar_dist = 0.0
        self._sugar_dist = 0.0
        self._bitter_dist = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        half = self.food_config.arena_half_extent * 0.55
        self._pos = self._rng.uniform(-half, half, size=2).astype(np.float32)
        self._yaw = float(self._rng.uniform(-np.pi, np.pi))
        self._sugar = self._rng.uniform(-half, half, size=2).astype(np.float32)
        self._bitter = self._rng.uniform(-half, half, size=2).astype(np.float32)
        # Keep patches separated
        for _ in range(24):
            if np.linalg.norm(self._sugar - self._bitter) > 0.2:
                break
            self._bitter = self._rng.uniform(-half, half, size=2).astype(np.float32)
        self._energy = float(self.food_config.initial_energy)
        self._hunger = 1.0 - self._energy
        self._damage = 0.0
        self._toxin = 0.0
        self._sugar_consumed = 0.0
        self._bitter_contacts = 0
        self._refresh_distances()
        self._start_sugar_dist = self._sugar_dist
        obs = self._write_obs()
        info = {
            **info,
            "energy": self._energy,
            "hunger": self._hunger,
            "distance_to_sugar": self._sugar_dist,
            "distance_to_bitter": self._bitter_dist,
            "sugar_consumed": self._sugar_consumed,
            "bitter_contacts": self._bitter_contacts,
            "has_food_xyz_in_obs": False,
            "agent_obs_dim": int(obs.shape[0]),
            "_privileged_sugar": self._sugar.copy(),
            "_privileged_bitter": self._bitter.copy(),
        }
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        forward = float(np.clip(action[0] if action.size > 0 else 0.0, -1.0, 1.0))
        turn = float(np.clip(action[1] if action.size > 1 else 0.0, -1.0, 1.0))
        proboscis = float(np.clip(action[4] if action.size > 4 else 0.0, 0.0, 1.0))

        self._yaw += turn * 0.4
        direction = np.array([np.sin(self._yaw), np.cos(self._yaw)], dtype=np.float32)
        self._pos = self._pos + direction * (forward * 0.045)
        half = self.food_config.arena_half_extent
        self._pos = np.clip(self._pos, -half, half)

        self._energy = float(np.clip(self._energy - self.food_config.energy_drain, 0.0, 1.0))
        self._hunger = float(np.clip(1.0 - self._energy, 0.0, 1.0))

        self._refresh_distances()
        sugar_near = self._sugar_dist <= self.food_config.patch_radius
        bitter_near = self._bitter_dist <= self.food_config.patch_radius
        consumed = 0.0
        if proboscis > 0.5 and sugar_near:
            gain = self.food_config.sugar_consume_rate
            self._energy = float(np.clip(self._energy + gain, 0.0, 1.0))
            self._hunger = float(np.clip(1.0 - self._energy, 0.0, 1.0))
            self._sugar_consumed += gain
            consumed = gain
        if proboscis > 0.5 and bitter_near:
            self._bitter_contacts += 1
            self._toxin = float(np.clip(self._toxin + self.food_config.bitter_toxin_rate, 0.0, 1.0))
            self._damage = float(
                np.clip(self._damage + self.food_config.bitter_toxin_rate * 0.5, 0.0, 1.0)
            )

        self._step += 1
        obs = self._write_obs(
            proboscis_ext=proboscis, sugar_near=sugar_near, bitter_near=bitter_near
        )
        # Reward uses distance change from previous step stored on self.
        approach = getattr(self, "_prev_sugar_dist", self._sugar_dist) - self._sugar_dist
        self._prev_sugar_dist = self._sugar_dist

        from flybrain.training.rewards import HomeostaticRewardConfig, compute_total_reward

        reward = compute_total_reward(
            curiosity=0.0,
            energy_improvement=consumed,
            hydration_improvement=0.0,
            damage=self._damage,
            motor_effort=float(np.mean(np.abs(action))),
            instability=0.0,
            config=HomeostaticRewardConfig(),
            sugar_consume=consumed,
            bitter_contact=1.0 if (proboscis > 0.5 and bitter_near) else 0.0,
            hunger=self._hunger,
        )
        reward += 2.0 * approach

        terminated = self._energy <= 0.0
        truncated = self._step >= self.food_config.max_steps
        info = {
            "energy": self._energy,
            "hunger": self._hunger,
            "damage": self._damage,
            "toxin": self._toxin,
            "distance_to_sugar": self._sugar_dist,
            "distance_to_bitter": self._bitter_dist,
            "start_sugar_distance": self._start_sugar_dist,
            "sugar_consumed": self._sugar_consumed,
            "bitter_contacts": self._bitter_contacts,
            "proboscis": proboscis,
            "sugar_near": sugar_near,
            "bitter_near": bitter_near,
            "has_food_xyz_in_obs": False,
            "agent_obs_dim": int(obs.shape[0]),
            "_privileged_sugar": self._sugar.copy(),
            "_privileged_bitter": self._bitter.copy(),
            "_privileged_sugar_bearing": self._bearing_to(self._sugar),
            "_privileged_bitter_bearing": self._bearing_to(self._bitter),
        }
        return obs, float(reward), terminated, truncated, info

    def food_policy_action(self, *, hungry: bool | None = None) -> np.ndarray:
        """Greedy seek/avoid from chem cues already written into the last obs path."""
        if hungry is None:
            hungry = self._hunger >= 0.45
        sugar_bearing = self._bearing_to(self._sugar)
        bitter_bearing = self._bearing_to(self._bitter)
        sugar_near = self._sugar_dist <= self.food_config.patch_radius
        bitter_near = self._bitter_dist <= self.food_config.patch_radius

        if not hungry:
            # Sated: mild wander, retract, avoid bitter if close
            turn = float(np.clip(-0.6 * bitter_bearing if bitter_near else 0.05, -1.0, 1.0))
            forward = 0.15 if not bitter_near else 0.05
            proboscis = 0.0
        else:
            # Hungry: approach sugar, veer from bitter, extend when on sugar
            turn = float(np.clip(0.85 * sugar_bearing - 0.55 * bitter_bearing, -1.0, 1.0))
            forward = 0.7 if not bitter_near else 0.25
            if sugar_near:
                forward = 0.2
            if bitter_near and self._bitter_dist < self._sugar_dist:
                turn = float(np.clip(-np.sign(bitter_bearing + 1e-6) * 0.9, -1.0, 1.0))
                forward = 0.35
            proboscis = (
                1.0
                if sugar_near
                else (0.2 if self._sugar_dist < self.food_config.patch_radius * 1.5 else 0.0)
            )

        return np.asarray([forward, turn, 0.5, 0.55, proboscis], dtype=np.float32)

    def chem_bearings(self) -> dict[str, float]:
        """Relative bearings for mind composer (not written into obs vector)."""
        return {
            "sugar_bearing": self._bearing_to(self._sugar),
            "bitter_bearing": self._bearing_to(self._bitter),
            "hunger": float(self._hunger),
            "energy": float(self._energy),
            "proboscis_contact": (
                1.0
                if (min(self._sugar_dist, self._bitter_dist) <= self.food_config.patch_radius)
                else 0.0
            ),
        }

    def _bearing_to(self, target: np.ndarray) -> float:
        rel = target - self._pos
        world_angle = float(np.arctan2(rel[0], rel[1]))
        bearing_rad = world_angle - self._yaw
        return float(
            np.clip(np.arctan2(np.sin(bearing_rad), np.cos(bearing_rad)) / np.pi, -1.0, 1.0)
        )

    def _refresh_distances(self) -> None:
        self._sugar_dist = float(np.linalg.norm(self._sugar - self._pos))
        self._bitter_dist = float(np.linalg.norm(self._bitter - self._pos))

    def _write_obs(
        self,
        *,
        proboscis_ext: float = 0.0,
        sugar_near: bool = False,
        bitter_near: bool = False,
    ) -> np.ndarray:
        obs = np.zeros(DEFAULT_OBS_DIM, dtype=np.float32)
        # Diffuse chem by distance (taste without giving XYZ)
        sugar_strength = float(
            np.clip(1.0 - self._sugar_dist / (self.food_config.arena_half_extent * 1.6), 0.0, 1.0)
        )
        bitter_strength = float(
            np.clip(1.0 - self._bitter_dist / (self.food_config.arena_half_extent * 1.6), 0.0, 1.0)
        )
        if sugar_near:
            sugar_strength = max(sugar_strength, 0.85)
        if bitter_near:
            bitter_strength = max(bitter_strength, 0.85)

        # First foot taste channels: sugar=0, bitter=1 (×6 feet)
        for i in range(6):
            base = _OFF_FOOT_TASTE + i * 8
            obs[base] = sugar_strength * (0.4 if not sugar_near else 1.0)
            obs[base + 1] = bitter_strength * (0.4 if not bitter_near else 1.0)

        # Proboscis: extension, contact, sugar, bitter... (Unity layout)
        obs[_OFF_PROBOSCIS + PROBOSCIS_CONTACT_INDEX] = (
            1.0 if (sugar_near or bitter_near) and proboscis_ext > 0.5 else 0.0
        )
        obs[_OFF_PROBOSCIS + PROBOSCIS_EXTENSION_INDEX] = proboscis_ext
        obs[_OFF_PROBOSCIS + 2] = sugar_strength if sugar_near else sugar_strength * 0.2
        obs[_OFF_PROBOSCIS + 3] = bitter_strength if bitter_near else bitter_strength * 0.2

        obs[_OFF_INTERNAL] = self._energy
        obs[_OFF_INTERNAL + 1] = 0.8  # hydration
        obs[_OFF_INTERNAL + 3] = self._damage
        obs[_OFF_INTERNAL + 5] = self._toxin
        obs[_OFF_INTERNAL + 6] = self._hunger

        self._obs = obs
        return obs.copy()
