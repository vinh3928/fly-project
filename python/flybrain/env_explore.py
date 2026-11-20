"""Fake exploration env: high energy, sparse sugar, coverage metrics for Phase 8."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flybrain.curiosity.novelty import NoveltyConfig, NoveltyTracker
from flybrain.env import FakeFlyConfig, FakeFlyEnv
from flybrain.phase8 import (
    EXPLORE_ARENA_HALF,
    EXPLORE_EPISODE_STEPS,
    EXPLORE_INITIAL_ENERGY,
    GRID_CELL_SIZE,
)
from flybrain.training.rewards import HomeostaticRewardConfig, compute_total_reward
from flybrain.unity_env import DEFAULT_ACTION_DIM, DEFAULT_OBS_DIM

_OFF_INTERNAL = 176
_OFF_FOOT_TASTE = 102
_OFF_PROBOSCIS = 166


@dataclass
class FakeExplorationConfig(FakeFlyConfig):
    arena_half_extent: float = EXPLORE_ARENA_HALF
    max_steps: int = EXPLORE_EPISODE_STEPS
    observation_dim: int = DEFAULT_OBS_DIM
    action_dim: int = DEFAULT_ACTION_DIM
    initial_energy: float = EXPLORE_INITIAL_ENERGY
    energy_drain: float = 0.0015
    grid_cell_size: float = GRID_CELL_SIZE
    landmark_count: int = 5
    landmark_radius: float = 0.08
    sugar_consume_rate: float = 0.08


class FakeExplorationEnv(FakeFlyEnv):
    """Roam-focused arena: sated start, weak far sugar, scent landmarks for novelty."""

    def __init__(self, config: FakeExplorationConfig | None = None) -> None:
        self.explore_config = config or FakeExplorationConfig()
        super().__init__(self.explore_config)
        self._pos = np.zeros(2, dtype=np.float32)
        self._yaw = 0.0
        self._sugar = np.array([0.35, 0.35], dtype=np.float32)
        self._landmarks: list[np.ndarray] = []
        self._energy = float(self.explore_config.initial_energy)
        self._hunger = 1.0 - self._energy
        self._tracker = NoveltyTracker(
            NoveltyConfig(
                grid_cell_size=self.explore_config.grid_cell_size,
                arena_half_extent=self.explore_config.arena_half_extent,
            )
        )
        self._last_curiosity = 0.0
        self._sugar_consumed = 0.0

    def agent_position(self) -> np.ndarray:
        return self._pos.copy()

    def chem_vector(self) -> np.ndarray:
        """Privileged chem summary for novelty hashing (not XYZ)."""
        sugar_d = float(np.linalg.norm(self._pos - self._sugar))
        sugar_sig = float(np.exp(-sugar_d / 0.25))
        landmark_sigs = []
        for lm in self._landmarks:
            d = float(np.linalg.norm(self._pos - lm))
            landmark_sigs.append(float(np.exp(-d / 0.12)))
        while len(landmark_sigs) < 4:
            landmark_sigs.append(0.0)
        return np.asarray([sugar_sig, *landmark_sigs[:3]], dtype=np.float32)

    def chem_bearings(self) -> dict[str, float]:
        sugar_bearing = self._bearing_to(self._sugar)
        return {
            "sugar_bearing": sugar_bearing,
            "bitter_bearing": 0.0,
            "hunger": self._hunger,
            "energy": self._energy,
            "proboscis_contact": 0.0,
        }

    def _bearing_to(self, target: np.ndarray) -> float:
        delta = target - self._pos
        desired = float(np.arctan2(delta[0], delta[1]))
        err = (desired - self._yaw + np.pi) % (2 * np.pi) - np.pi
        return float(np.clip(err / np.pi, -1.0, 1.0))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        half = self.explore_config.arena_half_extent * 0.5
        self._pos = self._rng.uniform(-half * 0.3, half * 0.3, size=2).astype(np.float32)
        self._yaw = float(self._rng.uniform(-np.pi, np.pi))
        angle = float(self._rng.uniform(0, 2 * np.pi))
        radius = self.explore_config.arena_half_extent * 0.85
        self._sugar = np.asarray([radius * np.sin(angle), radius * np.cos(angle)], dtype=np.float32)
        self._landmarks = []
        for _ in range(self.explore_config.landmark_count):
            self._landmarks.append(
                self._rng.uniform(
                    -self.explore_config.arena_half_extent * 0.8,
                    self.explore_config.arena_half_extent * 0.8,
                    size=2,
                ).astype(np.float32)
            )
        self._energy = float(self.explore_config.initial_energy)
        self._hunger = float(np.clip(1.0 - self._energy, 0.0, 1.0))
        self._sugar_consumed = 0.0
        self._last_curiosity = 0.0
        self._tracker.reset()
        obs = self._write_obs()
        nov = self._tracker.observe(position=self._pos, chem_vector=self.chem_vector())
        info = {
            **info,
            "energy": self._energy,
            "hunger": self._hunger,
            "coverage": nov.coverage,
            "unique_cells": nov.unique_cells,
            "exploration_xp": nov.exploration_xp,
            "curiosity": 0.0,
            "distance_to_sugar": float(np.linalg.norm(self._pos - self._sugar)),
            "sugar_consumed": self._sugar_consumed,
            "has_food_xyz_in_obs": False,
            "_privileged_pos": self._pos.copy(),
        }
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        forward = float(np.clip(action[0] if action.size > 0 else 0.0, -1.0, 1.0))
        turn = float(np.clip(action[1] if action.size > 1 else 0.0, -1.0, 1.0))
        proboscis = float(np.clip(action[4] if action.size > 4 else 0.0, 0.0, 1.0))

        self._yaw += turn * 0.45
        direction = np.array([np.sin(self._yaw), np.cos(self._yaw)], dtype=np.float32)
        self._pos = self._pos + direction * (forward * 0.05)
        half = self.explore_config.arena_half_extent
        self._pos = np.clip(self._pos, -half, half)

        self._energy = float(np.clip(self._energy - self.explore_config.energy_drain, 0.0, 1.0))
        self._hunger = float(np.clip(1.0 - self._energy, 0.0, 1.0))

        sugar_dist = float(np.linalg.norm(self._pos - self._sugar))
        consumed = 0.0
        if proboscis > 0.5 and sugar_dist < 0.08:
            consumed = self.explore_config.sugar_consume_rate
            self._energy = float(np.clip(self._energy + consumed, 0.0, 1.0))
            self._hunger = float(np.clip(1.0 - self._energy, 0.0, 1.0))
            self._sugar_consumed += consumed

        events: list[str] = []
        for i, lm in enumerate(self._landmarks):
            if float(np.linalg.norm(self._pos - lm)) <= self.explore_config.landmark_radius:
                events.append(f"landmark_{i}")

        nov = self._tracker.observe(
            position=self._pos,
            chem_vector=self.chem_vector(),
            events=events,
        )
        self._last_curiosity = float(nov.novelty_score)
        self._step += 1
        obs = self._write_obs(proboscis_ext=proboscis, sugar_near=sugar_dist < 0.08)

        reward = compute_total_reward(
            curiosity=self._last_curiosity,
            energy_improvement=consumed,
            hydration_improvement=0.0,
            damage=0.0,
            motor_effort=float(np.mean(np.abs(action))),
            instability=0.0,
            config=HomeostaticRewardConfig(curiosity_weight=0.5),
            sugar_consume=consumed,
            bitter_contact=0.0,
            hunger=self._hunger,
        )

        terminated = self._energy <= 0.0
        truncated = self._step >= self.explore_config.max_steps
        info = {
            "energy": self._energy,
            "hunger": self._hunger,
            "coverage": nov.coverage,
            "unique_cells": nov.unique_cells,
            "exploration_xp": nov.exploration_xp,
            "streak": nov.streak,
            "curiosity": self._last_curiosity,
            "first_visit": nov.first_visit,
            "distance_to_sugar": sugar_dist,
            "sugar_consumed": self._sugar_consumed,
            "has_food_xyz_in_obs": False,
            "_privileged_pos": self._pos.copy(),
        }
        return obs, float(reward), terminated, truncated, info

    def explore_policy_action(self) -> np.ndarray:
        """Heuristic: head toward least-visited neighboring cell center."""
        size = self.explore_config.grid_cell_size
        best_dir = None
        best_score = -1e9
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)):
            candidate = self._pos + np.asarray([dx, dy], dtype=np.float32) * size
            half = self.explore_config.arena_half_extent
            if np.any(np.abs(candidate) > half):
                continue
            key = (
                int(np.floor(float(candidate[0]) / size)),
                int(np.floor(float(candidate[1]) / size)),
            )
            visits = self._tracker._cell_visits.get(key, 0)
            score = -float(visits) + 0.01 * float(self._rng.random())
            if score > best_score:
                best_score = score
                best_dir = candidate - self._pos
        if best_dir is None:
            return np.asarray([0.4, 0.2, 0.5, 0.6, 0.0], dtype=np.float32)
        desired = float(np.arctan2(best_dir[0], best_dir[1]))
        err = (desired - self._yaw + np.pi) % (2 * np.pi) - np.pi
        turn = float(np.clip(err / np.pi, -1.0, 1.0))
        forward = 0.65 if abs(err) < 0.8 else 0.25
        return np.asarray([forward, turn, 0.5, 0.7, 0.0], dtype=np.float32)

    def _write_obs(self, *, proboscis_ext: float = 0.0, sugar_near: bool = False) -> np.ndarray:
        obs = np.zeros(self.explore_config.observation_dim, dtype=np.float32)
        chem = self.chem_vector()
        obs[80:84] = chem
        if sugar_near:
            obs[_OFF_FOOT_TASTE : _OFF_FOOT_TASTE + 2] = np.array([0.8, 0.1], dtype=np.float32)
        obs[_OFF_PROBOSCIS] = float(proboscis_ext)
        # SensorManager internal layout: energy, hydration, fatigue, damage, temp, toxin, hunger, arousal
        obs[_OFF_INTERNAL] = self._energy
        obs[_OFF_INTERNAL + 1] = 1.0
        obs[_OFF_INTERNAL + 2] = 0.0
        obs[_OFF_INTERNAL + 3] = 0.0
        obs[_OFF_INTERNAL + 4] = 0.5
        obs[_OFF_INTERNAL + 5] = 0.0
        obs[_OFF_INTERNAL + 6] = self._hunger
        obs[_OFF_INTERNAL + 7] = 0.5
        self._obs = obs
        return obs.copy()
