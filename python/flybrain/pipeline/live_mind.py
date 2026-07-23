"""Live multimodal mind pipeline: observe → embed → STM → think → act."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from flybrain.affect.estimator import AffectEstimator
from flybrain.curiosity import CuriosityConfig, CuriosityDrive
from flybrain.env import FakeFlyEnv
from flybrain.memory.embedder import MindEmbedder
from flybrain.memory.pgvector import ShortTermMemory, open_short_term_memory
from flybrain.signals.builder import SignalBuilder
from flybrain.thinking.service import ThinkingConfig, ThinkingService
from flybrain.vision.recurrent_head import (
    RecurrentVisionHead,
    VisionNavCues,
    heuristic_cues_from_frames,
)
from flybrain.world_model.outcomes import LifeExperience
from flybrain.world_model.predictor import NextStateWorldModel


def load_mind_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "mind.yaml"
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@dataclass
class LiveMindConfig:
    fly_id: str = "fly_001"
    experiment_id: str = "experiment_mind_000001"
    max_steps: int = 200
    backend: str = "in_memory"
    dsn: str | None = None
    llm_mock: bool = True
    action_clamps: dict[str, float] | None = None
    use_vision_head: bool = True
    use_heuristic_if_untrained: bool = True
    vision_checkpoint: str | None = None
    world_model_checkpoint: str | None = None
    world_model_latent_dim: int = 256
    world_model_action_dim: int = 5
    world_model_hidden_dim: int = 512
    priorities: dict[str, Any] | None = None
    retrospective: dict[str, Any] | None = None
    enable_retrospective: bool = True
    curiosity: CuriosityConfig | None = None

    @classmethod
    def from_yaml(cls, data: dict[str, Any], visual_nav: dict[str, Any] | None = None) -> LiveMindConfig:
        db = data.get("database", {})
        thinking = data.get("thinking", {})
        live = data.get("live", {})
        llm = data.get("llm", {})
        wm = data.get("world_model", {}) or {}
        vn = visual_nav or {}
        vn_live = vn.get("live", {})
        clamps = dict(thinking.get("action_clamps") or {})
        if "wall_hazard_forward_scale" in vn_live:
            clamps["wall_hazard_forward_scale"] = vn_live["wall_hazard_forward_scale"]
        elif "wall_hazard_forward_scale" in vn.get("arena", {}):
            clamps.setdefault("wall_hazard_forward_scale", 0.2)
        clamps.setdefault("wall_hazard_forward_scale", float(vn_live.get("wall_hazard_forward_scale", 0.2)))
        return cls(
            fly_id=str(data.get("fly_id", "fly_001")),
            experiment_id=str(data.get("experiment_id", "experiment_mind_000001")),
            max_steps=int(live.get("max_steps", 200)),
            backend=str(db.get("backend", "in_memory")),
            dsn=db.get("dsn"),
            llm_mock=bool(llm.get("mock", True)),
            action_clamps=clamps,
            use_vision_head=bool(vn_live.get("use_vision_head", True)),
            use_heuristic_if_untrained=bool(vn_live.get("use_heuristic_if_untrained", True)),
            vision_checkpoint=(vn.get("train") or {}).get("checkpoint_path"),
            world_model_checkpoint=wm.get("checkpoint_path"),
            world_model_latent_dim=int(wm.get("latent_dim", 256)),
            world_model_action_dim=int(wm.get("action_dim", 5)),
            world_model_hidden_dim=int(wm.get("hidden_dim", 512)),
            priorities=dict(wm.get("priorities") or {}),
            retrospective=dict(wm.get("retrospective") or {}),
            enable_retrospective=bool(thinking.get("enable_retrospective", True)),
            curiosity=CuriosityConfig.from_dict(data.get("curiosity")),
        )


class LiveMindPipeline:
    """One-process orchestrator for Phase 4/5/6/7 live mind loop."""

    def __init__(
        self,
        env: Any,
        *,
        config: LiveMindConfig | None = None,
        memory: ShortTermMemory | None = None,
        vision_head: RecurrentVisionHead | None = None,
        world_model: NextStateWorldModel | None = None,
        experience: LifeExperience | None = None,
    ) -> None:
        self.env = env
        self.config = config or LiveMindConfig()
        self.signals = SignalBuilder()
        self.affect = AffectEstimator()
        self.embedder = MindEmbedder()
        self.memory = memory or open_short_term_memory(
            backend=self.config.backend,
            dsn=self.config.dsn,
            dim=self.embedder.total_dim,
        )
        self.world_model = world_model
        self._wm_trained = world_model is not None
        if self.world_model is None:
            self.world_model = self._load_world_model()
        # Use WM in thinking only when a trained checkpoint (or injected model) is available.
        wm_for_thinking = self.world_model if self._wm_trained else None
        self.experience = experience or LifeExperience.from_config(self.config.priorities)
        self.thinking = ThinkingService(
            self.memory,
            ThinkingConfig(
                fly_id=self.config.fly_id,
                action_clamps=self.config.action_clamps,
                llm_mock=self.config.llm_mock,
                priorities=self.config.priorities,
                retrospective=self.config.retrospective,
                enable_retrospective=self.config.enable_retrospective,
            ),
            experience=self.experience,
            world_model=wm_for_thinking,
        )
        self.vision_head = vision_head
        self._vision_trained = False
        if self.config.use_vision_head:
            self.vision_head = self.vision_head or RecurrentVisionHead()
            ckpt = self.config.vision_checkpoint
            if ckpt:
                path = Path(ckpt)
                if not path.is_absolute():
                    path = Path(__file__).resolve().parents[2] / path
                if path.exists():
                    state = torch.load(path, map_location="cpu", weights_only=True)
                    self.vision_head.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
                    self._vision_trained = True
            self.vision_head.eval()
            self.vision_head.reset_state(1)
        self.curiosity = CuriosityDrive(self.config.curiosity or CuriosityConfig())
        self._step = 0
        self._pending_action: np.ndarray | None = None

    def _load_world_model(self) -> NextStateWorldModel | None:
        model = NextStateWorldModel(
            latent_dim=self.config.world_model_latent_dim,
            action_dim=self.config.world_model_action_dim,
            hidden_dim=self.config.world_model_hidden_dim,
        )
        ckpt = self.config.world_model_checkpoint
        if not ckpt:
            return None
        path = Path(ckpt)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        if not path.exists():
            return None
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
        model.eval()
        self._wm_trained = True
        return model

    def close(self) -> None:
        self.memory.close()
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def _vision_cues(self, left: np.ndarray, right: np.ndarray) -> VisionNavCues | None:
        if not self.config.use_vision_head:
            return None
        if self._vision_trained and self.vision_head is not None:
            return self.vision_head.infer_cues(left, right)
        if self.config.use_heuristic_if_untrained:
            return heuristic_cues_from_frames(left, right)
        return None

    def process_observation(
        self,
        observation: np.ndarray,
        *,
        sim_time: float | None = None,
        left_eye: np.ndarray | None = None,
        right_eye: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if left_eye is None:
            left_eye = getattr(self.env, "left_eye", None)
        if right_eye is None:
            right_eye = getattr(self.env, "right_eye", None)

        sensory = self.signals.build(observation, left_eye=left_eye, right_eye=right_eye)
        cues = None
        if left_eye is not None and right_eye is not None:
            cues = self._vision_cues(left_eye, right_eye)
            if cues is not None:
                sensory.summary.update(cues.to_dict())

        # Phase 6: chem gradient bearings from food-search envs (not XYZ in obs).
        bearing_fn = getattr(self.env, "chem_bearings", None)
        if callable(bearing_fn):
            sensory.summary.update(bearing_fn())

        if sensory.summary.get("internal_all_zero") and self._step == 0:
            print(
                "warning: Unity obs internal block is all zeros "
                f"(energy={sensory.summary.get('energy')}). "
                "Check FlyAgent CollectObservations / SensorManager homeostasis binding.",
                flush=True,
            )
        affect = self.affect.estimate(sensory)
        embedding = self.embedder.embed(sensory, affect)
        signal_payload = sensory.to_jsonable()
        if cues is not None:
            signal_payload.update(cues.to_dict())
        for key in ("sugar_bearing", "bitter_bearing", "hunger", "energy", "proboscis_contact"):
            if key in sensory.summary:
                signal_payload[key] = sensory.summary[key]

        position = None
        pos_fn = getattr(self.env, "agent_position", None)
        if callable(pos_fn):
            position = np.asarray(pos_fn(), dtype=np.float32)
        chem_vec = None
        chem_fn = getattr(self.env, "chem_vector", None)
        if callable(chem_fn):
            chem_vec = np.asarray(chem_fn(), dtype=np.float32)

        curiosity_step = self.curiosity.step(
            embedding=embedding,
            action=self._pending_action,
            signals=signal_payload,
            affect=affect.to_jsonable(),
            position=position,
            chem_vector=chem_vec,
        )
        signal_payload["curiosity"] = curiosity_step.curiosity
        signal_payload["curiosity_drive"] = curiosity_step.explore_bias
        signal_payload["explore_bias"] = curiosity_step.explore_bias
        signal_payload["novelty"] = curiosity_step.novelty.novelty_score
        signal_payload["exploration_xp"] = curiosity_step.novelty.exploration_xp
        # Soft-bump affect exploration/novelty for thinking when gate allows.
        affect_payload = affect.to_jsonable()
        if curiosity_step.gate > 0.05:
            affect_payload["exploration"] = float(
                np.clip(affect_payload.get("exploration", 0.0) + 0.35 * curiosity_step.explore_bias, 0.0, 1.0)
            )
            affect_payload["novelty"] = float(
                np.clip(affect_payload.get("novelty", 0.0) + 0.4 * curiosity_step.novelty.novelty_score, 0.0, 1.0)
            )

        stm, thought, thought_rec = self.thinking.think_on_embedding(
            embedding=embedding,
            signals=signal_payload,
            affect=affect_payload,
            episode_id=self.config.experiment_id,
            step=self._step,
            sim_time=float(sim_time if sim_time is not None else self._step * 0.05),
            fly_id=self.config.fly_id,
        )
        action = self.curiosity.apply_action_bias(thought.action, curiosity_step.explore_bias)
        self._pending_action = action.copy()
        retro = self.thinking._last_retrospective
        info = {
            "stm_id": stm.id,
            "thought_id": thought_rec.id,
            "refined_thought": thought.refined_thought,
            "devil_critique": thought.devil_critique,
            "intent": thought.intent,
            "affect": affect_payload,
            "vision_cues": cues.to_dict() if cues else None,
            "maturity": self.experience.maturity(),
            "priority_weights": (thought.intent.get("future") or {}).get("priority_weights"),
            "retrospective_ran": bool(retro.ran) if retro else False,
            "labeled_outcomes": self.experience.labeled_outcomes,
            "curiosity": curiosity_step.curiosity,
            "curiosity_gated": curiosity_step.gated_curiosity,
            "curiosity_gate": curiosity_step.gate,
            "explore_bias": curiosity_step.explore_bias,
            "coverage": curiosity_step.novelty.coverage,
            "unique_cells": curiosity_step.novelty.unique_cells,
            "exploration_xp": curiosity_step.novelty.exploration_xp,
            "novelty_first_visit": curiosity_step.novelty.first_visit,
        }
        return action, info

    def run(self, *, max_steps: int | None = None) -> dict[str, Any]:
        limit = max_steps if max_steps is not None else self.config.max_steps
        self.curiosity.reset()
        self._pending_action = None
        obs, reset_info = self.env.reset()
        last_info: dict[str, Any] = {}
        terminated = False
        truncated = False
        start_dist = 0.0
        if isinstance(reset_info, dict):
            start_dist = float(
                reset_info.get(
                    "distance_to_sugar",
                    reset_info.get("distance_to_target", 0.0),
                )
            )
        for _ in range(limit):
            action, info = self.process_observation(obs)
            # Blend food-search heuristic when available (Phase 6 live improvement loop).
            food_policy = getattr(self.env, "food_policy_action", None)
            if callable(food_policy):
                food_action = food_policy()
                action = action.copy()
                action[0] = 0.4 * float(action[0]) + 0.6 * float(food_action[0])
                action[1] = 0.35 * float(action[1]) + 0.65 * float(food_action[1])
                action[4] = 0.3 * float(action[4]) + 0.7 * float(food_action[4])
            # Phase 8: optional explore heuristic blend when env provides it and gate is open.
            explore_policy = getattr(self.env, "explore_policy_action", None)
            if callable(explore_policy) and float(info.get("curiosity_gate", 0.0)) > 0.2:
                explore_action = explore_policy()
                blend = 0.45 * float(info.get("explore_bias", 0.0))
                action = action.copy()
                action[0] = (1.0 - blend) * float(action[0]) + blend * float(explore_action[0])
                action[1] = (1.0 - blend) * float(action[1]) + blend * float(explore_action[1])
                action[3] = (1.0 - 0.5 * blend) * float(action[3]) + 0.5 * blend * float(explore_action[3])
                self._pending_action = action.copy()
            last_info = info
            obs, _reward, terminated, truncated, step_info = self.env.step(action)
            if isinstance(step_info, dict):
                last_info = {**last_info, **step_info}
            self._step += 1
            if terminated or truncated:
                break
        self.experience.end_episode()
        end_dist = float(
            last_info.get(
                "distance_to_sugar",
                last_info.get("distance_to_target", start_dist),
            )
        )
        curiosity_summary = self.curiosity.summary()
        return {
            "steps": self._step,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "last": last_info,
            "start_distance": start_dist,
            "end_distance": end_dist,
            "wall_hits": int(last_info.get("wall_hits", 0)),
            "success": bool(last_info.get("success", False)),
            "sugar_consumed": float(last_info.get("sugar_consumed", 0.0)),
            "bitter_contacts": int(last_info.get("bitter_contacts", 0)),
            "maturity": self.experience.maturity(),
            "labeled_outcomes": self.experience.labeled_outcomes,
            "episodes": self.experience.episodes,
            "coverage": float(last_info.get("coverage", curiosity_summary.get("coverage", 0.0))),
            "unique_cells": int(last_info.get("unique_cells", curiosity_summary.get("unique_cells", 0))),
            "exploration_xp": float(
                last_info.get("exploration_xp", curiosity_summary.get("exploration_xp", 0.0))
            ),
            "curiosity_reward_mean": float(curiosity_summary.get("curiosity_reward_mean", 0.0)),
            "curiosity": curiosity_summary,
        }


def make_live_env(
    *,
    fake: bool = True,
    fake_visual_nav: bool = False,
    fake_food_search: bool = False,
    fake_explore: bool = False,
    editor: bool = False,
    build: str | None = None,
    port: int | None = None,
):
    if fake_explore:
        from flybrain.env_explore import FakeExplorationEnv

        return FakeExplorationEnv()
    if fake_food_search:
        from flybrain.env_food_search import FakeFoodSearchEnv

        return FakeFoodSearchEnv()
    if fake_visual_nav:
        from flybrain.env_visual_nav import FakeVisualNavEnv

        return FakeVisualNavEnv()
    if fake:
        return FakeFlyEnv()
    from flybrain.unity_env import UnityFlyConfig, UnityFlyEnv

    # ML-Agents: Editor trainer port is 5004; standalone executables use 5005+.
    if port is None:
        port = 5004 if editor else 5005
    file_name = None if editor else build
    return UnityFlyEnv(
        UnityFlyConfig(
            file_name=file_name,
            base_port=port,
            no_graphics=not editor and file_name is not None,
            editor_connect=bool(editor),
            timeout_wait=60 if editor else 120,
        )
    )
