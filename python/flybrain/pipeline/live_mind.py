"""Live multimodal mind pipeline: observe → embed → STM → think → act."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

import numpy as np
import torch
import yaml

from flybrain.affect.estimator import AffectEstimator
from flybrain.curiosity import CuriosityConfig, CuriosityDrive
from flybrain.env import FakeFlyEnv
from flybrain.memory.embedder import MindEmbedder
from flybrain.memory.pgvector import ShortTermMemory, open_short_term_memory
from flybrain.motor import MotorLayer, MotorLayerConfig, goal_from_intent
from flybrain.phase9 import DEFAULT_MIND_INTERVAL
from flybrain.signals.builder import SignalBuilder
from flybrain.thinking.service import ThinkingConfig, ThinkingService
from flybrain.vision.recurrent_head import (
    RecurrentVisionHead,
    VisionNavCues,
    heuristic_cues_from_frames,
)
from flybrain.world_model.outcomes import LifeExperience
from flybrain.world_model.predictor import NextStateWorldModel


def format_live_step_log(
    step: int,
    action: np.ndarray,
    info: dict[str, Any],
    *,
    verbose: bool = False,
) -> str:
    """Console summary: every step shows motor + current mind snapshot."""
    refreshed = bool(info.get("mind_tick"))
    tag = "MIND*" if refreshed else "motor"
    act = np.asarray(action, dtype=np.float32).reshape(-1)
    fwd = float(act[0]) if act.size else 0.0
    turn = float(act[1]) if act.size > 1 else 0.0
    goal = info.get("motor_goal") or info.get("goal")
    if goal is not None:
        g = np.asarray(goal, dtype=np.float32).reshape(-1)
        goal_s = f"goal=[{float(g[0]):+.2f},{float(g[1]):+.2f}]" if g.size >= 2 else "goal=?"
    else:
        intent = info.get("intent") or {}
        goal_s = (
            f"intent=[{float(intent.get('forward', 0.0)):+.2f},"
            f"{float(intent.get('turn', 0.0)):+.2f}]"
        )
    track = info.get("motor_tracking_error")
    track_s = f" track={float(track):.3f}" if track is not None else ""
    energy = info.get("energy")
    energy_s = f" E={float(energy):.2f}" if energy is not None else ""
    hunger = info.get("hunger")
    hunger_s = f" H={float(hunger):.2f}" if hunger is not None else ""
    cov = info.get("coverage")
    cov_s = f" cov={float(cov):.3f}" if cov is not None else ""

    intent = info.get("intent") or {}
    future = intent.get("future") if isinstance(intent, dict) else None
    chosen = ""
    l1_s = ""
    if isinstance(future, dict) and future.get("chosen_name"):
        chosen = f" plan={future.get('chosen_name')}"
        layer1 = future.get("layer1_values") or []
        if isinstance(layer1, list) and layer1:
            top = []
            for row in layer1[:3]:
                if not isinstance(row, dict):
                    continue
                top.append(f"{row.get('name')}={float(row.get('value', 0.0)):.2f}")
            if top:
                l1_s = f" L1[{', '.join(top)}]"

    affect = info.get("affect") or {}
    survival = affect.get("survival_risk") if isinstance(affect, dict) else None
    surv_s = f" surv={float(survival):.2f}" if survival is not None else ""
    fear = affect.get("fear") if isinstance(affect, dict) else None
    fear_s = f" fear={float(fear):.2f}" if fear is not None else ""
    cues = info.get("vision_cues") if isinstance(info.get("vision_cues"), dict) else {}
    wall = cues.get("wall_hazard") if cues else None
    wall_s = f" wall={float(wall):.2f}" if wall is not None else ""
    bearing = cues.get("target_bearing") if cues else None
    bearing_s = f" bear={float(bearing):+.2f}" if bearing is not None else ""
    cur = info.get("curiosity_gated")
    cur_s = f" cur={float(cur):.3f}" if cur is not None else ""

    thought = " ".join(str(info.get("refined_thought") or "").split())
    critique = " ".join(str(info.get("devil_critique") or "").split())
    thought_limit = 500 if verbose else 220
    critique_limit = 280 if verbose else 140
    # Always show current mind snapshot (held across motor ticks until the next MIND*).
    thought_s = f" | mind: {thought[:thought_limit]}" if thought else ""
    critique_s = f" | devil: {critique[:critique_limit]}" if critique else ""

    return (
        f"[live {step:04d} {tag}] {goal_s} act=[{fwd:+.2f},{turn:+.2f}]"
        f"{track_s}{energy_s}{hunger_s}{surv_s}{fear_s}{wall_s}{bearing_s}{cur_s}"
        f"{cov_s}{chosen}{l1_s}{thought_s}{critique_s}"
    )


def format_live_mind_detail(step: int, info: dict[str, Any]) -> str:
    """Multi-line dump for MIND* ticks when verbose logging is on."""
    import json

    intent = info.get("intent") or {}
    future = intent.get("future") if isinstance(intent, dict) else {}
    payload = {
        "step": step,
        "plan": (future or {}).get("chosen_name"),
        "intent": {
            "forward": intent.get("forward") if isinstance(intent, dict) else None,
            "turn": intent.get("turn") if isinstance(intent, dict) else None,
            "confidence": intent.get("confidence") if isinstance(intent, dict) else None,
        },
        "vision_cues": info.get("vision_cues"),
        "affect": {
            k: (info.get("affect") or {}).get(k)
            for k in ("survival_risk", "fear", "novelty", "exploration", "joy")
        },
        "layer1_values": (future or {}).get("layer1_values"),
        "refined_thought": info.get("refined_thought"),
        "devil_critique": info.get("devil_critique"),
        "curiosity_gated": info.get("curiosity_gated"),
        "energy": info.get("energy"),
        "hunger": info.get("hunger"),
    }
    return f"[live {step:04d} DETAIL] " + json.dumps(payload, default=str)


def log_live_step(
    step: int,
    action: np.ndarray,
    info: dict[str, Any],
    *,
    log_path: str | Path | None = None,
    verbose: bool = False,
) -> None:
    """Print a live step line to the console and optionally append to a file."""
    line = format_live_step_log(step, action, info, verbose=verbose)
    # stdout + stderr: some IDE/Unity-adjacent terminals only surface one stream.
    print(line, flush=True)
    print(line, file=sys.stderr, flush=True)
    detail = ""
    if verbose and bool(info.get("mind_tick")):
        detail = format_live_mind_detail(step, info)
        print(detail, flush=True)
        print(detail, file=sys.stderr, flush=True)
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            if detail:
                handle.write(detail + "\n")
            handle.flush()

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
    stimulus_thresholds: dict[str, float] | None = None
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
    seed: int = 42
    # Phase 9 architecture B: mind goals at mind_interval; motor tracks every step.
    enable_motor_layer: bool = True
    mind_interval: int = DEFAULT_MIND_INTERVAL
    motor: MotorLayerConfig | None = None
    console_log: bool = False
    console_log_every: int = 1
    # When set, each [live …] line is also appended here (easier than hunting terminals).
    log_path: str | None = None
    # Longer mind text + full DETAIL JSON lines on each MIND* tick.
    console_log_verbose: bool = False

    @classmethod
    def from_yaml(
        cls, data: dict[str, Any], visual_nav: dict[str, Any] | None = None
    ) -> LiveMindConfig:
        db = data.get("database", {})
        thinking = data.get("thinking", {})
        live = data.get("live", {})
        llm = data.get("llm", {})
        wm = data.get("world_model", {}) or {}
        motor_raw = data.get("motor", {}) or {}
        vn = visual_nav or {}
        vn_live = vn.get("live", {})
        clamps = dict(thinking.get("action_clamps") or {})
        if "wall_hazard_forward_scale" in vn_live:
            clamps["wall_hazard_forward_scale"] = vn_live["wall_hazard_forward_scale"]
        elif "wall_hazard_forward_scale" in vn.get("arena", {}):
            clamps.setdefault("wall_hazard_forward_scale", 0.2)
        clamps.setdefault(
            "wall_hazard_forward_scale", float(vn_live.get("wall_hazard_forward_scale", 0.2))
        )
        return cls(
            fly_id=str(data.get("fly_id", "fly_001")),
            experiment_id=str(data.get("experiment_id", "experiment_mind_000001")),
            max_steps=int(live.get("max_steps", 200)),
            backend=str(db.get("backend", "in_memory")),
            dsn=db.get("dsn"),
            llm_mock=bool(llm.get("mock", True)),
            action_clamps=clamps,
            stimulus_thresholds=dict(thinking.get("stimulus_thresholds") or {}) or None,
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
            seed=int(data.get("seed", live.get("seed", 42))),
            enable_motor_layer=bool(motor_raw.get("enabled", live.get("enable_motor_layer", True))),
            mind_interval=int(
                motor_raw.get("mind_interval", live.get("mind_interval", DEFAULT_MIND_INTERVAL))
            ),
            motor=MotorLayerConfig.from_dict(motor_raw),
            console_log=bool(live.get("console_log", False)),
            console_log_every=max(1, int(live.get("console_log_every", 1))),
            log_path=str(live["log_path"]) if live.get("log_path") else None,
            console_log_verbose=bool(live.get("console_log_verbose", False)),
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
                stimulus_thresholds=self.config.stimulus_thresholds,
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
                    self.vision_head.load_state_dict(
                        state["model"] if isinstance(state, dict) and "model" in state else state
                    )
                    self._vision_trained = True
            self.vision_head.eval()
            self.vision_head.reset_state(1)
        self.curiosity = CuriosityDrive(self.config.curiosity or CuriosityConfig())
        motor_cfg = self.config.motor or MotorLayerConfig()
        if self.config.motor is None and not self.config.enable_motor_layer:
            motor_cfg = MotorLayerConfig(mode="heuristic")
        self.motor = MotorLayer(motor_cfg) if self.config.enable_motor_layer else None
        self._step = 0
        self._pending_action: np.ndarray | None = None
        self._last_mind_info: dict[str, Any] = {}
        self._mind_ticks = 0
        self._motor_ticks = 0

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
        model.load_state_dict(
            state["model"] if isinstance(state, dict) and "model" in state else state
        )
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
                np.clip(
                    affect_payload.get("exploration", 0.0) + 0.35 * curiosity_step.explore_bias,
                    0.0,
                    1.0,
                )
            )
            affect_payload["novelty"] = float(
                np.clip(
                    affect_payload.get("novelty", 0.0) + 0.4 * curiosity_step.novelty.novelty_score,
                    0.0,
                    1.0,
                )
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
        mind_action = self.curiosity.apply_action_bias(thought.action, curiosity_step.explore_bias)
        self._mind_ticks += 1
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
            "mind_tick": True,
            "energy": signal_payload.get("energy", sensory.summary.get("energy")),
            "hunger": signal_payload.get("hunger", sensory.summary.get("hunger")),
        }
        self._last_mind_info = dict(info)

        if self.motor is not None:
            # Prefer L1/L2 plan as the motor goal; curiosity may still nudge it.
            raw_goal = goal_from_intent(thought.intent, mind_action)
            goal = self.curiosity.apply_action_bias(raw_goal, curiosity_step.explore_bias)
            self.motor.set_goal(goal)
            set_goal = getattr(self.env, "set_goal", None)
            if callable(set_goal):
                set_goal(goal)
            action = self.motor.act(observation)
            self._motor_ticks += 1
            info["motor_goal"] = goal.tolist()
            info["motor_action"] = action.tolist()
            info["motor_tracking_error"] = self.motor.tracking_error()
            info["architecture"] = "B"
        else:
            action = mind_action
            info["architecture"] = "A"
        self._pending_action = action.copy()
        return action, info

    def process_motor_only(self, observation: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Motor tick without a new mind thought (architecture B)."""
        info = dict(self._last_mind_info)
        info["mind_tick"] = False
        # Keep novelty / curiosity ticking so exploration metrics stay live during motor steps.
        sensory = self.signals.build(observation)
        affect = self.affect.estimate(sensory)
        embedding = self.embedder.embed(sensory, affect)
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
            signals=sensory.to_jsonable(),
            affect=affect.to_jsonable(),
            position=position,
            chem_vector=chem_vec,
        )
        info["curiosity"] = curiosity_step.curiosity
        info["curiosity_gated"] = curiosity_step.gated_curiosity
        info["curiosity_gate"] = curiosity_step.gate
        info["explore_bias"] = curiosity_step.explore_bias
        info["coverage"] = curiosity_step.novelty.coverage
        info["unique_cells"] = curiosity_step.novelty.unique_cells
        info["exploration_xp"] = curiosity_step.novelty.exploration_xp
        summary = sensory.to_jsonable() if hasattr(sensory, "to_jsonable") else sensory.summary
        info["energy"] = summary.get("energy", sensory.summary.get("energy"))
        info["hunger"] = summary.get("hunger", sensory.summary.get("hunger"))
        if self.motor is None:
            action = (
                self._pending_action
                if self._pending_action is not None
                else np.asarray([0.0, 0.0, 0.5, 0.5, 0.0], dtype=np.float32)
            )
            info["architecture"] = "A"
            return action, info
        action = self.motor.act(observation)
        self._motor_ticks += 1
        info["motor_goal"] = self.motor.goal.tolist()
        info["motor_action"] = action.tolist()
        info["motor_tracking_error"] = self.motor.tracking_error()
        info["architecture"] = "B"
        self._pending_action = action.copy()
        return action, info

    def run(
        self,
        *,
        max_steps: int | None = None,
        continuous: bool = False,
        realtime_hz: float | None = None,
    ) -> dict[str, Any]:
        import time

        if continuous:
            step_limit: int | None = None
        else:
            step_limit = max_steps if max_steps is not None else self.config.max_steps
        self._step = 0
        self._mind_ticks = 0
        self._motor_ticks = 0
        self.curiosity.reset()
        self._pending_action = None
        self._last_mind_info = {}
        if self.motor is not None:
            self.motor.reset()
        obs, reset_info = self.env.reset(seed=self.config.seed)
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
        interval = max(1, int(self.config.mind_interval))
        period = (1.0 / float(realtime_hz)) if realtime_hz and realtime_hz > 0 else None
        if self.config.console_log:
            print(
                "live_mind_waiting_first_tick "
                "(first [live …] line appears after the first mind/motor compute; "
                "then each Unity Academy decision)",
                flush=True,
            )
        while step_limit is None or self._step < step_limit:
            tick_start = time.perf_counter()
            should_think = (
                (self.motor is None) or (self._step % interval == 0) or not self._last_mind_info
            )
            if should_think:
                if self.config.console_log and self._step == 0:
                    print("live_mind_thinking step=0 …", flush=True)
                action, info = self.process_observation(obs)
            else:
                action, info = self.process_motor_only(obs)
            # Blend food-search heuristic into the *goal* path when mind just thought.
            food_policy = getattr(self.env, "food_policy_action", None)
            if callable(food_policy) and info.get("mind_tick"):
                food_action = food_policy()
                if self.motor is not None:
                    blended = action.copy()
                    blended[0] = 0.4 * float(self.motor.goal[0]) + 0.6 * float(food_action[0])
                    blended[1] = 0.35 * float(self.motor.goal[1]) + 0.65 * float(food_action[1])
                    blended[4] = 0.3 * float(self.motor.goal[4]) + 0.7 * float(food_action[4])
                    self.motor.set_goal(blended)
                    set_goal = getattr(self.env, "set_goal", None)
                    if callable(set_goal):
                        set_goal(blended)
                    action = self.motor.act(obs)
                else:
                    action = action.copy()
                    action[0] = 0.4 * float(action[0]) + 0.6 * float(food_action[0])
                    action[1] = 0.35 * float(action[1]) + 0.65 * float(food_action[1])
                    action[4] = 0.3 * float(action[4]) + 0.7 * float(food_action[4])
            vision_policy = getattr(self.env, "vision_policy_action", None)
            if callable(vision_policy) and info.get("mind_tick"):
                vision_action = vision_policy()
                if self.motor is not None:
                    blended = self.motor.goal.copy()
                    blended[0] = 0.35 * float(blended[0]) + 0.65 * float(vision_action[0])
                    blended[1] = 0.25 * float(blended[1]) + 0.75 * float(vision_action[1])
                    self.motor.set_goal(blended)
                    action = self.motor.act(obs)
                else:
                    action = action.copy()
                    action[0] = 0.35 * float(action[0]) + 0.65 * float(vision_action[0])
                    action[1] = 0.25 * float(action[1]) + 0.75 * float(vision_action[1])
                self._pending_action = action.copy()
            explore_policy = getattr(self.env, "explore_policy_action", None)
            if (
                callable(explore_policy)
                and info.get("mind_tick")
                and float(info.get("curiosity_gate", 0.0)) > 0.2
            ):
                explore_action = explore_policy()
                blend = 0.45 * float(info.get("explore_bias", 0.0))
                if self.motor is not None:
                    g = self.motor.goal.copy()
                    g[0] = (1.0 - blend) * float(g[0]) + blend * float(explore_action[0])
                    g[1] = (1.0 - blend) * float(g[1]) + blend * float(explore_action[1])
                    g[3] = (1.0 - 0.5 * blend) * float(g[3]) + 0.5 * blend * float(
                        explore_action[3]
                    )
                    self.motor.set_goal(g)
                    action = self.motor.act(obs)
                else:
                    action = action.copy()
                    action[0] = (1.0 - blend) * float(action[0]) + blend * float(explore_action[0])
                    action[1] = (1.0 - blend) * float(action[1]) + blend * float(explore_action[1])
                    action[3] = (1.0 - 0.5 * blend) * float(action[3]) + 0.5 * blend * float(
                        explore_action[3]
                    )
                self._pending_action = action.copy()
            last_info = info
            # Log BEFORE env.step: Unity/ML-Agents step() blocks until the next
            # Academy decision, so post-step logging looks like "nothing until exit".
            if self.config.console_log and (self._step % self.config.console_log_every == 0):
                log_live_step(
                    self._step,
                    action,
                    last_info,
                    log_path=self.config.log_path,
                    verbose=self.config.console_log_verbose,
                )
            obs, _reward, terminated, truncated, step_info = self.env.step(action)
            if isinstance(step_info, dict):
                last_info = {**last_info, **step_info}
            self._step += 1
            if period is not None:
                elapsed = time.perf_counter() - tick_start
                sleep_for = period - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
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
            "unique_cells": int(
                last_info.get("unique_cells", curiosity_summary.get("unique_cells", 0))
            ),
            "exploration_xp": float(
                last_info.get("exploration_xp", curiosity_summary.get("exploration_xp", 0.0))
            ),
            "curiosity_reward_mean": float(curiosity_summary.get("curiosity_reward_mean", 0.0)),
            "curiosity": curiosity_summary,
            "mind_ticks": self._mind_ticks,
            "motor_ticks": self._motor_ticks,
            "mind_interval": interval,
            "architecture": "B" if self.motor is not None else "A",
        }


def make_live_env(
    *,
    fake: bool = True,
    fake_visual_nav: bool = False,
    fake_food_search: bool = False,
    fake_explore: bool = False,
    fake_locomotion: bool = False,
    editor: bool = False,
    build: str | None = None,
    port: int | None = None,
    seed: int | None = None,
):
    env_seed = 42 if seed is None else int(seed)
    if fake_locomotion:
        from flybrain.env_locomotion import FakeLocomotionConfig, FakeLocomotionEnv

        return FakeLocomotionEnv(FakeLocomotionConfig(seed=env_seed))
    if fake_explore:
        from flybrain.env_explore import FakeExplorationConfig, FakeExplorationEnv

        return FakeExplorationEnv(FakeExplorationConfig(seed=env_seed))
    if fake_food_search:
        from flybrain.env_food_search import FakeFoodSearchConfig, FakeFoodSearchEnv

        return FakeFoodSearchEnv(FakeFoodSearchConfig(seed=env_seed))
    if fake_visual_nav:
        from flybrain.env_visual_nav import FakeVisualNavConfig, FakeVisualNavEnv

        return FakeVisualNavEnv(FakeVisualNavConfig(seed=env_seed))
    if fake:
        from flybrain.env import FakeFlyConfig

        return FakeFlyEnv(FakeFlyConfig(seed=env_seed))
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
            timeout_wait=600 if editor else 120,
            seed=env_seed,
        )
    )
