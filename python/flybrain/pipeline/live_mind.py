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
from flybrain.curiosity.boredom import (
    BoredomConfig,
    BoredomTracker,
    apply_boredom_escape_goal,
)
from flybrain.env import FakeFlyEnv
from flybrain.memory.embedder import MindEmbedder
from flybrain.memory.pgvector import ShortTermMemory, open_short_term_memory
from flybrain.memory.place_graph import PlaceGraphConfig, open_place_graph
from flybrain.sensors.mapped_feedback import MappedSensorFeedback
from flybrain.motor import MotorLayer, MotorLayerConfig, goal_from_intent
from flybrain.phase9 import DEFAULT_MIND_INTERVAL
from flybrain.signals.builder import SignalBuilder
from flybrain.thinking.service import ThinkingConfig, ThinkingService
from flybrain.thinking.visual_thinker import VisualThinker, VisualThinkerConfig
from flybrain.thinking.personality import PersonalityState
from flybrain.thinking.judgement import (
    DevilProposal,
    JudgementConfig,
    JudgementModule,
    apply_proposal,
    blend_goals,
)
from flybrain.thinking.interest import (
    InterestConfig,
    InterestPointStore,
    apply_investigation_goal,
)
from flybrain.thinking.conditioning import (
    ConditioningConfig,
    HazardConditioning,
    conditioning_snapshot_signals,
    load_conditioning_from_stm,
    rewrite_present_after_judgement,
)
from flybrain.thinking.devils_advocate import propose_critique
from flybrain.thinking.retrospective import RetrospectiveConfig
from flybrain.thinking.stimulus_interrupt import (
    STOP_GOAL,
    PassiveSmellGate,
    PatchGridHistoryCache,
    RelativeHistoryBaseline,
    StimulusInterruptConfig,
    evaluate_stimulus_interrupt,
    heighten_sensory_for_deep_thought,
)
from flybrain.vision.recurrent_head import (
    RecurrentVisionHead,
    VisionNavCues,
    heuristic_cues_from_frames,
)
from flybrain.vision.novelty import (
    VisualNoveltyConfig,
    VisualNoveltyTracker,
    apply_visual_explore_goal,
    stereo_patch_embedding,
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
    place = info.get("place") if isinstance(info.get("place"), dict) else None
    place_s = ""
    if place and place.get("place_id") is not None:
        label = place.get("label") or "place"
        place_s = (
            f" place={label}:{place.get('place_id')}"
            f"{'*' if place.get('is_new') else ''}"
            f"(sim={float(place.get('similarity', 0.0)):.2f}"
            f",fam={float(place.get('mean_familiarity', 0.0)):.2f}"
            f",raw={float(place.get('wall_hazard_raw', wall or 0.0)):.2f})"
        )
        senses = place.get("senses") if isinstance(place.get("senses"), dict) else {}
        bitter = senses.get("bitter") if isinstance(senses, dict) else None
        if isinstance(bitter, dict) and bitter.get("value") is not None:
            place_s += f" smells(bitter={float(bitter['value']):.2f})"
        elif isinstance(senses, dict):
            odor = senses.get("odor") if isinstance(senses.get("odor"), dict) else None
            if odor and odor.get("value") is not None:
                place_s += f" smells(odor={float(odor['value']):.2f})"
    bearing = cues.get("target_bearing") if cues else None
    bearing_s = f" bear={float(bearing):+.2f}" if bearing is not None else ""
    cur = info.get("curiosity_gated")
    cur_s = f" cur={float(cur):.3f}" if cur is not None else ""
    bore = info.get("boredom")
    bore_s = f" bore={float(bore):.2f}" if bore is not None else ""
    tend = info.get("tendency") if isinstance(info.get("tendency"), dict) else None
    tend_s = ""
    if tend and tend.get("dominant"):
        tend_s = (
            f" tend={tend.get('dominant')}"
            f"(c={float(tend.get('curiosity', 0)):.2f}"
            f",s={float(tend.get('sex', 0)):.2f}"
            f",f={float(tend.get('forage', 0)):.2f}"
            f"{',safe' if tend.get('safe_mode') else ''})"
        )
    viz = info.get("visual_novelty")
    viz_s = ""
    if viz is not None:
        nb = info.get("novelty_bearing")
        nb_s = f",nb={float(nb):+.2f}" if nb is not None else ""
        new = "!" if info.get("new_stimulus") else ""
        viz_s = f" viz={float(viz):.2f}{nb_s}{new}"
    antenna_s = ""
    antenna = info.get("antenna_debug")
    if verbose and isinstance(antenna, dict):
        antenna_s = (
            " ant["
            f"L(s={float(antenna.get('left_sugar', 0.0)):.4f},"
            f"b={float(antenna.get('left_bitter', 0.0)):.4f},"
            f"o={float(antenna.get('left_odor', 0.0)):.4f}) "
            f"R(s={float(antenna.get('right_sugar', 0.0)):.4f},"
            f"b={float(antenna.get('right_bitter', 0.0)):.4f},"
            f"o={float(antenna.get('right_odor', 0.0)):.4f})"
            "]"
        )
    vthink_s = ""
    if info.get("visual_thinking"):
        vthink_s = " vthink=…"
    else:
        insight = info.get("visual_insight")
        if isinstance(insight, dict) and insight.get("ran") and insight.get("decision") != "ignore":
            vthink_s = f" vthink={insight.get('decision')}(sal={float(insight.get('salience', 0.0)):.2f})"
    judge_s = ""
    judgement = info.get("judgement")
    if isinstance(judgement, dict) and judgement.get("decision"):
        dec = judgement.get("decision")
        if dec == "accept":
            judge_s = f" judge=accept(w={float(judgement.get('blend_weight', 0.0)):.2f})"
        elif dec == "acknowledge":
            judge_s = " judge=acknowledge"
        else:
            judge_s = f" judge={dec}"
        eye_m = info.get("eye_match")
        if eye_m is not None and float(eye_m) > 0.0:
            judge_s += f"(eye={float(eye_m):.2f})"
    invest = info.get("investigation")
    if isinstance(invest, dict) and invest.get("remaining") is not None:
        judge_s += (
            f" invest={invest.get('kind')}(rem={int(invest.get('remaining', 0))})"
        )

    si_s = ""
    si_dbg = info.get("interrupt_debug")
    if isinstance(si_dbg, dict):
        scores = si_dbg.get("scores") or {}
        bases = si_dbg.get("baselines") or {}
        sc_s = ",".join(f"{k[:1]}{k[-1]}={scores[k]:.3f}" for k in scores)
        base_s = ",".join(f"{k[:1]}{k[-1]}={bases[k]:.3f}" for k in bases) if bases else "-"
        wc = si_dbg.get("worst_cell")
        wc_s = f" wc={int(wc)}" if isinstance(wc, (int, float)) and int(wc) >= 0 else ""
        si_s = (
            f" si[{si_dbg.get('reason')} cue={si_dbg.get('cue')}"
            f" sim({sc_s}) base({base_s}){wc_s}"
            f" novel={','.join(si_dbg.get('novel') or []) or '-'}"
            f" soft={int(bool(si_dbg.get('soft_low_load')))}"
            f" miss={int(bool(si_dbg.get('history_miss')))}"
            f" armed={int(bool(si_dbg.get('armed')))}]"
        )

    thought = " ".join(str(info.get("refined_thought") or "").split())
    critique = " ".join(str(info.get("devil_critique") or "").split())
    thought_limit = 500 if verbose else 220
    critique_limit = 280 if verbose else 140
    # Always show current mind snapshot (held across motor ticks until the next MIND*).
    thought_s = f" | mind: {thought[:thought_limit]}" if thought else ""
    critique_s = f" | devil: {critique[:critique_limit]}" if critique else ""

    return (
        f"[live {step:04d} {tag}] {goal_s} act=[{fwd:+.2f},{turn:+.2f}]"
        f"{track_s}{energy_s}{hunger_s}{surv_s}{fear_s}{wall_s}{place_s}{bearing_s}{cur_s}{bore_s}{tend_s}{viz_s}{antenna_s}{vthink_s}{judge_s}"
        f"{si_s}{cov_s}{chosen}{l1_s}{thought_s}{critique_s}"
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
        "place": info.get("place"),
        "mapped_sensors": info.get("mapped_sensors"),
        "affect": {
            k: (info.get("affect") or {}).get(k)
            for k in ("survival_risk", "fear", "novelty", "exploration", "joy")
        },
        "layer1_values": (future or {}).get("layer1_values"),
        "refined_thought": info.get("refined_thought"),
        "devil_critique": info.get("devil_critique"),
        "devil_proposal": info.get("devil_proposal"),
        "judgement": info.get("judgement"),
        "personality": info.get("personality"),
        "interest_point": info.get("interest_point"),
        "investigation": info.get("investigation"),
        "interest_pending": info.get("interest_pending"),
        "eye_match": info.get("eye_match"),
        "eye_match_interest_id": info.get("eye_match_interest_id"),
        "curiosity_gated": info.get("curiosity_gated"),
        "energy": info.get("energy"),
        "hunger": info.get("hunger"),
        "antenna_debug": info.get("antenna_debug"),
    }
    return f"[live {step:04d} DETAIL] " + json.dumps(payload, default=str)


def antenna_debug_payload(sensory: Any) -> dict[str, float]:
    """Expose raw antenna Sugar/Bitter/Odor channels for debug logging."""
    smell = np.asarray(getattr(sensory, "smell_vec", []), dtype=np.float32).reshape(-1)

    def value(index: int) -> float:
        return float(smell[index]) if smell.size > index else 0.0

    return {
        "left_sugar": value(0),
        "left_bitter": value(1),
        "left_odor": value(7),
        "right_sugar": value(8),
        "right_bitter": value(9),
        "right_odor": value(15),
    }


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
    llm_kwargs: dict[str, Any] | None = None
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
    boredom: BoredomConfig | None = None
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
    tendency: dict[str, Any] | None = None
    visual_novelty: VisualNoveltyConfig | None = None
    visual_thinker: VisualThinkerConfig | None = None
    personality: dict[str, Any] | None = None
    judgement: JudgementConfig | None = None
    interest: InterestConfig | None = None
    stimulus_interrupt: StimulusInterruptConfig | None = None
    place_graph: PlaceGraphConfig | None = None

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
        curiosity_raw = data.get("curiosity") or {}
        boredom_raw = data.get("boredom") or curiosity_raw.get("boredom") or {}
        vn = visual_nav or {}
        vn_live = vn.get("live", {})
        visual_novelty_raw = data.get("visual_novelty") or (vn.get("visual_novelty") or {})
        visual_thinker_raw = data.get("visual_thinking") or (vn.get("visual_thinking") or {})
        personality_raw = data.get("personality") or {}
        judgement_raw = data.get("judgement") or {}
        interest_raw = data.get("interest") or {}
        interrupt_raw = data.get("stimulus_interrupt") or {}
        place_raw = data.get("place_graph") or {}
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
            llm_kwargs={
                key: llm[key]
                for key in (
                    "base_url",
                    "model",
                    "api_key",
                    "timeout_seconds",
                    "background",
                    "max_tokens",
                    "temperature",
                    "keep_alive",
                )
                if llm.get(key) is not None
            },
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
            curiosity=CuriosityConfig.from_dict(curiosity_raw),
            boredom=BoredomConfig.from_dict(boredom_raw),
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
            tendency=dict(data.get("tendency") or {}),
            visual_novelty=VisualNoveltyConfig.from_dict(visual_novelty_raw),
            visual_thinker=VisualThinkerConfig.from_dict(visual_thinker_raw),
            personality=dict(personality_raw),
            judgement=JudgementConfig.from_dict(judgement_raw),
            interest=InterestConfig.from_dict(interest_raw),
            stimulus_interrupt=StimulusInterruptConfig.from_dict(interrupt_raw),
            place_graph=PlaceGraphConfig.from_dict(place_raw),
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
                llm_kwargs=self.config.llm_kwargs,
                priorities=self.config.priorities,
                retrospective=self.config.retrospective,
                enable_retrospective=self.config.enable_retrospective,
                tendency=self.config.tendency,
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
        self.boredom = BoredomTracker(self.config.boredom or BoredomConfig())
        self.visual_novelty = VisualNoveltyTracker(
            self.config.visual_novelty or VisualNoveltyConfig()
        )
        self.visual_thinker = VisualThinker(
            self.config.visual_thinker or VisualThinkerConfig(),
            novelty_config=self.config.visual_novelty or VisualNoveltyConfig(),
        )
        self.personality = PersonalityState.from_config(self.config.personality)
        self.judgement = JudgementModule(self.config.judgement or JudgementConfig())
        self.interests = InterestPointStore(self.config.interest or InterestConfig())
        self.thinking.interest_store = self.interests
        self.stimulus_interrupt = self.config.stimulus_interrupt or StimulusInterruptConfig()
        self._history_baseline = RelativeHistoryBaseline.from_config(self.stimulus_interrupt)
        self._patch_history = PatchGridHistoryCache.from_config(self.stimulus_interrupt)
        self._passive_smell = PassiveSmellGate.from_config(self.stimulus_interrupt)
        si = self.stimulus_interrupt
        self.conditioning = HazardConditioning(
            ConditioningConfig(
                hit_threshold=int(getattr(si, "conditioning_hit_threshold", 3) or 3),
                judgement_confirmations=int(getattr(si, "judgement_confirmations", 2) or 2),
            )
        )
        pg_cfg = self.config.place_graph or PlaceGraphConfig(enabled=False, backend="in_memory")
        self.place_graph_config = pg_cfg
        self.place_graph = None
        self.mapped_sensors: MappedSensorFeedback | None = None
        if pg_cfg.enabled:
            self.place_graph = open_place_graph(
                backend=pg_cfg.backend,
                uri=pg_cfg.uri,
                user=pg_cfg.user,
                password=pg_cfg.password,
                database=pg_cfg.database,
                view_dim=int(pg_cfg.view_dim),
                cache=pg_cfg.cache,
            )
            self.mapped_sensors = MappedSensorFeedback(
                store=self.place_graph,
                config=pg_cfg,
                fly_id=self.config.fly_id,
            )
        self._interrupt_pending = False
        self._interrupt_remaining = 0
        self._interrupt_reason = ""
        self._interrupt_meta: dict[str, Any] = {}
        self._interrupt_patch_vec: np.ndarray | None = None
        self._interrupt_smell_vec: np.ndarray | None = None
        self._interrupt_sensors_vec: np.ndarray | None = None
        self._interrupt_joint_vec: np.ndarray | None = None
        self._interrupt_cooldown_until = 0
        self._resume_goal: np.ndarray | None = None
        motor_cfg = self.config.motor or MotorLayerConfig()
        if self.config.motor is None and not self.config.enable_motor_layer:
            motor_cfg = MotorLayerConfig(mode="heuristic")
        self.motor = MotorLayer(motor_cfg) if self.config.enable_motor_layer else None
        self._step = 0
        self._pending_action: np.ndarray | None = None
        self._last_mind_info: dict[str, Any] = {}
        self._mind_ticks = 0
        self._motor_ticks = 0

    def _retrospective_cfg(self) -> RetrospectiveConfig:
        raw = dict(self.config.retrospective or {})
        known = {f.name for f in RetrospectiveConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return RetrospectiveConfig(**{k: v for k, v in raw.items() if k in known})

    def _hydrate_conditioning_from_stm(self) -> bool:
        """Restore hazard conditioning from the newest STM snapshot."""
        return load_conditioning_from_stm(
            self.conditioning,
            self.memory,
            fly_id=self.config.fly_id,
        )

    def _persist_conditioning_to_stm(self, *, info: dict[str, Any] | None = None) -> Any:
        """Append a conditioning checkpoint row into short-term memory."""
        snap = self.conditioning.to_dict()
        if not snap.get("entries"):
            return None
        signals = conditioning_snapshot_signals(self.conditioning)
        try:
            rec = self.memory.insert_embedding(
                embedding=np.zeros(self.embedder.total_dim, dtype=np.float32),
                signals=signals,
                affect={},
                episode_id=self.config.experiment_id,
                step=int(self._step),
                sim_time=float(self._step) * 0.05,
                fly_id=self.config.fly_id,
            )
        except Exception as exc:  # pragma: no cover - DB optional in some runs
            if self.config.console_log:
                print(f"conditioning_stm_save_failed step={self._step} err={exc}", flush=True)
            return None
        if info is not None:
            info["conditioning_stm_id"] = getattr(rec, "id", None)
            info["conditioning"] = snap
        return rec

    def _stamp_conditioning_signals(self, signal_payload: dict[str, Any]) -> None:
        """Embed live conditioning tip keys + full snapshot into mind-tick signals."""
        signal_payload.update(self.conditioning.to_signals())
        signal_payload["conditioning"] = self.conditioning.to_dict()


    def _arm_stimulus_interrupt(
        self,
        *,
        observation: np.ndarray,
        left_eye: np.ndarray | None,
        right_eye: np.ndarray | None,
        sensory: Any,
        affect: Any,
        visual_nov: Any | None,
        info: dict[str, Any] | None = None,
    ) -> None:
        """Evaluate cue vs STM joint history; arm when novel under soft load."""
        del observation  # reserved for future observation-driven cues
        if self._interrupt_pending:
            return
        cfg = self.stimulus_interrupt
        if not cfg.enabled:
            return
        signals = sensory.to_jsonable() if hasattr(sensory, "to_jsonable") else dict(sensory.summary)
        affect_payload = affect.to_jsonable() if hasattr(affect, "to_jsonable") else dict(affect)
        backend = str(cfg.history_backend).lower()
        parts = None
        sensors = None
        if backend in ("stm", "joint"):
            parts = self.embedder.embed_parts(sensory, affect)
            sensors = np.asarray(parts.sensors_vec, dtype=np.float32)
        smell = None
        if hasattr(sensory, "smell_vec"):
            smell = np.asarray(sensory.smell_vec, dtype=np.float32)
        decision = evaluate_stimulus_interrupt(
            memory=self.memory,
            parts=parts,
            signals=signals,
            affect=affect_payload,
            fly_id=self.config.fly_id,
            step=int(self._step),
            cooldown_until_step=int(self._interrupt_cooldown_until),
            config=cfg,
            stimulus_thresholds=self.config.stimulus_thresholds,
            retrospective=self._retrospective_cfg(),
            visual_novelty=float(visual_nov.visual_novelty) if visual_nov is not None else 0.0,
            new_stimulus=bool(visual_nov.new_stimulus) if visual_nov is not None else False,
            baseline=self._history_baseline,
            patch_cache=self._patch_history,
            left_eye=left_eye,
            right_eye=right_eye,
            smell_vec=smell,
            sensors_vec=sensors,
            passive_smell=self._passive_smell,
        )
        # Debug: expose *why* an interrupt did / didn't arm whenever a cue was present.
        if decision.candidate:
            debug = {
                "reason": decision.reason,
                "cue": decision.cue,
                "backend": str(cfg.history_backend),
                "min_sim": round(float(decision.best_similarity), 4),
                "scores": {k: round(float(v), 4) for k, v in decision.modality_scores.items()},
                "baselines": {
                    k: round(float(v), 4) for k, v in decision.history_baselines.items()
                },
                "novel": list(decision.novel_modalities),
                "worst_cell": int(decision.patch_worst_cell),
                "soft_low_load": bool(decision.soft_low_load),
                "history_miss": bool(decision.history_miss),
                "armed": bool(decision.should_arm),
                "passive_smell": decision.passive_smell_reason or "closed",
                "passive_delta": round(float(decision.passive_smell_delta), 4),
                "smell_probe": bool(decision.smell_probe_ran),
            }
            if info is not None:
                info["interrupt_debug"] = debug
            if self.config.console_log:
                print(f"si_debug step={self._step} {debug}", flush=True)
        if decision.should_arm:
            self._interrupt_pending = True
            self._interrupt_remaining = max(1, int(cfg.deep_thought_steps))
            self._interrupt_reason = decision.reason
            self.conditioning.begin_freeze()
            self._resume_goal = None
            self._interrupt_meta = {
                "cue": decision.cue,
                "backend": str(cfg.history_backend),
                "best_similarity": decision.best_similarity,
                "modality_scores": dict(decision.modality_scores),
                "history_baselines": dict(decision.history_baselines),
                "novel_modalities": list(decision.novel_modalities),
                "reason": decision.reason,
                "deep_thought_steps": int(self._interrupt_remaining),
            }
            # Hold patch+smell+sensors+joint until the freeze clears.
            if decision.patch_embedding is not None:
                self._interrupt_patch_vec = np.asarray(
                    decision.patch_embedding, dtype=np.float32
                ).copy()
            else:
                self._interrupt_patch_vec = None
            if decision.smell_embedding is not None:
                self._interrupt_smell_vec = np.asarray(
                    decision.smell_embedding, dtype=np.float32
                ).copy()
            else:
                self._interrupt_smell_vec = None
            if decision.sensors_embedding is not None:
                self._interrupt_sensors_vec = np.asarray(
                    decision.sensors_embedding, dtype=np.float32
                ).copy()
            else:
                self._interrupt_sensors_vec = None
            if decision.joint_embedding is not None:
                self._interrupt_joint_vec = np.asarray(
                    decision.joint_embedding, dtype=np.float32
                ).copy()
            else:
                self._interrupt_joint_vec = None

    def _clear_stimulus_interrupt(self) -> None:
        cfg = self.stimulus_interrupt
        # Confirmed interrupt: persist joint history so the next matching
        # novelty cue becomes a history_hit (not-novel path).
        patch = self._interrupt_patch_vec
        smell = self._interrupt_smell_vec
        sensors = self._interrupt_sensors_vec
        joint = self._interrupt_joint_vec
        if patch is not None and float(np.linalg.norm(patch)) > 1e-6:
            try:
                self.memory.insert_embedding(
                    embedding=np.zeros(self.embedder.total_dim, dtype=np.float32),
                    signals={
                        "stimulus_interrupt": True,
                        "interrupt_reason": self._interrupt_reason,
                        "cue": (self._interrupt_meta or {}).get("cue"),
                    },
                    affect={},
                    episode_id=self.config.experiment_id,
                    step=int(self._step),
                    sim_time=float(self._step) * 0.05,
                    fly_id=self.config.fly_id,
                    sensors_vec=sensors,
                    patch_vec=patch,
                    smell_vec=smell,
                    joint_vec=joint,
                )
                if self.config.console_log:
                    print(
                        f"si_saved_patch_stm step={self._step} "
                        f"reason={self._interrupt_reason}",
                        flush=True,
                    )
            except Exception as exc:  # pragma: no cover - DB optional in some runs
                if self.config.console_log:
                    print(f"si_save_patch_stm_failed step={self._step} err={exc}", flush=True)
        self._interrupt_pending = False
        self._interrupt_remaining = 0
        self._interrupt_reason = ""
        self._interrupt_meta = {}
        self._interrupt_patch_vec = None
        self._interrupt_smell_vec = None
        self._interrupt_sensors_vec = None
        self._interrupt_joint_vec = None
        self._interrupt_cooldown_until = int(self._step) + int(cfg.cooldown_steps)
        self.conditioning.end_freeze()
        # Natural / early unlock: install last accept blend so motion resumes conditioned.
        if self._resume_goal is not None and self.motor is not None:
            goal = np.asarray(self._resume_goal, dtype=np.float32).copy()
            self.motor.set_goal(goal)
            set_goal = getattr(self.env, "set_goal", None)
            if callable(set_goal):
                set_goal(goal)
            self._pending_action = goal.copy()
        self._resume_goal = None
        # Persist post-freeze conditioning (acks + hits) for later sessions.
        self._persist_conditioning_to_stm()

    def _apply_hard_stop(
        self,
        info: dict[str, Any],
        observation: np.ndarray | None = None,
        *,
        mind_tick: bool = False,
    ) -> np.ndarray:
        """Emit a hard-stop motor action while an interrupt freeze is active."""
        assert self.motor is not None
        self.motor.set_goal(STOP_GOAL.copy())
        set_goal = getattr(self.env, "set_goal", None)
        if callable(set_goal):
            set_goal(STOP_GOAL.copy())
        action = self.motor.act(observation)
        info["stimulus_interrupt"] = True
        info["stimulus_interrupt_pending"] = True
        info["stimulus_interrupt_reason"] = self._interrupt_reason
        info["stimulus_interrupt_meta"] = dict(self._interrupt_meta)
        info["interrupt_remaining"] = int(self._interrupt_remaining)
        info["deep_thought"] = bool(mind_tick)
        info["motor_goal"] = self.motor.goal.tolist()
        info["motor_action"] = action.tolist()
        info["architecture"] = "B"
        info["mind_tick"] = bool(mind_tick)
        return action

    def _tick_interrupt_freeze(self) -> None:
        """Count down deep-thought freeze; clear + save STM when finished."""
        if not self._interrupt_pending:
            return
        self._interrupt_remaining = max(0, int(self._interrupt_remaining) - 1)
        if self._interrupt_remaining <= 0:
            self._clear_stimulus_interrupt()

    def _process_deep_thought(
        self,
        observation: np.ndarray,
        *,
        sim_time: float | None = None,
        left_eye: np.ndarray | None = None,
        right_eye: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Mind tick during interrupt freeze: heighten senses, run devil, no motor."""
        if left_eye is None:
            left_eye = getattr(self.env, "left_eye", None)
        if right_eye is None:
            right_eye = getattr(self.env, "right_eye", None)

        sensory = self.signals.build(observation, left_eye=left_eye, right_eye=right_eye)
        cues = None
        visual_nov = None
        if left_eye is not None and right_eye is not None:
            cues = self._vision_cues(left_eye, right_eye)
            if cues is not None:
                sensory.summary.update(cues.to_dict())
            visual_nov = self.visual_novelty.observe(left_eye, right_eye)
            sensory.summary.update(visual_nov.to_dict())

        bearing_fn = getattr(self.env, "chem_bearings", None)
        if callable(bearing_fn):
            sensory.summary.update(bearing_fn())

        # Map feedback on raw transducers first; reported rates feed deep thought.
        pre_signal = sensory.to_jsonable()
        if cues is not None:
            pre_signal.update(cues.to_dict())
        place_info: dict[str, Any] = {
            "vision_cues": cues.to_dict() if cues else None,
            "_sensory_summary": sensory.summary,
        }
        self._apply_place_map(
            left_eye=left_eye,
            right_eye=right_eye,
            signal_payload=pre_signal,
            info=place_info,
            cues=cues,
            smell_vec=np.asarray(getattr(sensory, "smell_vec", []), dtype=np.float32),
        )

        # Embeddings / STM stay on mapped-reported senses; devil sees heightened copy.
        affect_raw = self.affect.estimate(sensory)
        parts = self.embedder.embed_parts(sensory, affect_raw)
        gain = float(self.stimulus_interrupt.deep_thought_gain)
        sensory_h = heighten_sensory_for_deep_thought(sensory, gain=gain)
        affect_h = self.affect.estimate(sensory_h)
        signal_payload = sensory_h.to_jsonable()
        # Re-apply reported vision cues with deep-thought gain (not raw transducer).
        vision_reported = place_info.get("vision_cues") or {}
        for key, val in vision_reported.items():
            if isinstance(val, (int, float, np.floating)):
                signal_payload[key] = float(val) * gain
            else:
                signal_payload[key] = val
        if visual_nov is not None:
            for key, val in visual_nov.to_dict().items():
                if isinstance(val, (int, float, np.floating)):
                    signal_payload[key] = float(val) * gain
                else:
                    signal_payload[key] = val
        for key in ("sugar_bearing", "bitter_bearing", "hunger", "energy", "proboscis_contact"):
            if key in sensory_h.summary:
                signal_payload[key] = sensory_h.summary[key]
        meta = self._interrupt_meta or {}
        signal_payload["deep_thought"] = True
        signal_payload["deep_thought_gain"] = gain
        signal_payload["stimulus_interrupt"] = True
        signal_payload["interrupt_remaining"] = int(self._interrupt_remaining)
        signal_payload["interrupt_cue"] = meta.get("cue") or ""
        signal_payload["interrupt_reason"] = self._interrupt_reason or ""
        place = place_info.get("place") or {}
        if place.get("place_id") is not None:
            signal_payload["place_id"] = place["place_id"]
            signal_payload["place_mapped"] = bool(
                place.get("mapped", not place.get("is_new", True))
            )
            signal_payload["place_familiarity"] = float(place.get("mean_familiarity", 0.0))
        for channel, reading in (place_info.get("mapped_sensors") or {}).items():
            if isinstance(reading, dict) and "raw" in reading:
                signal_payload[f"{channel}_raw"] = float(reading["raw"])
        self._stamp_conditioning_signals(signal_payload)

        position = None
        pos_fn = getattr(self.env, "agent_position", None)
        if callable(pos_fn):
            position = np.asarray(pos_fn(), dtype=np.float32)
        chem_vec = None
        chem_fn = getattr(self.env, "chem_vector", None)
        if callable(chem_fn):
            chem_vec = np.asarray(chem_fn(), dtype=np.float32)

        curiosity_step = self.curiosity.step(
            embedding=parts.embedding,
            action=self._pending_action,
            signals=signal_payload,
            affect=affect_h.to_jsonable(),
            position=position,
            chem_vector=chem_vec,
        )
        signal_payload["curiosity"] = curiosity_step.curiosity
        explore_bias = float(curiosity_step.explore_bias)
        signal_payload["curiosity_drive"] = explore_bias
        signal_payload["explore_bias"] = explore_bias
        affect_payload = affect_h.to_jsonable()

        stm, thought, thought_rec = self.thinking.think_on_embedding(
            embedding=parts.embedding,
            signals=signal_payload,
            affect=affect_payload,
            episode_id=self.config.experiment_id,
            step=self._step,
            sim_time=float(sim_time if sim_time is not None else self._step * 0.05),
            fly_id=self.config.fly_id,
            left_eye_vec=parts.left_eye_vec,
            right_eye_vec=parts.right_eye_vec,
            sensors_vec=parts.sensors_vec,
            status_vec=parts.status_vec,
            affect_vec=parts.affect_vec,
        )
        self._mind_ticks += 1
        plan_name = None
        if isinstance(thought.intent, dict):
            plan_name = (thought.intent.get("future") or {}).get("chosen_name")
        bore_state = self.boredom.observe_thought(
            plan_name=plan_name,
            refined_thought=thought.refined_thought,
        )
        info: dict[str, Any] = {
            "stm_id": stm.id,
            "thought_id": thought_rec.id,
            "refined_thought": thought.refined_thought,
            "devil_critique": thought.devil_critique,
            "intent": thought.intent,
            "affect": affect_payload,
            "vision_cues": place_info.get("vision_cues")
            if place_info.get("vision_cues") is not None
            else (cues.to_dict() if cues else None),
            "place": place_info.get("place"),
            "mapped_sensors": place_info.get("mapped_sensors"),
            "curiosity": curiosity_step.curiosity,
            "curiosity_gated": curiosity_step.gated_curiosity,
            "curiosity_gate": curiosity_step.gate,
            "explore_bias": explore_bias,
            "coverage": curiosity_step.novelty.coverage,
            "unique_cells": curiosity_step.novelty.unique_cells,
            "exploration_xp": curiosity_step.novelty.exploration_xp,
            "visual_novelty": visual_nov.visual_novelty if visual_nov else None,
            "novelty_bearing": visual_nov.novelty_bearing if visual_nov else None,
            "new_stimulus": visual_nov.new_stimulus if visual_nov else None,
            "boredom": bore_state.boredom,
            "thought_stuck": bore_state.thought_stuck,
            "physical_stuck": bore_state.physical_stuck,
            "force_locomotion": bore_state.force_locomotion,
            "energy": signal_payload.get("energy", sensory.summary.get("energy")),
            "hunger": signal_payload.get("hunger", sensory.summary.get("hunger")),
            "antenna_debug": antenna_debug_payload(sensory_h),
            "deep_thought": True,
            "deep_thought_gain": gain,
            "interrupt_remaining": int(self._interrupt_remaining),
            "stimulus_interrupt": True,
            "stimulus_interrupt_pending": True,
            "stimulus_interrupt_reason": self._interrupt_reason,
            "stimulus_interrupt_meta": dict(self._interrupt_meta),
            "mind_tick": True,
            "maturity": self.experience.maturity(),
        }

        # Judge this deep tick (conditioning observe happens inside _apply_judgement).
        proposal = getattr(self.thinking, "_last_devil_proposal", None)
        l1_goal = goal_from_intent(thought.intent, thought.action)
        judged_goal = self._apply_judgement(
            l1_goal,
            info=info,
            signals=signal_payload,
            affect=affect_payload,
            draft=thought.refined_thought,
            left_eye=left_eye,
            right_eye=right_eye,
            deep_freeze=True,
        )
        judgement = info.get("judgement") if isinstance(info.get("judgement"), dict) else {}
        decision = str(judgement.get("decision") or "ignore")
        prop = judgement.get("proposal") if isinstance(judgement.get("proposal"), dict) else {}
        risk = str(prop.get("risk") or (getattr(proposal, "risk", "none") if proposal else "none"))
        confirm_count = self.conditioning.record_judgement(decision, risk)
        info["judgement_confirm_count"] = int(confirm_count)
        info["conditioning"] = self.conditioning.to_dict()
        self._persist_conditioning_to_stm(info=info)
        cond_summary = self.conditioning.memory_block(include_acknowledged=False)
        info["refined_thought"] = rewrite_present_after_judgement(
            str(info.get("refined_thought") or thought.refined_thought),
            signals=signal_payload,
            affect=affect_payload,
            maturity=float(self.experience.maturity()),
            judgement=judgement,
            conditioning_summary=cond_summary,
        )
        if decision == "accept" and float(judgement.get("blend_weight", 0.0) or 0.0) > 0.0:
            self._resume_goal = np.asarray(judged_goal, dtype=np.float32).copy()

        early_unlock = self.conditioning.confirmed()
        if early_unlock:
            info["stimulus_interrupt_early_unlock"] = True
            info["stimulus_interrupt_pending"] = False
            info["deep_thought"] = False
            # Clear freeze (installs resume goal on accept path via _resume_goal).
            self._clear_stimulus_interrupt()
            if self.motor is not None:
                goal = judged_goal if decision == "accept" else l1_goal
                if decision == "accept":
                    self._resume_goal = np.asarray(goal, dtype=np.float32).copy()
                goal = self._apply_active_investigation(np.asarray(goal, dtype=np.float32), info)
                self.motor.set_goal(goal)
                set_goal = getattr(self.env, "set_goal", None)
                if callable(set_goal):
                    set_goal(goal)
                action = self.motor.act(observation)
                self._motor_ticks += 1
                info["motor_goal"] = goal.tolist()
                info["motor_action"] = action.tolist()
                info["architecture"] = "B"
                info["mind_tick"] = True
                info["stimulus_interrupt"] = False
                self._pending_action = action.copy()
            else:
                action = (
                    np.asarray(judged_goal, dtype=np.float32).copy()
                    if decision == "accept"
                    else STOP_GOAL.copy()
                )
                info["architecture"] = "A"
                info["mind_tick"] = True
                info["motor_action"] = action.tolist()
                info["stimulus_interrupt"] = False
                self._pending_action = action.copy()
            bore_phys = self.boredom.observe_physical(position=position, action=action)
            info["boredom"] = bore_phys.boredom
            info["thought_stuck"] = bore_phys.thought_stuck
            info["physical_stuck"] = bore_phys.physical_stuck
            info["force_locomotion"] = bore_phys.force_locomotion
            self._last_mind_info = dict(info)
            return action, info

        if self.motor is not None:
            action = self._apply_hard_stop(info, observation, mind_tick=True)
            self._motor_ticks += 1
        else:
            action = STOP_GOAL.copy()
            info["architecture"] = "A"
            info["mind_tick"] = True
            info["motor_action"] = action.tolist()
        bore_phys = self.boredom.observe_physical(position=position, action=action)
        info["boredom"] = bore_phys.boredom
        info["thought_stuck"] = bore_phys.thought_stuck
        info["physical_stuck"] = bore_phys.physical_stuck
        info["force_locomotion"] = bore_phys.force_locomotion
        self._last_mind_info = dict(info)
        self._pending_action = action.copy()
        return action, info

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
        self.visual_thinker.close()
        llm_close = getattr(self.thinking.llm, "close", None)
        if callable(llm_close):
            llm_close()
        pg_close = getattr(self.place_graph, "close", None)
        if callable(pg_close):
            pg_close()
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def _signals_after_mapped_feedback(
        self,
        sensory: Any,
        *,
        visual_nov: Any | None,
        place_info: dict[str, Any],
    ) -> dict[str, Any]:
        """Rebuild signal payload from sensory summary after mapped-sensor feedback."""
        signal_payload = sensory.to_jsonable() if hasattr(sensory, "to_jsonable") else dict(
            getattr(sensory, "summary", {}) or {}
        )
        if place_info.get("vision_cues"):
            signal_payload.update(place_info["vision_cues"])
        if visual_nov is not None and hasattr(visual_nov, "to_dict"):
            signal_payload.update(visual_nov.to_dict())
        for key in ("sugar_bearing", "bitter_bearing", "hunger", "energy", "proboscis_contact"):
            if key in getattr(sensory, "summary", {}):
                signal_payload[key] = sensory.summary[key]
        place = place_info.get("place") or {}
        if place.get("place_id") is not None:
            signal_payload["place_id"] = place["place_id"]
            signal_payload["place_mapped"] = bool(
                place.get("mapped", not place.get("is_new", True))
            )
            signal_payload["place_familiarity"] = float(place.get("mean_familiarity", 0.0))
        if "wall_hazard_raw" in place:
            signal_payload["wall_hazard_raw"] = place["wall_hazard_raw"]
        for channel, reading in (place_info.get("mapped_sensors") or {}).items():
            if isinstance(reading, dict):
                signal_payload[channel] = reading.get("reported")
                signal_payload[f"{channel}_raw"] = reading.get("raw")
        return signal_payload

    def _apply_place_map(
        self,
        *,
        left_eye: np.ndarray | None,
        right_eye: np.ndarray | None,
        signal_payload: dict[str, Any],
        info: dict[str, Any],
        cues: VisionNavCues | None = None,
        smell_vec: np.ndarray | None = None,
    ) -> MappedSensorReport | None:
        """Sensor feedback: map place in Neo4j, report known items at decayed rates."""
        cfg = self.place_graph_config
        if self.mapped_sensors is None or not cfg.enabled:
            return None
        if int(self._step) % max(1, int(cfg.update_every)) != 0:
            return None

        raw: dict[str, float] = {}
        if cues is not None:
            raw["wall_hazard"] = float(cues.wall_hazard)
        elif "wall_hazard" in signal_payload:
            raw["wall_hazard"] = float(signal_payload.get("wall_hazard", 0.0) or 0.0)
        for channel in ("odor", "odor_left", "odor_right", "bitter", "sugar"):
            if channel in signal_payload:
                raw[channel] = float(signal_payload.get(channel, 0.0) or 0.0)

        view = None
        if left_eye is not None and right_eye is not None:
            view = stereo_patch_embedding(left_eye, right_eye)

        smell = smell_vec
        if smell is None and "smell_vec" in signal_payload:
            smell = np.asarray(signal_payload.get("smell_vec"), dtype=np.float32)

        report = self.mapped_sensors.sense(
            raw=raw,
            view_vec=view,
            smell_vec=smell,
            step=int(self._step),
            now_ms=float(info.get("now_ms") or self._step),
        )
        for channel, reading in report.readings.items():
            signal_payload[channel] = float(reading.reported)
            signal_payload[f"{channel}_raw"] = float(reading.raw)
        if report.place is not None:
            signal_payload["place_id"] = report.place.id
            signal_payload["place_mapped"] = not report.is_new
            signal_payload["place_familiarity"] = float(report.mean_familiarity)
            signal_payload["place_label"] = str(
                (report.place.props or {}).get("label") or report.label
            )

        place_info = report.to_place_info()
        info["place"] = place_info
        info["mapped_sensors"] = {k: v.to_dict() for k, v in report.readings.items()}

        vision = info.get("vision_cues")
        if isinstance(vision, dict) and "wall_hazard" in report.readings:
            vision = dict(vision)
            wall = report.readings["wall_hazard"]
            vision["wall_hazard"] = float(wall.reported)
            vision["wall_hazard_raw"] = float(wall.raw)
            info["vision_cues"] = vision

        # Feedback into the live sensory summary so affect/thinking see reported rates.
        summary = info.get("_sensory_summary")
        if isinstance(summary, dict):
            for channel, reading in report.readings.items():
                summary[channel] = float(reading.reported)
                summary[f"{channel}_raw"] = float(reading.raw)
        return report

    def _vision_cues(self, left: np.ndarray, right: np.ndarray) -> VisionNavCues | None:
        if not self.config.use_vision_head:
            return None
        if self._vision_trained and self.vision_head is not None:
            return self.vision_head.infer_cues(left, right)
        if self.config.use_heuristic_if_untrained:
            return heuristic_cues_from_frames(left, right)
        return None

    def _apply_judgement(
        self,
        l1_goal: np.ndarray,
        *,
        info: dict[str, Any],
        signals: dict[str, Any],
        affect: dict[str, Any],
        draft: str | None = None,
        left_eye: np.ndarray | None = None,
        right_eye: np.ndarray | None = None,
        deep_freeze: bool = False,
    ) -> np.ndarray:
        """Trait-weighted judgement of devil proposal → accept-blend / reject / ignore / reconsider."""
        self.judgement.advance_tick(self._step)
        if not deep_freeze:
            # Non-deep: allow previously acknowledged hazards to surface again.
            self.conditioning.begin_judgement_cycle()
        proposal = getattr(self.thinking, "_last_devil_proposal", None)
        if proposal is None:
            raw = ((info.get("intent") or {}).get("future") or {}).get("devil_proposal")
            if isinstance(raw, dict):
                proposal = DevilProposal(
                    risk=str(raw.get("risk", "none")),
                    forward_scale=float(raw.get("forward_scale", 1.0)),
                    turn_delta=float(raw.get("turn_delta", 0.0)),
                    confidence=float(raw.get("confidence", 0.0)),
                    reason=str(raw.get("reason", "")),
                    ran=bool(raw.get("ran", True)),
                    source=str(raw.get("source", "intent")),
                )
        if proposal is None:
            info["judgement"] = {"decision": "ignore", "reason": "no_proposal", "blend_weight": 0.0}
            info["personality"] = self.personality.as_dict()
            return l1_goal

        if proposal.risk not in ("none", "", None):
            self.conditioning.observe(
                str(proposal.risk),
                confidence=float(proposal.confidence or 0.0),
                reason=str(proposal.reason or ""),
                signals=signals,
            )

        traits = self.personality.snapshot()
        judge_signals = dict(signals)
        judge_signals.setdefault("survival_risk", float(affect.get("survival_risk", 0.0)))
        judge_signals.update(self.conditioning.to_signals())
        kind_hint = (
            proposal.risk
            if proposal.risk in ("wall", "toxin", "odor", "fall", "energy")
            else "novelty"
        )
        stimuli_sig = {
            **{
                k: signals.get(k)
                for k in (
                    "wall_hazard",
                    "visual_novelty",
                    "novelty_bearing",
                    "bitter",
                    "odor",
                    "energy",
                    "sugar",
                )
            },
            "survival_risk": judge_signals.get("survival_risk"),
        }
        eye_match, matched = self.interests.max_eye_similarity(
            left_eye,
            right_eye,
            kind=kind_hint,
            stimuli_signature=stimuli_sig,
        )
        info["eye_match"] = float(eye_match)
        if matched is not None:
            info["eye_match_interest_id"] = matched.id

        result = self.judgement.evaluate(
            proposal,
            traits=traits,
            signals=judge_signals,
            reconsider_used=0,
            eye_match_score=eye_match,
        )
        if result.decision == "reconsider":
            prior = proposal.reason or result.reason
            redraft = draft or str(info.get("refined_thought") or "reconsider plan")
            proposal = propose_critique(
                self.thinking.llm,
                draft=redraft,
                affect=affect,
                signals=signals,
                prior_reason=prior,
            )
            self.thinking._last_devil_proposal = proposal
            result = self.judgement.evaluate(
                proposal,
                traits=traits,
                signals=judge_signals,
                reconsider_used=1,
                eye_match_score=eye_match,
            )
            if result.decision == "reconsider":
                result.decision = "ignore"
                result.reason = "reconsider_budget_exhausted"
                result.blend_weight = 0.0

        if result.decision == "reject":
            self.judgement.record_reject(proposal)

        if result.decision == "acknowledge":
            bearing = float(signals.get("novelty_bearing", signals.get("target_bearing", 0.0)) or 0.0)
            if abs(bearing) < 0.05 and abs(proposal.turn_delta) > 0.05:
                bearing = float(np.clip(proposal.turn_delta, -1.0, 1.0))
            novelty = float(signals.get("visual_novelty", signals.get("novelty", 0.0)) or 0.0)
            if bool(signals.get("new_stimulus")):
                novelty = max(novelty, 0.55)
            if proposal.risk in ("wall", "toxin", "odor", "fall"):
                novelty = max(novelty, 0.35)
            kind = (
                proposal.risk
                if proposal.risk in ("wall", "toxin", "odor", "fall", "energy")
                else "novelty"
            )
            point = self.interests.acknowledge(
                stm_id=int(info.get("stm_id") or 0),
                step=int(self._step),
                tick=int(self._step),
                kind=kind,
                bearing=bearing,
                novelty=novelty,
                reason=proposal.reason or result.reason,
                fly_id=self.config.fly_id,
                proposal_risk=str(proposal.risk),
                left_eye=left_eye,
                right_eye=right_eye,
                stimuli_signature=stimuli_sig,
            )
            info["interest_point"] = point.to_dict() if point is not None else None
            self.conditioning.acknowledge(str(proposal.risk))
            # Re-run retrospective so a just-acknowledged point can start investigation
            # this tick when survival/toxin load allows.
            retro = self.thinking.maybe_retrospective(
                signals=signals,
                affect=affect,
                fly_id=self.config.fly_id,
                tick=self._step,
            )
            if retro.investigations:
                info["investigation"] = retro.investigations[0].to_dict()
                info["retrospective_ran"] = True
                info["retrospective_reason"] = retro.reason

        goal = l1_goal
        if result.decision == "accept" and result.blend_weight > 0.0:
            proposed = apply_proposal(l1_goal, proposal)
            goal = blend_goals(l1_goal, proposed, result.blend_weight)

        info["judgement"] = result.to_dict()
        info["personality"] = self.personality.as_dict()
        info["devil_proposal"] = proposal.to_dict()
        info["interest_pending"] = len(self.interests.pending())
        info["conditioning"] = self.conditioning.to_dict()
        if not deep_freeze:
            cond_summary = self.conditioning.memory_block(include_acknowledged=False)
            info["refined_thought"] = rewrite_present_after_judgement(
                str(info.get("refined_thought") or draft or ""),
                signals=judge_signals,
                affect=affect,
                maturity=float(info.get("maturity") or self.experience.maturity()),
                judgement=info["judgement"],
                conditioning_summary=cond_summary,
            )
            self._persist_conditioning_to_stm(info=info)
        return goal

    def _apply_active_investigation(self, goal: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        """Advance / apply any in-flight deferred investigation motor bias."""
        active = self.interests.advance_tick(self._step)
        if active is None:
            active = self.interests.active
        if active is None:
            return goal
        info["investigation"] = active.to_dict()
        info["interest_pending"] = len(self.interests.pending())
        return apply_investigation_goal(goal, active)

    def _visual_think(
        self,
        goal: np.ndarray,
        *,
        info: dict[str, Any],
        left_eye: np.ndarray | None,
        right_eye: np.ndarray | None,
        visual_nov: Any,
        wall_hazard: float,
        survival_risk: float,
    ) -> np.ndarray:
        """Novelty → low-load vision-LLM thinking; bias motor goal to inspect/approach."""
        self.visual_thinker.poll()
        if visual_nov is not None:
            gate = self.visual_thinker.maybe_think(
                tick=self._step,
                visual_novelty=float(visual_nov.visual_novelty),
                new_stimulus=bool(visual_nov.new_stimulus),
                wall_hazard=float(wall_hazard),
                survival_risk=float(survival_risk),
                left_eye=left_eye,
                right_eye=right_eye,
                bearing=float(visual_nov.novelty_bearing),
            )
            info["visual_think_gate"] = gate
        goal = self.visual_thinker.bias_goal(
            goal, tick=self._step, wall_hazard=float(wall_hazard)
        )
        info["visual_thinking"] = self.visual_thinker.is_pending
        insight = self.visual_thinker.last_insight
        if insight is not None:
            info["visual_insight"] = insight.to_dict()
        return goal

    def process_observation(
        self,
        observation: np.ndarray,
        *,
        sim_time: float | None = None,
        left_eye: np.ndarray | None = None,
        right_eye: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self._interrupt_pending and self._interrupt_remaining > 0:
            return self._process_deep_thought(
                observation,
                sim_time=sim_time,
                left_eye=left_eye,
                right_eye=right_eye,
            )
        if left_eye is None:
            left_eye = getattr(self.env, "left_eye", None)
        if right_eye is None:
            right_eye = getattr(self.env, "right_eye", None)

        sensory = self.signals.build(observation, left_eye=left_eye, right_eye=right_eye)
        cues = None
        visual_nov = None
        if left_eye is not None and right_eye is not None:
            cues = self._vision_cues(left_eye, right_eye)
            if cues is not None:
                sensory.summary.update(cues.to_dict())
            visual_nov = self.visual_novelty.observe(left_eye, right_eye)
            sensory.summary.update(visual_nov.to_dict())

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

        # Sensor feedback before affect: Neo4j-mapped items report at decayed rates.
        signal_payload = sensory.to_jsonable()
        if cues is not None:
            signal_payload.update(cues.to_dict())
        if visual_nov is not None:
            signal_payload.update(visual_nov.to_dict())
        for key in ("sugar_bearing", "bitter_bearing", "hunger", "energy", "proboscis_contact"):
            if key in sensory.summary:
                signal_payload[key] = sensory.summary[key]
        place_info: dict[str, Any] = {
            "vision_cues": cues.to_dict() if cues else None,
            "_sensory_summary": sensory.summary,
        }
        self._apply_place_map(
            left_eye=left_eye,
            right_eye=right_eye,
            signal_payload=signal_payload,
            info=place_info,
            cues=cues,
            smell_vec=np.asarray(getattr(sensory, "smell_vec", []), dtype=np.float32),
        )
        signal_payload = self._signals_after_mapped_feedback(
            sensory, visual_nov=visual_nov, place_info=place_info
        )

        affect = self.affect.estimate(sensory)
        parts = self.embedder.embed_parts(sensory, affect)
        embedding = parts.embedding

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
        explore_bias = float(curiosity_step.explore_bias)
        if visual_nov is not None and visual_nov.visual_novelty > 0.15:
            explore_bias = float(
                np.clip(explore_bias + 0.55 * visual_nov.visual_novelty, 0.0, 1.0)
            )
        signal_payload["curiosity_drive"] = explore_bias
        signal_payload["explore_bias"] = explore_bias
        signal_payload["novelty"] = curiosity_step.novelty.novelty_score
        signal_payload["exploration_xp"] = curiosity_step.novelty.exploration_xp
        # Boredom from prior stuck thoughts/body (stimulus-agnostic) biases L1 this tick.
        bore_pre = self.boredom.state()
        signal_payload["boredom"] = bore_pre.boredom
        signal_payload["force_locomotion"] = bore_pre.force_locomotion
        # Soft-bump affect exploration/novelty for thinking when gate allows.
        affect_payload = affect.to_jsonable()
        if curiosity_step.gate > 0.05 or (visual_nov is not None and visual_nov.visual_novelty > 0.15):
            viz_boost = float(visual_nov.visual_novelty) if visual_nov is not None else 0.0
            affect_payload["exploration"] = float(
                np.clip(
                    affect_payload.get("exploration", 0.0)
                    + 0.35 * explore_bias
                    + 0.4 * viz_boost,
                    0.0,
                    1.0,
                )
            )
            affect_payload["novelty"] = float(
                np.clip(
                    affect_payload.get("novelty", 0.0)
                    + 0.4 * curiosity_step.novelty.novelty_score
                    + 0.45 * viz_boost,
                    0.0,
                    1.0,
                )
            )

        self._stamp_conditioning_signals(signal_payload)
        stm, thought, thought_rec = self.thinking.think_on_embedding(
            embedding=embedding,
            signals=signal_payload,
            affect=affect_payload,
            episode_id=self.config.experiment_id,
            step=self._step,
            sim_time=float(sim_time if sim_time is not None else self._step * 0.05),
            fly_id=self.config.fly_id,
            left_eye_vec=parts.left_eye_vec,
            right_eye_vec=parts.right_eye_vec,
            sensors_vec=parts.sensors_vec,
            status_vec=parts.status_vec,
            affect_vec=parts.affect_vec,
        )
        mind_action = self.curiosity.apply_action_bias(thought.action, explore_bias)
        self._mind_ticks += 1
        plan_name = None
        if isinstance(thought.intent, dict):
            plan_name = (thought.intent.get("future") or {}).get("chosen_name")
        bore_state = self.boredom.observe_thought(
            plan_name=plan_name,
            refined_thought=thought.refined_thought,
        )
        retro = self.thinking._last_retrospective
        info = {
            "stm_id": stm.id,
            "thought_id": thought_rec.id,
            "refined_thought": thought.refined_thought,
            "devil_critique": thought.devil_critique,
            "intent": thought.intent,
            "affect": affect_payload,
            "vision_cues": place_info.get("vision_cues")
            if place_info.get("vision_cues") is not None
            else (cues.to_dict() if cues else None),
            "place": place_info.get("place"),
            "mapped_sensors": place_info.get("mapped_sensors"),
            "maturity": self.experience.maturity(),
            "priority_weights": (thought.intent.get("future") or {}).get("priority_weights"),
            "retrospective_ran": bool(retro.ran) if retro else False,
            "labeled_outcomes": self.experience.labeled_outcomes,
            "curiosity": curiosity_step.curiosity,
            "curiosity_gated": curiosity_step.gated_curiosity,
            "curiosity_gate": curiosity_step.gate,
            "explore_bias": explore_bias,
            "coverage": curiosity_step.novelty.coverage,
            "unique_cells": curiosity_step.novelty.unique_cells,
            "exploration_xp": curiosity_step.novelty.exploration_xp,
            "novelty_first_visit": curiosity_step.novelty.first_visit,
            "visual_novelty": visual_nov.visual_novelty if visual_nov else None,
            "novelty_bearing": visual_nov.novelty_bearing if visual_nov else None,
            "new_stimulus": visual_nov.new_stimulus if visual_nov else None,
            "boredom": bore_state.boredom,
            "thought_stuck": bore_state.thought_stuck,
            "physical_stuck": bore_state.physical_stuck,
            "force_locomotion": bore_state.force_locomotion,
            "tendency": (thought.intent.get("future") or {}).get("tendency"),
            "mind_tick": True,
            "energy": signal_payload.get("energy", sensory.summary.get("energy")),
            "hunger": signal_payload.get("hunger", sensory.summary.get("hunger")),
            "antenna_debug": antenna_debug_payload(sensory),
        }
        self._last_mind_info = dict(info)

        if self.motor is not None:
            # Prefer L1/L2 plan as the motor goal; curiosity may still nudge it.
            raw_goal = goal_from_intent(thought.intent, mind_action)
            goal = self.curiosity.apply_action_bias(raw_goal, explore_bias)
            goal = self._apply_judgement(
                goal,
                info=info,
                signals=signal_payload,
                affect=affect_payload,
                draft=thought.refined_thought,
                left_eye=left_eye,
                right_eye=right_eye,
            )
            if visual_nov is not None and str(plan_name) in (
                "explore",
                "court",
                "forward",
                "turn_left",
                "turn_right",
            ):
                wall = float(signal_payload.get("wall_hazard", 0.0))
                goal = apply_visual_explore_goal(
                    goal,
                    visual_nov,
                    config=self.visual_novelty.config,
                    wall_hazard=wall,
                )
            goal = apply_boredom_escape_goal(
                goal,
                bore_state,
                config=self.boredom.config,
                escape_sign=self.boredom._escape_sign,
            )
            goal = self._visual_think(
                goal,
                info=info,
                left_eye=left_eye,
                right_eye=right_eye,
                visual_nov=visual_nov,
                wall_hazard=float(signal_payload.get("wall_hazard", 0.0)),
                survival_risk=float(affect_payload.get("survival_risk", 0.0)),
            )
            goal = self._apply_active_investigation(goal, info)
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
        bore_state = self.boredom.observe_physical(position=position, action=action)
        info["boredom"] = bore_state.boredom
        info["thought_stuck"] = bore_state.thought_stuck
        info["physical_stuck"] = bore_state.physical_stuck
        info["force_locomotion"] = bore_state.force_locomotion
        self._last_mind_info = dict(info)
        self._pending_action = action.copy()
        return action, info

    def process_motor_only(self, observation: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Motor tick without a new mind thought (architecture B)."""
        info = dict(self._last_mind_info)
        info["mind_tick"] = False
        # Keep novelty / curiosity ticking so exploration metrics stay live during motor steps.
        left_eye = getattr(self.env, "left_eye", None)
        right_eye = getattr(self.env, "right_eye", None)
        sensory = self.signals.build(observation, left_eye=left_eye, right_eye=right_eye)
        affect = self.affect.estimate(sensory)
        embedding = self.embedder.embed(sensory, affect)
        info["antenna_debug"] = antenna_debug_payload(sensory)
        position = None
        pos_fn = getattr(self.env, "agent_position", None)
        if callable(pos_fn):
            position = np.asarray(pos_fn(), dtype=np.float32)
        chem_vec = None
        chem_fn = getattr(self.env, "chem_vector", None)
        if callable(chem_fn):
            chem_vec = np.asarray(chem_fn(), dtype=np.float32)
        visual_nov = None
        if left_eye is not None and right_eye is not None:
            visual_nov = self.visual_novelty.observe(left_eye, right_eye)
            info["visual_novelty"] = visual_nov.visual_novelty
            info["novelty_bearing"] = visual_nov.novelty_bearing
            info["new_stimulus"] = visual_nov.new_stimulus
            cues = self._vision_cues(left_eye, right_eye)
            if cues is not None:
                signal_payload = dict(cues.to_dict())
                sensory_json = (
                    sensory.to_jsonable() if hasattr(sensory, "to_jsonable") else sensory.summary
                )
                for channel in ("odor", "odor_left", "odor_right", "bitter", "sugar"):
                    if channel in sensory_json:
                        signal_payload[channel] = float(sensory_json[channel] or 0.0)
                if not isinstance(info.get("vision_cues"), dict):
                    info["vision_cues"] = dict(cues.to_dict())
                self._apply_place_map(
                    left_eye=left_eye,
                    right_eye=right_eye,
                    signal_payload=signal_payload,
                    info=info,
                    cues=cues,
                    smell_vec=np.asarray(getattr(sensory, "smell_vec", []), dtype=np.float32),
                )
                if info.get("place"):
                    self._last_mind_info["place"] = info["place"]
                    self._last_mind_info["vision_cues"] = info.get("vision_cues")
                    if info.get("mapped_sensors"):
                        self._last_mind_info["mapped_sensors"] = info["mapped_sensors"]
                # Keep motor path vision wall in sync with reported sensor.
                wall_r = signal_payload.get("wall_hazard")
                if wall_r is not None and isinstance(info.get("vision_cues"), dict):
                    info["vision_cues"]["wall_hazard"] = float(wall_r)

        # Candidate novel-vs-STM → arm interrupt (hard-stop freeze + deep thought).
        self._arm_stimulus_interrupt(
            observation=observation,
            left_eye=left_eye,
            right_eye=right_eye,
            sensory=sensory,
            affect=affect,
            visual_nov=visual_nov,
            info=info,
        )
        if self._interrupt_pending and self.motor is not None:
            action = self._apply_hard_stop(info, observation, mind_tick=False)
            self._motor_ticks += 1
            bore_state = self.boredom.observe_physical(position=position, action=action)
            info["boredom"] = bore_state.boredom
            info["thought_stuck"] = bore_state.thought_stuck
            info["physical_stuck"] = bore_state.physical_stuck
            info["force_locomotion"] = bore_state.force_locomotion
            self._pending_action = action.copy()
            return action, info

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
            bore_state = self.boredom.observe_physical(position=position, action=action)
            info["boredom"] = bore_state.boredom
            info["thought_stuck"] = bore_state.thought_stuck
            info["physical_stuck"] = bore_state.physical_stuck
            info["force_locomotion"] = bore_state.force_locomotion
            self._pending_action = action.copy()
            return action, info
        bore_pre = self.boredom.state()
        plan_name = None
        intent = info.get("intent") if isinstance(info.get("intent"), dict) else {}
        plan_name = (intent.get("future") or {}).get("chosen_name")
        if visual_nov is not None and str(plan_name) in (
            "explore",
            "court",
            "forward",
            "turn_left",
            "turn_right",
        ):
            wall = float((info.get("vision_cues") or {}).get("wall_hazard", 0.0) or 0.0)
            self.motor.set_goal(
                apply_visual_explore_goal(
                    self.motor.goal,
                    visual_nov,
                    config=self.visual_novelty.config,
                    wall_hazard=wall,
                )
            )
        if bore_pre.force_locomotion:
            escaped = apply_boredom_escape_goal(
                self.motor.goal,
                bore_pre,
                config=self.boredom.config,
                escape_sign=self.boredom._escape_sign,
            )
            self.motor.set_goal(escaped)
        biased = self._visual_think(
            self.motor.goal,
            info=info,
            left_eye=left_eye,
            right_eye=right_eye,
            visual_nov=visual_nov,
            wall_hazard=float((info.get("vision_cues") or {}).get("wall_hazard", 0.0) or 0.0),
            survival_risk=float((info.get("affect") or {}).get("survival_risk", 0.0)),
        )
        biased = self._apply_active_investigation(biased, info)
        self.motor.set_goal(biased)
        action = self.motor.act(observation)
        self._motor_ticks += 1
        info["motor_goal"] = self.motor.goal.tolist()
        info["motor_action"] = action.tolist()
        info["motor_tracking_error"] = self.motor.tracking_error()
        info["architecture"] = "B"
        bore_state = self.boredom.observe_physical(position=position, action=action)
        info["boredom"] = bore_state.boredom
        info["thought_stuck"] = bore_state.thought_stuck
        info["physical_stuck"] = bore_state.physical_stuck
        info["force_locomotion"] = bore_state.force_locomotion
        self._last_mind_info["boredom"] = bore_state.boredom
        self._last_mind_info["thought_stuck"] = bore_state.thought_stuck
        self._last_mind_info["physical_stuck"] = bore_state.physical_stuck
        self._last_mind_info["force_locomotion"] = bore_state.force_locomotion
        if visual_nov is not None:
            self._last_mind_info["visual_novelty"] = visual_nov.visual_novelty
            self._last_mind_info["novelty_bearing"] = visual_nov.novelty_bearing
            self._last_mind_info["new_stimulus"] = visual_nov.new_stimulus
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
        self.boredom.reset()
        self.visual_novelty.reset()
        self.visual_thinker.reset()
        self.judgement.reset()
        self.interests.reset()
        self.conditioning.reset()
        self._hydrate_conditioning_from_stm()
        self._history_baseline.reset()
        self._patch_history.reset()
        self._passive_smell.reset()
        self._interrupt_pending = False
        self._interrupt_remaining = 0
        self._interrupt_reason = ""
        self._interrupt_meta = {}
        self._interrupt_patch_vec = None
        self._interrupt_smell_vec = None
        self._interrupt_sensors_vec = None
        self._interrupt_joint_vec = None
        self._interrupt_cooldown_until = 0
        self._resume_goal = None
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
        deep_interval = max(
            1, int(getattr(self.stimulus_interrupt, "deep_thought_mind_interval", 4) or 4)
        )
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
            think_interval = deep_interval if self._interrupt_pending else interval
            should_think = (
                (self.motor is None)
                or (self._step % think_interval == 0)
                or not self._last_mind_info
            )
            if should_think and self._interrupt_pending:
                if self.config.console_log:
                    print(
                        f"live_mind_deep_thought step={self._step} "
                        f"remaining={self._interrupt_remaining} "
                        f"reason={self._interrupt_reason}",
                        flush=True,
                    )
                action, info = self._process_deep_thought(obs)
            elif should_think:
                if self.config.console_log and self._step == 0:
                    print("live_mind_thinking step=0 …", flush=True)
                action, info = self.process_observation(obs)
            else:
                action, info = self.process_motor_only(obs)
            # Blend food-search heuristic into the *goal* path when mind just thought.
            food_policy = getattr(self.env, "food_policy_action", None)
            if (
                callable(food_policy)
                and info.get("mind_tick")
                and not info.get("stimulus_interrupt")
            ):
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
            if (
                callable(vision_policy)
                and info.get("mind_tick")
                and not info.get("stimulus_interrupt")
            ):
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
                and not info.get("stimulus_interrupt")
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
            self._tick_interrupt_freeze()
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
