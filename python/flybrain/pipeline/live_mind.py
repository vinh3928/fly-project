"""Live multimodal mind pipeline: observe → embed → STM → think → act."""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import yaml

from flybrain.affect.estimator import AffectEstimator
from flybrain.curiosity import CuriosityConfig, CuriosityDrive
from flybrain.curiosity.boredom import (
    BoredomConfig,
    BoredomTracker,
    apply_blocked_escape_goal,
    apply_boredom_escape_goal,
)
from flybrain.env import FakeFlyEnv
from flybrain.memory.embedder import MindEmbedder
from flybrain.memory.grounded_phrase import (
    GroundedPhrase,
    GroundedPhraseConfig,
    GroundedSentence,
    match_grounded_phrase,
    match_grounded_sentence,
    match_top_grounded_phrases,
    open_grounded_phrase_store,
)
from flybrain.memory.pgvector import ShortTermMemory, open_short_term_memory
from flybrain.memory.place_graph import PlaceGraphConfig, open_place_graph
from flybrain.memory.skill_cortex import (
    SkillCortexConfig,
    SkillProgram,
    open_skill_cortex,
)
from flybrain.memory.speech_cortex import (
    SpeechContext,
    SpeechCortexConfig,
    SpeechImage,
    build_babble_utterance,
    build_hear_utterance,
    open_speech_cortex,
    reinforce_wordlike_babble,
)
from flybrain.motor import (
    MotorActController,
    MotorLayer,
    MotorLayerConfig,
    SkillExecutor,
    babble_gate,
    build_candidate_skill,
    goal_from_intent,
    make_directive,
    mode_for_plan,
    orient_goal,
    sample_babble_program,
    score_babble_episode,
)
from flybrain.motor.skill_babble import signals_for_gate
from flybrain.phase9 import DEFAULT_MIND_INTERVAL
from flybrain.pipeline.async_mind import (
    MIND_JOB_DEEP,
    MIND_JOB_THINK,
    AsyncMindScheduler,
    FairLock,
    MindJobResult,
    SerializedAccess,
)
from flybrain.sensors.mapped_feedback import MappedSensorFeedback, MappedSensorReport
from flybrain.signals.builder import SignalBuilder
from flybrain.speech import babble as speech_babble
from flybrain.speech.articulatory import clip_audio_bands
from flybrain.speech.babble import expand_controls, sample_babble_utterance
from flybrain.speech.executor import SpeechExecutor
from flybrain.thinking.composer import FEEDING_UNTIL_SATED_KEY
from flybrain.thinking.conditioning import (
    ConditioningConfig,
    HazardConditioning,
    conditioning_snapshot_signals,
    load_conditioning_from_stm,
    rewrite_present_after_judgement,
)
from flybrain.thinking.devils_advocate import propose_critique
from flybrain.thinking.father_advocate import (
    FatherAdvocateConfig,
    FatherLesson,
    FatherLessonStore,
    FatherQuestion,
    FatherReply,
    answer_father_question,
    blend_goal_with_father,
    directive_fields_from_father_lesson,
    lesson_from_reply,
    make_father_client,
    propose_father_question,
)
from flybrain.thinking.interest import (
    InterestConfig,
    InterestPointStore,
    apply_investigation_goal,
)
from flybrain.thinking.judgement import (
    DevilProposal,
    JudgementConfig,
    JudgementModule,
    apply_proposal,
    blend_goals,
)
from flybrain.thinking.personality import PersonalityState
from flybrain.thinking.phrase_labeler import PhraseGroundingService, make_phrase_client
from flybrain.thinking.retrospective import RetrospectiveConfig
from flybrain.thinking.route_home import RouteHomeConfig, RouteHomeTracker
from flybrain.thinking.sentence_composer import (
    SentenceComposeService,
    drive_snapshot,
)
from flybrain.thinking.service import ThinkingConfig, ThinkingService
from flybrain.thinking.stimulus_interrupt import (
    STATUS_REFRAME_CUES,
    STOP_GOAL,
    PassiveSmellGate,
    PatchGridHistoryCache,
    RelativeHistoryBaseline,
    StatusReframeTracker,
    StimulusInterruptConfig,
    evaluate_stimulus_interrupt,
    heighten_sensory_for_deep_thought,
)
from flybrain.thinking.sugar_search import (
    SugarSearchConfig,
    SugarSearchTracker,
)
from flybrain.thinking.visual_thinker import VisualThinker, VisualThinkerConfig
from flybrain.vision.novelty import (
    VisualNoveltyConfig,
    VisualNoveltyTracker,
    apply_visual_explore_goal,
    stereo_patch_embedding,
)
from flybrain.vision.recurrent_head import (
    IncompatibleVisionCheckpoint,
    RecurrentVisionHead,
    VisionNavCues,
    heuristic_cues_from_frames,
    load_vision_head,
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
    sugar_smell = info.get("sugar_smell")
    sniff_s = ""
    if sugar_smell is not None:
        search = info.get("sugar_search") or {}
        search_s = ""
        if search.get("turn") is not None:
            search_s = f",{search.get('mode')}={float(search['turn']):+.2f}"
        orient = info.get("sugar_orient")
        if orient is not None:
            mode = info.get("sugar_orient_mode") or "orient"
            search_s += f",{mode}=[{float(orient[0]):+.2f},{float(orient[1]):+.2f}"
            if len(orient) > 2:
                search_s += f",p{float(orient[2]):.1f}"
            search_s += "]"
        sniff_s = (
            f" sniff(sugar={float(sugar_smell):.3f}"
            f",bear={float(info.get('sugar_bearing') or 0.0):+.2f}{search_s})"
        )
    if info.get("proboscis_extension") is not None:
        sniff_s += (
            f" prob(ext={float(info['proboscis_extension']):.2f}"
            f",touch={float(info.get('proboscis_contact') or 0.0):.0f}"
            f",taste={float(info.get('taste_sugar') or 0.0):.2f})"
        )
    cov = info.get("coverage")
    cov_s = f" cov={float(cov):.3f}" if cov is not None else ""

    intent = info.get("intent") or {}
    future = intent.get("future") if isinstance(intent, dict) else None
    chosen = ""
    l1_s = ""
    if isinstance(future, dict) and future.get("chosen_name"):
        chosen = f" plan={future.get('chosen_name')}"
        layer1 = future.get("layer1_values") or []
        chosen_name = str(future.get("chosen_name") or "")
        if isinstance(layer1, list) and layer1:
            top = []
            seen: set[str] = set()
            for row in layer1[:3]:
                if not isinstance(row, dict):
                    continue
                name = str(row.get("name") or "")
                top.append(f"{name}={float(row.get('value', 0.0)):.2f}")
                seen.add(name)
            # Always surface the chosen plan's score (feed often ranks below explore raw).
            if chosen_name and chosen_name not in seen:
                for row in layer1:
                    if isinstance(row, dict) and str(row.get("name") or "") == chosen_name:
                        top.append(f"{chosen_name}={float(row.get('value', 0.0)):.2f}")
                        break
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
        route = info.get("route_home") if isinstance(info.get("route_home"), dict) else None
        if route and route.get("mode") == "route" and route.get("next_id") is not None:
            home = route.get("home_id")
            reason = route.get("reason") or ""
            place_s += (
                f" route={route.get('next_id')}(hops={int(route.get('hops_left') or 0)}"
                f",home={home}{',' + reason if reason else ''})"
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
    opening = cues.get("opening_strength") if cues else None
    if opening is not None:
        bearing_s += f" open={float(opening):.2f}@{float(cues.get('opening_bearing') or 0.0):+.2f}"
    cur = info.get("curiosity_gated")
    cur_s = f" cur={float(cur):.3f}" if cur is not None else ""
    exp_nov = info.get("experience_novelty")
    if exp_nov is not None:
        cur_s += f" exp={float(exp_nov):.2f}"
    bore = info.get("boredom")
    bore_s = f" bore={float(bore):.2f}" if bore is not None else ""
    if info.get("blocked_escape"):
        bore_s += " ESCAPE"
    elif int(info.get("blocked_streak") or 0) > 0:
        bore_s += f" blocked={int(info['blocked_streak'])}"
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
    speech_s = ""
    if info.get("speech_hear") or info.get("speech_babble") or info.get("speech_playback"):
        parts = []
        if info.get("speech_hear"):
            parts.append(f"hear={info.get('speech_alphabet') or '?'}")
        if info.get("speech_babble"):
            parts.append(f"babble={info.get('speech_alphabet') or info.get('speech_name') or '?'}")
        elif info.get("speech_playback"):
            parts.append(f"mouth={info.get('speech_name') or 'play'}")
        if info.get("audio_rms") is not None:
            parts.append(f"rms={float(info['audio_rms']):.2f}")
        speech_s = f" speech[{','.join(parts)}]"
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
            vthink_s = (
                f" vthink={insight.get('decision')}(sal={float(insight.get('salience', 0.0)):.2f})"
            )
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
        judge_s += f" invest={invest.get('kind')}(rem={int(invest.get('remaining', 0))})"

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
        f"{track_s}{energy_s}{hunger_s}{sniff_s}{surv_s}{fear_s}{wall_s}{place_s}{bearing_s}{cur_s}{bore_s}{tend_s}{viz_s}{speech_s}{antenna_s}{vthink_s}{judge_s}"
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


def _proboscis_feedback(summary: dict[str, Any]) -> dict[str, float | None]:
    """Unity-reported proboscis state (extension actually applied, contact, taste)."""
    return {
        "proboscis_extension": summary.get("proboscis_extension"),
        "proboscis_contact": summary.get("proboscis_contact"),
        "taste_sugar": summary.get("sugar"),
    }


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


def _load_live_vision_head(path: Path) -> RecurrentVisionHead | None:
    """Unity-trained (RGB) head, or ``None`` so live play falls back to the heuristic cues."""
    try:
        head = load_vision_head(path)
    except IncompatibleVisionCheckpoint as exc:
        reason = f"incompatible ({exc})"
    else:
        if head.in_channels == 3:
            return head
        reason = "grayscale synthetic head; live eyes are RGB"
    print(
        f"vision_checkpoint_skipped path={path} reason={reason} "
        "fix=Tools/Fly/Record Vision Dataset then scripts.train_vision_head --unity",
        flush=True,
    )
    return None


def log_console_line(line: str, *, log_path: str | Path | None = None) -> None:
    """Print a console event to stdout + stderr and optionally append to the live log file."""
    print(line, flush=True)
    print(line, file=sys.stderr, flush=True)
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


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
    log_console_line(line, log_path=log_path)
    if verbose and bool(info.get("mind_tick")):
        detail = format_live_mind_detail(step, info)
        log_console_line(detail, log_path=log_path)


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
    # Async mind: thoughts run on a background worker while the motor steps every tick;
    # mind_interval is then unused. A stimulus interrupt discards the in-flight thought
    # and pauses for deep thought until at least one reframed thought commits.
    async_mind: bool = False
    # Minimum motor steps between async thought submissions (ignored during deep thought).
    async_min_steps: int = 4
    # Freeze cap past deep_thought_steps while waiting for a reframed deep thought.
    async_reframe_max_steps: int = 400
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
    father_advocate: FatherAdvocateConfig | None = None
    place_graph: PlaceGraphConfig | None = None
    skill_cortex: SkillCortexConfig | None = None
    speech_cortex: SpeechCortexConfig | None = None
    phrase_grounding: GroundedPhraseConfig | None = None
    sugar_search: SugarSearchConfig | None = None
    route_home: RouteHomeConfig | None = None

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
        father_raw = data.get("father_advocate") or {}
        place_raw = data.get("place_graph") or {}
        skill_raw = data.get("skill_cortex") or {}
        speech_raw = data.get("speech_cortex") or {}
        phrase_raw = data.get("phrase_grounding") or {}
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
            async_mind=bool(motor_raw.get("async_mind", False)),
            async_min_steps=max(1, int(motor_raw.get("async_min_steps", 4))),
            async_reframe_max_steps=max(0, int(motor_raw.get("async_reframe_max_steps", 400))),
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
            father_advocate=FatherAdvocateConfig.from_dict(father_raw),
            place_graph=PlaceGraphConfig.from_dict(place_raw),
            skill_cortex=SkillCortexConfig.from_dict(skill_raw),
            speech_cortex=SpeechCortexConfig.from_dict(speech_raw),
            phrase_grounding=GroundedPhraseConfig.from_dict(phrase_raw),
            sugar_search=SugarSearchConfig.from_dict(data.get("sugar_search") or {}),
            route_home=RouteHomeConfig.from_dict(data.get("route_home") or {}),
        )


class _StaleMindJob(Exception):
    """Raised on the async worker when a new stimulus invalidated the running thought."""


@dataclass
class MindInputs:
    """Senses captured on the motor loop for one thought."""

    observation: np.ndarray
    left_eye: np.ndarray | None
    right_eye: np.ndarray | None
    chem_bearings: dict[str, Any] | None
    position: np.ndarray | None
    chem_vector: np.ndarray | None
    step: int
    sim_time: float


@dataclass
class MindThought:
    """Result of a mind tick's thinking half, committed on the motor loop."""

    info: dict[str, Any]
    goal: np.ndarray | None
    mind_action: np.ndarray
    plan_name: str | None
    sensory: Any
    position: np.ndarray | None


@dataclass
class DeepThought:
    """Result of a deep-thought tick's thinking half, committed on the motor loop."""

    info: dict[str, Any]
    judged_goal: np.ndarray
    l1_goal: np.ndarray
    accepted: bool
    decision: str
    early_unlock: bool
    plan_name: str | None
    position: np.ndarray | None


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
        # Async: the mind worker and motor-side interrupt checks share STM.
        if self.config.async_mind:
            self.memory = SerializedAccess(self.memory)
        self._lock = FairLock()
        self._mind_tls = threading.local()
        self.mind_scheduler: AsyncMindScheduler | None = (
            AsyncMindScheduler() if self.config.async_mind else None
        )
        self._deep_thoughts_done = 0
        self._reframe_wait = 0
        self._last_mind_submit_step = 0
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
            ckpt = self.config.vision_checkpoint
            path = Path(ckpt) if ckpt else None
            if path is not None and not path.is_absolute():
                path = Path(__file__).resolve().parents[2] / path
            if self.vision_head is None and path is not None and path.exists():
                self.vision_head = _load_live_vision_head(path)
                self._vision_trained = self.vision_head is not None
            self.vision_head = self.vision_head or RecurrentVisionHead()
            self.vision_head.eval()
            self.vision_head.reset_state(1)
        self.curiosity = CuriosityDrive(self.config.curiosity or CuriosityConfig())
        self.boredom = BoredomTracker(self.config.boredom or BoredomConfig())
        self.sugar_search = SugarSearchTracker(self.config.sugar_search or SugarSearchConfig())
        self._last_sugar_search: dict[str, Any] | None = None
        self.route_home = RouteHomeTracker(
            config=self.config.route_home or RouteHomeConfig(),
            fly_id=self.config.fly_id,
        )
        self._last_route_home: dict[str, Any] | None = None
        self._last_place_node = None
        from flybrain.debug.father_dialog_channel import FatherDialogChannel
        from flybrain.debug.mind_log_channel import MindLogDebugChannel
        from flybrain.debug.route_path_channel import RoutePathDebugChannel

        env_channel = getattr(env, "route_path_channel", None)
        self.route_path = env_channel if env_channel is not None else RoutePathDebugChannel()
        mind_log_channel = getattr(env, "mind_log_channel", None)
        self.mind_log = mind_log_channel if mind_log_channel is not None else MindLogDebugChannel()
        father_ch = getattr(env, "father_dialog_channel", None)
        self.father_dialog = father_ch if father_ch is not None else FatherDialogChannel()
        self.father_config = self.config.father_advocate or FatherAdvocateConfig()
        father_mock = (
            self.config.llm_mock
            if self.father_config.mock is None
            else bool(self.father_config.mock)
        )
        self.father_client = make_father_client(
            mock=father_mock,
            llm_kwargs=self.config.llm_kwargs,
        )
        self.father_lessons = FatherLessonStore(max_lessons=int(self.father_config.max_lessons))
        self._father_waiting = False
        self._father_wait_steps = 0
        self._father_wait_started: float | None = None
        self._father_request_id = 0
        self._father_question: FatherQuestion | None = None
        self._father_lesson: FatherLesson | None = None
        self._next_father_request_id = 1
        self._orient_base: np.ndarray | None = None
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
        self._status_reframe = StatusReframeTracker()
        si = self.stimulus_interrupt
        self.conditioning = HazardConditioning(
            ConditioningConfig(
                hit_threshold=int(getattr(si, "conditioning_hit_threshold", 3) or 3),
                judgement_confirmations=int(getattr(si, "judgement_confirmations", 2) or 2),
            )
        )
        pg_cfg = self.config.place_graph or PlaceGraphConfig(enabled=False, backend="in_memory")
        # Isolate place-cache disk per fly so Fly 1 / 2 maps do not collide.
        if isinstance(pg_cfg.cache, dict):
            from flybrain.fly_profiles import place_cache_dir_for_fly

            base = str(pg_cfg.cache.get("dir") or "logs/place_cache")
            if not str(base).rstrip("/").endswith(self.config.fly_id):
                pg_cfg.cache = {
                    **pg_cfg.cache,
                    "dir": place_cache_dir_for_fly(self.config.fly_id, base=base),
                }
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
        sc_cfg = self.config.skill_cortex or SkillCortexConfig(enabled=False, backend="in_memory")
        self.skill_cortex_config = sc_cfg
        self.skill_cortex = None
        self.skill_executor = SkillExecutor()
        self._last_babble_step = -10_000
        self._last_babble_skill_id: str | None = None
        self._pending_skill_promote_id: str | None = None
        self._skill_promote_queue: list[str] = []
        self._skill_promote_showing = False
        self._skill_promote_request_id = 0
        self._next_skill_promote_request_id = 1
        self._skill_promote_shown_at: float | None = None
        self._skill_promote_auto_fired = False
        self._skill_playing_candidate = False
        from flybrain.debug.skill_promote_channel import SkillPromoteChannel

        promote_ch = getattr(env, "skill_promote_channel", None)
        self.skill_promote = promote_ch if promote_ch is not None else SkillPromoteChannel()
        if sc_cfg.enabled:
            self.skill_cortex = open_skill_cortex(
                backend=sc_cfg.backend,
                uri=sc_cfg.uri,
                user=sc_cfg.user,
                password=sc_cfg.password,
                database=sc_cfg.database,
            )
            self._refresh_skill_candidates()
        sp_cfg = self.config.speech_cortex or SpeechCortexConfig(enabled=False, backend="in_memory")
        self.speech_cortex_config = sp_cfg
        self.speech_cortex = None
        self.speech_executor = SpeechExecutor()
        self._last_speech_babble_step = -10_000
        self._last_hear_step = -10_000
        self._last_env_audio: dict[str, Any] = {}
        self._last_speech_babble_meta: dict[str, Any] = {}
        self._last_speech_hear_alphabet: str = ""
        self._last_speech_babble_alphabet: str = ""
        self._last_ground_phrase: str = ""
        if sp_cfg.enabled:
            self.speech_cortex = open_speech_cortex(
                backend=sp_cfg.backend,
                uri=sp_cfg.uri,
                user=sp_cfg.user,
                password=sp_cfg.password,
                database=sp_cfg.database,
            )
        # Grounded phrases live on the skill Neo4j cluster (same Bolt as skills).
        pg_phrase = self.config.phrase_grounding or GroundedPhraseConfig(enabled=False)
        self.phrase_grounding_config = pg_phrase
        self.phrase_store = None
        self.phrase_grounding: PhraseGroundingService | None = None
        self.sentence_compose: SentenceComposeService | None = None
        if pg_phrase.enabled:
            mock = pg_phrase.mock
            if mock is None:
                vt = self.config.visual_thinker or VisualThinkerConfig()
                mock = bool(vt.mock) if vt is not None else bool(self.config.llm_mock)
            backend = str(pg_phrase.backend or "in_memory")
            uri = pg_phrase.uri
            user = pg_phrase.user
            password = pg_phrase.password
            database = pg_phrase.database
            if backend == "neo4j" and sc_cfg.enabled:
                uri = sc_cfg.uri
                user = sc_cfg.user
                password = sc_cfg.password
                database = sc_cfg.database
            self.phrase_store = open_grounded_phrase_store(
                backend=backend,
                uri=uri,
                user=user,
                password=password,
                database=database,
            )
            phrase_client = make_phrase_client(
                mock=bool(mock),
                visual_cfg=self.config.visual_thinker or VisualThinkerConfig(),
            )
            self.phrase_grounding = PhraseGroundingService(
                pg_phrase,
                self.phrase_store,
                client=phrase_client,
                visual_cfg=self.config.visual_thinker or VisualThinkerConfig(),
            )
            if pg_phrase.sentence_compose:
                self.sentence_compose = SentenceComposeService(
                    pg_phrase,
                    self.phrase_store,
                    client=phrase_client,
                    visual_cfg=self.config.visual_thinker or VisualThinkerConfig(),
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
        self.act_controller: MotorActController | None = None
        if self.motor is not None and bool(getattr(motor_cfg, "act_state_enabled", True)):
            self.act_controller = MotorActController(self.motor)
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

    def _stamp_act_state(self, info: dict[str, Any]) -> None:
        """Expose current mind-pushed act mode on the live info dict."""
        if self.act_controller is None:
            return
        d = self.act_controller.current
        info["act_mode"] = d.mode
        info["act_plan"] = d.plan_name
        info["act_reason"] = d.reason

    def _push_motor_directive(
        self,
        *,
        mode: str | None = None,
        plan_name: str | None = None,
        goal: np.ndarray | None = None,
        reason: str = "",
        allow_pivot: bool = False,
    ) -> Any:
        """Mind → motor interrupt: push a directive onto the act-state controller."""
        if self.act_controller is None or self.motor is None:
            return None
        directive = make_directive(
            mode=mode,  # type: ignore[arg-type]
            plan_name=plan_name,
            goal=goal,
            reason=reason,
            step=int(self._step),
            allow_pivot=allow_pivot,
        )
        self.act_controller.push(directive)
        set_goal = getattr(self.env, "set_goal", None)
        if callable(set_goal):
            set_goal(self.motor.goal.copy())
        return directive

    def _abort_stimulus_interrupt_for_route(self) -> None:
        """Drop an in-flight freeze without STM persist so routing can keep moving."""
        if not self._interrupt_pending:
            return
        self._interrupt_pending = False
        self._interrupt_remaining = 0
        self._interrupt_reason = ""
        self._interrupt_meta = {}
        self._interrupt_patch_vec = None
        self._interrupt_smell_vec = None
        self._interrupt_sensors_vec = None
        self._interrupt_joint_vec = None
        self._resume_goal = None
        self.conditioning.end_freeze()
        if self.mind_scheduler is not None:
            self.mind_scheduler.invalidate("route_blocks_interrupt")

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
        # Active return path wins: do not freeze / reframe mid-route (hunger after
        # low_energy used to stop the body and spawn a second path).
        if self.route_home.route_in_progress:
            self._abort_stimulus_interrupt_for_route()
            if info is not None:
                info["interrupt_debug"] = {
                    "reason": "routing",
                    "armed": False,
                    "cue": None,
                }
            return
        if self._interrupt_pending:
            return
        # Feeding tastes sugar every step; a deep-thought freeze would retract the proboscis.
        if self.sugar_search.feeding_now(dict(sensory.summary)):
            return
        cfg = self.stimulus_interrupt
        if not cfg.enabled:
            return
        signals = (
            sensory.to_jsonable() if hasattr(sensory, "to_jsonable") else dict(sensory.summary)
        )
        affect_payload = affect.to_jsonable() if hasattr(affect, "to_jsonable") else dict(affect)
        # Avoid per-tick embed on the motor path. Joint SI pads missing sensors; patch
        # comes from eyes inside evaluate; smell from the antenna vector.
        smell = None
        if hasattr(sensory, "smell_vec"):
            smell = np.asarray(sensory.smell_vec, dtype=np.float32)
        decision = evaluate_stimulus_interrupt(
            memory=self.memory,
            parts=None,
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
            sensors_vec=None,
            passive_smell=self._passive_smell,
            status_tracker=self._status_reframe,
        )
        # Debug: expose *why* an interrupt did / didn't arm whenever a cue was present.
        if decision.candidate:
            debug = {
                "reason": decision.reason,
                "cue": decision.cue,
                "backend": str(cfg.history_backend),
                "min_sim": round(float(decision.best_similarity), 4),
                "scores": {k: round(float(v), 4) for k, v in decision.modality_scores.items()},
                "baselines": {k: round(float(v), 4) for k, v in decision.history_baselines.items()},
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
            status_steps = getattr(cfg, "status_deep_thought_steps", None)
            is_status = str(decision.cue or "") in STATUS_REFRAME_CUES
            if is_status and status_steps is not None:
                freeze_steps = int(status_steps)
            else:
                freeze_steps = int(cfg.deep_thought_steps)
            self._interrupt_remaining = max(1, freeze_steps)
            self._interrupt_reason = decision.reason
            # Reframe: a thought planned before this stimulus must not reach the motor.
            if self.mind_scheduler is not None:
                self.mind_scheduler.invalidate(f"stimulus:{decision.reason}")
            self._deep_thoughts_done = 0
            self._reframe_wait = 0
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
                "status_reframe": is_status,
            }
            # Mind → motor: idle for deep thought (single directive bus).
            self._push_motor_directive(
                mode="idle",
                plan_name="idle",
                goal=STOP_GOAL.copy(),
                reason=str(decision.reason or "stimulus_interrupt"),
            )
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
            self._begin_father_advocate(signals if isinstance(signals, dict) else {})

    def _begin_father_advocate(self, signals: dict[str, Any]) -> None:
        """Ask the Father advocate when a novel interrupt arms."""
        cfg = self.father_config
        self._father_lesson = None
        self._father_question = None
        self._father_waiting = False
        self._father_wait_steps = 0
        self._father_wait_started = None
        if not bool(cfg.enabled):
            return
        cue = str((self._interrupt_meta or {}).get("cue") or "")
        place_id = getattr(self._last_place_node, "id", None)
        prior = self._lookup_father_lesson(cue=cue, place_id=place_id)
        if prior is not None:
            # Already taught on this place+cue (or cue) — reuse without blocking.
            self._father_lesson = prior
            self._install_father_resume_goal(prior)
            if self.config.console_log:
                print(
                    f"father_reuse step={self._step} cue={cue} place={place_id} "
                    f"action={prior.action} lesson={prior.lesson[:80]}",
                    flush=True,
                )
            return
        request_id = int(self._next_father_request_id)
        self._next_father_request_id = request_id + 1
        question = propose_father_question(
            self.father_client,
            request_id=request_id,
            step=int(self._step),
            cue=cue,
            reason=str(self._interrupt_reason or ""),
            signals=signals,
            config=cfg,
        )
        self._father_question = question
        self._father_request_id = request_id
        mode = str(cfg.mode or "human_or_llm").lower()
        has_unity = self.father_dialog.side_channel is not None
        if mode == "llm" or not has_unity:
            reply = answer_father_question(
                self.father_client,
                question,
                config=cfg,
                source="llm",
            )
            self._apply_father_reply(reply)
            return
        allow_llm = mode in ("human_or_llm", "llm")
        self.father_dialog.send_ask(
            request_id=request_id,
            step=int(self._step),
            cue=cue,
            question=question.question,
            allow_llm=allow_llm,
        )
        # Hard-stop motor until human/LLM reply (or timeout_seconds → auto LLM).
        self._father_waiting = True
        self._father_wait_steps = 0
        self._father_wait_started = time.monotonic()
        self._father_polled_step = -1
        if self.motor is not None:
            pushed = self._push_motor_directive(
                mode="idle",
                plan_name="idle",
                goal=STOP_GOAL.copy(),
                reason="father_waiting",
            )
            if pushed is None:
                self.motor.set_goal(STOP_GOAL.copy())
                set_goal = getattr(self.env, "set_goal", None)
                if callable(set_goal):
                    set_goal(STOP_GOAL.copy())
        if self.config.console_log:
            away = float(getattr(cfg, "auto_llm_away_seconds", 0.0))
            print(
                f"father_ask step={self._step} id={request_id} cue={cue} "
                f"away_llm_s={away:.1f} q={question.question}",
                flush=True,
            )

    def _lookup_father_lesson(
        self,
        *,
        cue: str,
        place_id: int | str | None,
    ) -> FatherLesson | None:
        """Reuse RAM lesson, then place-graph ``HAS_LESSON`` for place+cue."""
        ram = self.father_lessons.latest_for_place_cue(place_id, cue)
        if ram is not None:
            return ram
        if place_id is None or self.place_graph is None:
            return None
        latest_fn = getattr(self.place_graph, "latest_lesson", None)
        if not callable(latest_fn):
            return None
        try:
            placed = latest_fn(place_id, cue=cue)
        except Exception:
            return None
        if placed is None:
            return None
        return FatherLesson(
            cue=str(getattr(placed, "cue", cue) or cue),
            question=str(getattr(placed, "question", "") or ""),
            answer=str(getattr(placed, "answer", "") or ""),
            lesson=str(getattr(placed, "lesson", "") or getattr(placed, "answer", "") or ""),
            risk=str(getattr(placed, "risk", "medium") or "medium"),
            source=str(getattr(placed, "source", "place") or "place"),
            step=int(getattr(placed, "step", 0) or 0),
            place_id=getattr(placed, "place_id", place_id),
            action=str(getattr(placed, "action", "idle") or "idle"),
            turn_bias=float(getattr(placed, "turn_bias", 0.0) or 0.0),
            confidence=float(getattr(placed, "confidence", 0.5) or 0.5),
        )

    def _persist_father_lesson(self, lesson: FatherLesson) -> None:
        """Attach ``(:Place)-[:HAS_LESSON]->(:Lesson)`` when a place is known."""
        if self.place_graph is None or lesson.place_id is None:
            return
        upsert = getattr(self.place_graph, "upsert_lesson", None)
        if not callable(upsert):
            return
        try:
            upsert(
                lesson.place_id,
                cue=lesson.cue,
                question=lesson.question,
                answer=lesson.answer,
                lesson=lesson.lesson,
                risk=lesson.risk,
                source=lesson.source,
                step=int(lesson.step),
                action=lesson.action,
                turn_bias=float(lesson.turn_bias),
                confidence=float(lesson.confidence),
            )
        except Exception as exc:
            if self.config.console_log:
                print(f"father_place_lesson_error step={self._step} err={exc}", flush=True)

    def _install_father_resume_goal(self, lesson: FatherLesson) -> None:
        """Blend structured Father action into the post-freeze resume goal."""
        if float(lesson.confidence) < 0.25:
            return
        self._resume_goal = blend_goal_with_father(self._resume_goal, lesson)

    def _push_father_motor_directive(self, lesson: FatherLesson) -> None:
        """Push structured Father action onto the motor directive bus."""
        if self.act_controller is None or self.motor is None:
            return
        if float(lesson.confidence) < 0.25:
            return
        if self._maybe_start_skill_from_lesson(lesson):
            return
        fields = directive_fields_from_father_lesson(lesson)
        self._push_motor_directive(
            mode=fields["mode"],
            plan_name=fields["plan_name"],
            goal=fields["goal"],
            reason=str(fields["reason"]),
            allow_pivot=bool(fields["allow_pivot"]),
        )

    def _refresh_skill_candidates(self) -> None:
        """Load promoted skills into L1 dynamic candidate pool."""
        thinking = getattr(self, "thinking", None)
        if thinking is None:
            return
        if self.skill_cortex is None:
            thinking.extra_action_candidates = []
            return
        try:
            promoted = self.skill_cortex.list_skills(
                fly_id=self.config.fly_id, status="promoted", limit=32
            )
        except Exception:
            thinking.extra_action_candidates = []
            return
        thinking.extra_action_candidates = [
            (str(p.name), np.asarray(p.first_goal, dtype=np.float32)) for p in promoted if p.steps
        ]

    def _lookup_promoted_skill(self, name: str) -> SkillProgram | None:
        if self.skill_cortex is None or not str(name or "").strip():
            return None
        key = str(name).strip()
        if key.startswith("skill_"):
            key = key[len("skill_") :]
        try:
            return self.skill_cortex.get_by_name(key, fly_id=self.config.fly_id, status="promoted")
        except Exception:
            return None

    def _start_skill_playback(self, program: SkillProgram, *, candidate: bool = False) -> None:
        self.skill_executor.start(program)
        self._skill_playing_candidate = bool(candidate)
        # Install first step without advancing duration; caller ticks next.
        if self.act_controller is not None and self.motor is not None and program.steps:
            step0 = program.steps[0]
            self._push_motor_directive(
                mode="act_anything",
                plan_name=f"skill_{program.name}",
                goal=np.asarray(step0.goal, dtype=np.float32).copy(),
                reason="skill_start",
                allow_pivot=True,
            )

    def _maybe_start_skill_from_lesson(self, lesson: FatherLesson) -> bool:
        """If lesson names a promoted skill, start playback instead of single-goal map."""
        name = str(lesson.skill_name or "").strip()
        if not name:
            # Place lessons may store skill name in action when Father taught a move.
            action = str(lesson.action or "").strip()
            if action and action not in ("approach", "avoid", "taste", "idle"):
                name = action
        if not name:
            return False
        program = self._lookup_promoted_skill(name)
        if program is None:
            return False
        self._start_skill_playback(program, candidate=False)
        return True

    def _enqueue_skill_promote(self, skill_id: str) -> None:
        """Queue a scored babble candidate for non-blocking Display 4 HITL."""
        sid = str(skill_id or "").strip()
        if not sid or sid in self._skill_promote_queue:
            return
        self._skill_promote_queue.append(sid)
        self._pending_skill_promote_id = sid
        self._show_skill_promote_ui()

    def _show_skill_promote_ui(self, *, suggested_name: str = "") -> None:
        if self.skill_cortex is None or not self._skill_promote_queue:
            if self._skill_promote_showing:
                self.skill_promote.send_hide(request_id=int(self._skill_promote_request_id))
                self._skill_promote_showing = False
            return
        skill_id = self._skill_promote_queue[0]
        try:
            program = self.skill_cortex.get_skill(skill_id)
        except Exception:
            program = None
        if program is None:
            self._skill_promote_queue.pop(0)
            self._show_skill_promote_ui(suggested_name=suggested_name)
            return
        from flybrain.debug.skill_promote_channel import suggest_skill_name

        request_id = int(self._next_skill_promote_request_id)
        self._next_skill_promote_request_id = request_id + 1
        self._skill_promote_request_id = request_id
        suggestion = suggested_name or suggest_skill_name(
            tags=list(program.tags), current_name=program.name
        )
        self.skill_promote.send_show(
            request_id=request_id,
            step=int(self._step),
            queue_count=len(self._skill_promote_queue),
            skill_id=program.skill_id,
            name=program.name,
            tags=",".join(program.tags),
            score=f"{float(program.score):.2f}",
            steps_count=str(len(program.steps)),
            suggested_name=suggestion,
            allow_llm=True,
        )
        self._skill_promote_showing = True
        self._skill_promote_shown_at = time.monotonic()
        self._skill_promote_auto_fired = False
        if self.config.console_log:
            print(
                f"skill_promote_show step={self._step} id={program.skill_id} "
                f"queue={len(self._skill_promote_queue)} name={program.name}",
                flush=True,
            )

    def _poll_skill_promote_ui(self) -> None:
        """Non-blocking: accept / ask_llm / refuse on Display 4; motor keeps running."""
        if not self._skill_promote_showing and not self._skill_promote_queue:
            return
        reply = self.skill_promote.pop_reply(self._skill_promote_request_id)
        if reply is None:
            reply = self.skill_promote.pop_reply(None)
        if reply is None:
            self._maybe_auto_ask_llm_when_away()
            return
        source = str(reply.get("source") or "")
        name = str(reply.get("name") or "").strip()
        if not self._skill_promote_queue:
            self._skill_promote_showing = False
            return
        skill_id = self._skill_promote_queue[0]
        if source == "ask_llm":
            self._ask_llm_adjudicate_skill(skill_id, via="display4_llm")
            return
        if source == "test":
            # Replay candidate on the body without dequeue / promote.
            if self.skill_cortex is not None:
                try:
                    program = self.skill_cortex.get_skill(skill_id)
                except Exception:
                    program = None
                if program is not None:
                    self._start_skill_playback(program, candidate=True)
                    if self.config.console_log:
                        print(
                            f"skill_promote_test step={self._step} id={program.skill_id} "
                            f"name={program.name}",
                            flush=True,
                        )
            return
        if source in ("accept", "refuse"):
            self._resolve_skill_promote_decision(skill_id, source=source, name=name)
            return

    def _maybe_auto_ask_llm_when_away(self) -> None:
        """Python fallback Ask-LLM when Unity never reports Display 4 focus.

        When Unity reports away (``user_on_display4 is False``), Unity owns the
        immediate ask_llm send so we do not double-adjudicate. When focus is
        unknown (fake env / no VIEW), auto after ``auto_llm_away_seconds``
        (``0`` = immediate, ``<0`` = disabled). Does not interrupt motor flow —
        only Unity HITL freeze on Display 4 stops the fly.
        """
        if self._skill_promote_auto_fired or not self._skill_promote_showing:
            return
        if not self._skill_promote_queue:
            return
        grace = float(getattr(self.skill_cortex_config, "auto_llm_away_seconds", 0.0))
        if grace < 0.0:
            return
        viewing = getattr(self.skill_promote, "user_on_display4", None)
        if viewing is True:
            return
        if viewing is False:
            # Unity TickAwayAutoJudge will SendReply(ask_llm); wait for that.
            return
        shown_at = float(self._skill_promote_shown_at or time.monotonic())
        if grace > 0.0 and (time.monotonic() - shown_at) < grace:
            return
        skill_id = self._skill_promote_queue[0]
        self._skill_promote_auto_fired = True
        if self.config.console_log:
            print(
                f"skill_promote_auto_llm step={self._step} id={skill_id} reason=no_view_signal",
                flush=True,
            )
        self._ask_llm_adjudicate_skill(skill_id, via="display4_llm_auto")

    def _ask_llm_adjudicate_skill(self, skill_id: str, *, via: str = "display4_llm") -> None:
        """Ask LLM (or mock heuristic) to accept/refuse the Display 4 head candidate."""
        from flybrain.debug.skill_promote_channel import adjudicate_skill_promote

        self._skill_promote_auto_fired = True
        program = None
        if self.skill_cortex is not None:
            try:
                program = self.skill_cortex.get_skill(skill_id)
            except Exception:
                program = None
        if program is None:
            self._resolve_skill_promote_decision(skill_id, source="refuse", name="")
            return
        min_score = float(getattr(self.skill_cortex_config, "promote_min_score", 0.4) or 0.4)
        # Prefer live father/LLM client; fall back to heuristic when mock.
        mock = bool(self.config.llm_mock)
        verdict = adjudicate_skill_promote(
            program,
            client=None if mock else getattr(self, "father_client", None),
            min_score=min_score,
            mock=mock,
        )
        if self.config.console_log:
            print(
                f"skill_promote_llm step={self._step} id={skill_id} "
                f"decision={verdict.decision} name={verdict.name or '-'} "
                f"src={verdict.source} reason={verdict.reason} via={via}",
                flush=True,
            )
        self._resolve_skill_promote_decision(
            skill_id,
            source=verdict.decision,
            name=verdict.name,
            via=via,
        )

    def _resolve_skill_promote_decision(
        self,
        skill_id: str,
        *,
        source: str,
        name: str = "",
        via: str = "display4",
    ) -> None:
        """Accept or refuse the queue head, then show next / hide."""
        if self._skill_promote_queue and self._skill_promote_queue[0] == skill_id:
            self._skill_promote_queue.pop(0)
        elif skill_id in self._skill_promote_queue:
            self._skill_promote_queue = [s for s in self._skill_promote_queue if s != skill_id]
        if source == "accept" and self.skill_cortex is not None:
            from flybrain.thinking.father_advocate import normalize_skill_name

            clean = normalize_skill_name(name) or name or None
            try:
                promoted = self.skill_cortex.promote_skill(
                    skill_id, name=clean, tags=["display4", "llm"] if "llm" in via else ["display4"]
                )
            except Exception as exc:
                promoted = None
                if self.config.console_log:
                    print(f"skill_promote_error step={self._step} err={exc}", flush=True)
            if promoted is not None:
                if self._pending_skill_promote_id == skill_id:
                    self._pending_skill_promote_id = None
                self._refresh_skill_candidates()
                if self.config.console_log:
                    print(
                        f"skill_promoted step={self._step} id={promoted.skill_id} "
                        f"name={promoted.name} via={via}",
                        flush=True,
                    )
        elif source == "refuse" and self.skill_cortex is not None:
            try:
                self.skill_cortex.retire_skill(skill_id)
            except Exception:
                pass
            if self._pending_skill_promote_id == skill_id:
                self._pending_skill_promote_id = None
            if self.config.console_log:
                print(
                    f"skill_promote_refuse step={self._step} id={skill_id} via={via}",
                    flush=True,
                )
        if self._skill_promote_queue:
            self._show_skill_promote_ui()
        else:
            self.skill_promote.send_hide(request_id=int(self._skill_promote_request_id))
            self._skill_promote_showing = False
            self._skill_promote_shown_at = None

    def _handle_father_skill_promote(self, lesson: FatherLesson) -> None:
        """Promote last babble candidate and/or bind skill_name into cortex."""
        if self.skill_cortex is None:
            return
        skill_id = self._pending_skill_promote_id or self._last_babble_skill_id
        if lesson.promote_last_candidate or lesson.skill_name:
            if skill_id:
                try:
                    promoted = self.skill_cortex.promote_skill(
                        skill_id,
                        name=lesson.skill_name or None,
                        tags=["father"],
                    )
                except Exception as exc:
                    if self.config.console_log:
                        print(
                            f"skill_promote_error step={self._step} err={exc}",
                            flush=True,
                        )
                    promoted = None
                if promoted is not None:
                    self._pending_skill_promote_id = None
                    if skill_id in self._skill_promote_queue:
                        self._skill_promote_queue = [
                            s for s in self._skill_promote_queue if s != skill_id
                        ]
                        if self._skill_promote_queue:
                            self._show_skill_promote_ui()
                        elif self._skill_promote_showing:
                            self.skill_promote.send_hide(
                                request_id=int(self._skill_promote_request_id)
                            )
                            self._skill_promote_showing = False
                    self._refresh_skill_candidates()
                    if self.config.console_log:
                        print(
                            f"skill_promoted step={self._step} id={promoted.skill_id} "
                            f"name={promoted.name}",
                            flush=True,
                        )
                    return
            # Father invent: no candidate — nothing to upsert without a program body.
            if lesson.skill_name and self.config.console_log:
                print(
                    f"skill_promote_miss step={self._step} name={lesson.skill_name}",
                    flush=True,
                )

    def _finish_skill_episode_if_needed(self) -> None:
        if self.skill_cortex is None:
            return
        if not self.skill_executor.finished or self.skill_executor.program is None:
            return
        stats = self.skill_executor.episode_stats()
        skill_id = str(stats.get("skill_id") or "")
        if not skill_id:
            self.skill_executor.clear()
            return
        score = score_babble_episode(
            motion_variance=float(stats.get("motion_variance") or 0.0),
            damage_spike=bool(stats.get("damage_spike")),
            novelty_during=float(stats.get("novelty_during") or 0.0),
            completed=bool(stats.get("completed")),
        )
        try:
            self.skill_cortex.set_score(skill_id, score)
        except Exception:
            pass
        if self._skill_playing_candidate and score >= float(
            self.skill_cortex_config.promote_min_score
        ):
            self._pending_skill_promote_id = skill_id
            self._enqueue_skill_promote(skill_id)
        self._skill_playing_candidate = False
        self.skill_executor.clear()

    def _maybe_babble_skill(self, signals: dict[str, Any], affect: dict[str, Any]) -> bool:
        """Start a babble candidate when gated; returns True if playback started."""
        if self.skill_cortex is None or not self.skill_cortex_config.enabled:
            return False
        if self.skill_executor.active:
            return False
        if self._interrupt_pending or self._father_waiting:
            return False
        gate_vals = signals_for_gate(signals, affect)
        gate = babble_gate(
            config=self.skill_cortex_config.babble,
            step=int(self._step),
            last_babble_step=int(self._last_babble_step),
            executor_active=self.skill_executor.active,
            novelty=gate_vals["novelty"],
            survival_risk=gate_vals["survival_risk"],
            hunger=gate_vals["hunger"],
            explore_tendency=gate_vals["explore_tendency"],
            library_score_weak=True,
        )
        if not gate.ok:
            return False
        rng = np.random.default_rng(int(self._step) ^ hash(self.config.fly_id) & 0xFFFFFFFF)
        name, steps, tags = sample_babble_program(rng, config=self.skill_cortex_config.babble)
        place_id = getattr(self._last_place_node, "id", None)
        program = build_candidate_skill(
            fly_id=self.config.fly_id,
            steps=steps,
            name=name,
            tags=tags,
            cue=str(signals.get("interrupt_cue") or signals.get("cue") or "babble"),
            place_id=str(place_id) if place_id is not None else None,
            step=int(self._step),
        )
        try:
            stored = self.skill_cortex.upsert_skill(program)
        except Exception as exc:
            if self.config.console_log:
                print(f"skill_babble_error step={self._step} err={exc}", flush=True)
            return False
        self._last_babble_step = int(self._step)
        self._last_babble_skill_id = stored.skill_id
        self._start_skill_playback(stored, candidate=True)
        if self.config.console_log:
            print(
                f"skill_babble step={self._step} id={stored.skill_id} "
                f"name={stored.name} steps={len(stored.steps)}",
                flush=True,
            )
        return True

    def _speech_eye_image(self) -> SpeechImage:
        """Bind current eye embedding (or empty) to an utterance."""
        place_id = getattr(self._last_place_node, "id", None)
        left = getattr(self.env, "left_eye", None)
        right = getattr(self.env, "right_eye", None)
        patch: list[float] = []
        if left is not None and right is not None:
            try:
                emb = stereo_patch_embedding(left, right)
                patch = [float(x) for x in np.asarray(emb, dtype=np.float32).reshape(-1)]
            except Exception:
                patch = []
        return SpeechImage(
            image_id=str(uuid4()),
            path_or_b64_thumb="",
            place_id=str(place_id) if place_id is not None else None,
            step=int(self._step),
            patch_vec=patch,
        )

    def _cache_env_audio(self, step_info: dict[str, Any] | None) -> None:
        if not isinstance(step_info, dict):
            return
        bands = step_info.get("audio_bands")
        if bands is None:
            return
        self._last_env_audio = {
            "audio_bands": list(bands),
            "audio_rms": float(step_info.get("audio_rms", 0.0) or 0.0),
            "audio_voiced": float(step_info.get("audio_voiced", 0.0) or 0.0),
        }

    def _push_mouth(self, control: np.ndarray) -> None:
        setter = getattr(self.env, "set_mouth_articulatory", None)
        if callable(setter):
            setter(control)
        ch = getattr(self.env, "speech_channel", None)
        send = getattr(ch, "send_mouth", None) if ch is not None else None
        if callable(send) and setter is None:
            send(control)

    def _push_speech_status(
        self,
        *,
        hear: str | None = None,
        babble: str | None = None,
        phrase: str | None = None,
    ) -> None:
        """Mirror hear / babble / ground_phrase to the Unity overhead label."""
        if hear is not None:
            self._last_speech_hear_alphabet = str(hear)
        if babble is not None:
            self._last_speech_babble_alphabet = str(babble)
        if phrase is not None:
            self._last_ground_phrase = str(phrase)
        ch = getattr(self.env, "speech_channel", None)
        send = getattr(ch, "send_status", None) if ch is not None else None
        if callable(send):
            send(
                hear=self._last_speech_hear_alphabet,
                babble=self._last_speech_babble_alphabet,
                phrase=self._last_ground_phrase,
            )

    def _poll_phrase_grounding(self, info: dict[str, Any] | None = None) -> None:
        """Commit async LLM phrase/sentence labels into skill Neo4j; reinforce words."""
        svc = self.phrase_grounding
        if svc is not None:
            stored = svc.poll()
            if stored is not None:
                if isinstance(info, dict):
                    info["phrase_grounded"] = stored.phrase
                    info["phrase_confidence"] = float(stored.confidence)
                    info["phrase_id"] = stored.phrase_id
                self._reinforce_phrase_tokens(stored.phrase)
                self._push_speech_status(phrase=stored.phrase)
                if self.config.console_log:
                    log_console_line(
                        f"phrase_grounded step={self._step} phrase={stored.phrase!r} "
                        f"conf={float(stored.confidence):.2f} id={stored.phrase_id}",
                        log_path=self.config.log_path,
                    )
        compose = self.sentence_compose
        if compose is None:
            return
        sent = compose.poll()
        if sent is None:
            return
        if isinstance(info, dict):
            info["sentence_composed"] = sent.sentence
            info["sentence_confidence"] = float(sent.confidence)
            info["sentence_id"] = sent.sentence_id
        self._reinforce_phrase_tokens(sent.sentence)
        # Prefer showing the composed sentence on the ground_phrase line.
        self._push_speech_status(phrase=sent.sentence)
        if self.config.console_log:
            log_console_line(
                f"sentence_composed step={self._step} sentence={sent.sentence!r} "
                f"conf={float(sent.confidence):.2f} id={sent.sentence_id}",
                log_path=self.config.log_path,
            )

    def _reinforce_phrase_tokens(self, phrase: str) -> None:
        cfg = self.phrase_grounding_config
        if not cfg.reinforce_speech_words or self.speech_cortex is None:
            return
        tokens = [t for t in str(phrase or "").lower().split() if t]
        if not tokens:
            return
        delta = float(cfg.word_weight_delta)
        # Full phrase + each token so multi-word and parts both gain weight.
        for token in [str(phrase).lower().strip(), *tokens]:
            if not token:
                continue
            try:
                self.speech_cortex.reinforce_word(
                    fly_id=self.config.fly_id,
                    word=token,
                    delta=delta,
                )
            except Exception:
                continue

    def _ground_or_recall_phrase(
        self,
        *,
        alphabet: str,
        spectrogram: list[list[float]],
        rms: float,
        voiced: float,
        utterance_id: str | None,
        place_id: str | None,
        cue: str,
        source: str,
        info: dict[str, Any] | None,
    ) -> None:
        """Recall matching image+sound phrase/sentence, else queue LLM grounding."""
        svc = self.phrase_grounding
        store = self.phrase_store
        cfg = self.phrase_grounding_config
        if svc is None or store is None or not cfg.enabled:
            return
        eye = self._speech_eye_image()
        patch = list(eye.patch_vec or [])
        hit = match_grounded_phrase(
            store,
            fly_id=self.config.fly_id,
            patch_vec=patch,
            spectrogram=spectrogram,
            config=cfg,
        )
        if hit is not None:
            bumped = store.bump_hit(hit.phrase.phrase_id, delta_score=0.05)
            row = bumped or hit.phrase
            if isinstance(info, dict):
                info["phrase_recall"] = row.phrase
                info["phrase_recall_score"] = float(hit.combined)
                info["phrase_id"] = row.phrase_id
            self._reinforce_phrase_tokens(row.phrase)
            self._push_speech_status(phrase=row.phrase)
            if self.config.console_log:
                log_console_line(
                    f"phrase_recall step={self._step} phrase={row.phrase!r} "
                    f"sim={float(hit.combined):.2f} hits={int(row.hits)}",
                    log_path=self.config.log_path,
                )
            self._maybe_recall_or_compose_sentence(
                patch_vec=patch,
                spectrogram=spectrogram,
                place_id=place_id,
                info=info,
            )
            return
        draft = GroundedPhrase(
            phrase_id=str(uuid4()),
            phrase="",
            alphabet=str(alphabet or ""),
            fly_id=self.config.fly_id,
            source=str(source or "hear"),
            patch_vec=patch,
            spectrogram=[list(r) for r in spectrogram],
            rms=float(rms),
            voiced=float(voiced),
            utterance_id=utterance_id,
            place_id=place_id,
            step=int(self._step),
        )
        bands = list(spectrogram[0]) if spectrogram else []
        left = getattr(self.env, "left_eye", None)
        right = getattr(self.env, "right_eye", None)
        svc.maybe_submit(
            step=int(self._step),
            draft=draft,
            left_eye=left,
            right_eye=right,
            bands=bands,
            cue=cue,
        )
        # Still try sentence compose from near-matching phrases for this scene.
        self._maybe_recall_or_compose_sentence(
            patch_vec=patch,
            spectrogram=spectrogram,
            place_id=place_id,
            info=info,
        )

    def _maybe_recall_or_compose_sentence(
        self,
        *,
        patch_vec: list[float],
        spectrogram: list[list[float]],
        place_id: str | None,
        info: dict[str, Any] | None,
    ) -> None:
        """Recall a stored sentence for this scene, else compose from top phrases + drives."""
        store = self.phrase_store
        cfg = self.phrase_grounding_config
        compose = self.sentence_compose
        if store is None or not cfg.enabled or not cfg.sentence_compose:
            return
        sent_hit = match_grounded_sentence(
            store,
            fly_id=self.config.fly_id,
            patch_vec=patch_vec,
            spectrogram=spectrogram,
            config=cfg,
        )
        if sent_hit is not None:
            bumped = store.bump_sentence_hit(sent_hit.sentence.sentence_id, delta_score=0.05)
            row = bumped or sent_hit.sentence
            if isinstance(info, dict):
                info["sentence_recall"] = row.sentence
                info["sentence_recall_score"] = float(sent_hit.combined)
                info["sentence_id"] = row.sentence_id
            self._reinforce_phrase_tokens(row.sentence)
            self._push_speech_status(phrase=row.sentence)
            if self.config.console_log:
                log_console_line(
                    f"sentence_recall step={self._step} sentence={row.sentence!r} "
                    f"sim={float(sent_hit.combined):.2f} hits={int(row.hits)}",
                    log_path=self.config.log_path,
                )
            return
        if compose is None:
            return
        top = match_top_grounded_phrases(
            store,
            fly_id=self.config.fly_id,
            patch_vec=patch_vec,
            spectrogram=spectrogram,
            config=cfg,
        )
        if len(top) < int(cfg.sentence_min_phrases):
            return
        senses, aff = self._speech_gate_signals(info)
        drive = drive_snapshot(senses, aff)
        draft = GroundedSentence(
            sentence_id=str(uuid4()),
            sentence="",
            phrases=[m.phrase.phrase for m in top],
            phrase_ids=[m.phrase.phrase_id for m in top],
            fly_id=self.config.fly_id,
            drive=drive,
            patch_vec=list(patch_vec),
            spectrogram=[list(r) for r in spectrogram],
            place_id=place_id,
            step=int(self._step),
        )
        left = getattr(self.env, "left_eye", None)
        right = getattr(self.env, "right_eye", None)
        compose.maybe_submit(
            step=int(self._step),
            draft=draft,
            left_eye=left,
            right_eye=right,
        )

    def _speech_gate_signals(
        self, info: dict[str, Any] | None, affect: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build babble-gate inputs from mind snapshot + top-level live fields."""
        snap = info if isinstance(info, dict) else {}
        senses: dict[str, Any] = {}
        raw_signals = snap.get("signals")
        if isinstance(raw_signals, dict):
            senses.update(raw_signals)
        for key in ("visual_novelty", "hunger", "novelty", "experience_novelty"):
            if key in snap and key not in senses:
                senses[key] = snap[key]
        if "tendency_curiosity" not in senses:
            tend = snap.get("tendency")
            if isinstance(tend, dict):
                senses["tendency_curiosity"] = float(
                    tend.get("curiosity", tend.get("c", 0.0)) or 0.0
                )
        aff = affect if isinstance(affect, dict) else {}
        if not aff and isinstance(snap.get("affect"), dict):
            aff = snap["affect"]
        return senses, aff

    def _speech_parallel_tick(self, info: dict[str, Any] | None = None) -> None:
        """Independent speech process: mouth every tick; hear/babble every speech_interval.

        Does not own locomotion — runs alongside mind/motor.
        """
        if self.speech_cortex is None or not self.speech_cortex_config.enabled:
            return
        snap = info if isinstance(info, dict) else {}
        # Pull latest ear features from Unity channel when present.
        ch = getattr(self.env, "speech_channel", None)
        if ch is not None:
            try:
                self._cache_env_audio(ch.audio_info())
            except Exception:
                pass
        if self._last_env_audio:
            snap.setdefault("audio_rms", self._last_env_audio.get("audio_rms"))
            snap.setdefault("audio_voiced", self._last_env_audio.get("audio_voiced"))

        # Mouth playback advances every motor tick (smooth articulatory sequence).
        self._tick_speech_executor(snap)

        interval = max(1, int(getattr(self.speech_cortex_config, "speech_interval", 4) or 4))
        if int(self._step) % interval != 0:
            if isinstance(info, dict):
                if snap.get("speech_playback"):
                    info["speech_playback"] = True
                    info["speech_name"] = snap.get("speech_name")
                    info["speech_utterance_id"] = snap.get("speech_utterance_id")
                if "audio_rms" in snap:
                    info["audio_rms"] = snap["audio_rms"]
            return

        snap["speech_tick"] = True
        heard = self._maybe_hear_utterance(snap)
        babbled = False
        if not self.speech_executor.active:
            senses, aff = self._speech_gate_signals(snap)
            babbled = self._maybe_babble_speech(senses, aff)
            if babbled:
                self._tick_speech_executor(snap)
        if isinstance(info, dict):
            info["speech_tick"] = True
            if heard:
                info["speech_hear"] = True
                info["speech_alphabet"] = snap.get("speech_alphabet")
                info["speech_utterance_id"] = snap.get("speech_utterance_id")
            if babbled:
                info.update(self._last_speech_babble_meta)
            if snap.get("speech_playback"):
                info["speech_playback"] = True
                info["speech_name"] = snap.get("speech_name") or info.get("speech_name")
            if "audio_rms" in snap:
                info["audio_rms"] = snap["audio_rms"]

    def _maybe_hear_utterance(self, info: dict[str, Any] | None = None) -> bool:
        """Upsert a hear-candidate when env audio energy spikes."""
        if self.speech_cortex is None or not self.speech_cortex_config.enabled:
            return False
        audio = self._last_env_audio
        if not audio:
            return False
        rms = float(audio.get("audio_rms", 0.0) or 0.0)
        if rms < float(self.speech_cortex_config.hear_rms_min):
            return False
        cooldown = int(self.speech_cortex_config.hear_cooldown_steps)
        if int(self._step) - int(self._last_hear_step) < cooldown:
            return False
        bands = clip_audio_bands(audio.get("audio_bands"))
        place_id = getattr(self._last_place_node, "id", None)
        cue = "env_audio"
        if isinstance(info, dict):
            cue = str(info.get("interrupt_cue") or info.get("cue") or cue)
        program = build_hear_utterance(
            fly_id=self.config.fly_id,
            spectrogram=[bands],
            rms=rms,
            voiced=float(audio.get("audio_voiced", 0.0) or 0.0),
            image=self._speech_eye_image(),
            context=SpeechContext(
                cue=cue,
                place_id=str(place_id) if place_id is not None else None,
                step=int(self._step),
            ),
        )
        try:
            stored = self.speech_cortex.upsert_utterance(program)
        except Exception as exc:
            if self.config.console_log:
                log_console_line(
                    f"speech_hear_error step={self._step} err={exc}",
                    log_path=self.config.log_path,
                )
            return False
        self._last_hear_step = int(self._step)
        if isinstance(info, dict):
            info["speech_hear"] = True
            info["speech_alphabet"] = stored.alphabet
            info["speech_utterance_id"] = stored.utterance_id
        self._push_speech_status(hear=stored.alphabet)
        self._ground_or_recall_phrase(
            alphabet=stored.alphabet,
            spectrogram=stored.spectrogram,
            rms=float(rms),
            voiced=float(audio.get("audio_voiced", 0.0) or 0.0),
            utterance_id=stored.utterance_id,
            place_id=str(place_id) if place_id is not None else None,
            cue=cue,
            source="hear",
            info=info if isinstance(info, dict) else None,
        )
        if self.config.console_log:
            log_console_line(
                f"speech_hear step={self._step} id={stored.utterance_id} "
                f"alphabet={stored.alphabet!r} rms={rms:.3f}",
                log_path=self.config.log_path,
            )
        return True

    def _maybe_babble_speech(self, signals: dict[str, Any], affect: dict[str, Any]) -> bool:
        """Start articulatory babble on the parallel mouth bus."""
        if self.speech_cortex is None or not self.speech_cortex_config.enabled:
            return False
        if self.speech_executor.active:
            return False
        if self._interrupt_pending or self._father_waiting:
            return False
        gate_vals = signals_for_gate(signals, affect)
        gate = speech_babble.babble_gate(
            config=self.speech_cortex_config.babble,
            step=int(self._step),
            last_babble_step=int(self._last_speech_babble_step),
            executor_active=self.speech_executor.active,
            novelty=gate_vals["novelty"],
            survival_risk=gate_vals["survival_risk"],
            hunger=gate_vals["hunger"],
            explore_tendency=gate_vals["explore_tendency"],
        )
        if not gate.ok:
            return False
        rng = np.random.default_rng(
            (int(self._step) * 9973) ^ (hash(self.config.fly_id) & 0xFFFFFFFF)
        )
        word_weights: list[tuple[str, float]] = []
        try:
            for row in self.speech_cortex.list_words(fly_id=self.config.fly_id, limit=64):
                word_weights.append((str(row.word), float(row.weight)))
        except Exception:
            word_weights = []
        name, steps, tags = sample_babble_utterance(
            rng,
            config=self.speech_cortex_config.babble,
            word_weights=word_weights,
        )
        controls = expand_controls(steps)
        place_id = getattr(self._last_place_node, "id", None)
        program = build_babble_utterance(
            fly_id=self.config.fly_id,
            articulatory=controls,
            image=self._speech_eye_image(),
            context=SpeechContext(
                cue=str(signals.get("interrupt_cue") or signals.get("cue") or "speech_babble"),
                place_id=str(place_id) if place_id is not None else None,
                step=int(self._step),
            ),
        )
        try:
            stored = self.speech_cortex.upsert_utterance(program)
        except Exception as exc:
            if self.config.console_log:
                log_console_line(
                    f"speech_babble_error step={self._step} err={exc}",
                    log_path=self.config.log_path,
                )
            return False
        word_hit = None
        try:
            stored, word_hit = reinforce_wordlike_babble(
                self.speech_cortex,
                stored,
                min_len=int(self.speech_cortex_config.word_min_len),
                delta=float(self.speech_cortex_config.word_weight_delta),
            )
        except Exception as exc:
            if self.config.console_log:
                log_console_line(
                    f"speech_word_reinforce_error step={self._step} err={exc}",
                    log_path=self.config.log_path,
                )
        self._last_speech_babble_step = int(self._step)
        self.speech_executor.start(steps, utterance_id=stored.utterance_id, name=name)
        bias_word = next(
            (t.split(":", 1)[1] for t in tags if str(t).startswith("word:")),
            None,
        )
        self._last_speech_babble_meta = {
            "speech_babble": True,
            "speech_alphabet": stored.alphabet,
            "speech_utterance_id": stored.utterance_id,
            "speech_name": name,
            "speech_score": float(stored.score),
        }
        if bias_word:
            self._last_speech_babble_meta["speech_word_bias"] = bias_word
        if word_hit is not None:
            self._last_speech_babble_meta.update(
                {
                    "speech_word": word_hit.word,
                    "speech_word_weight": float(word_hit.weight),
                    "speech_word_hits": int(word_hit.hits),
                }
            )
        self._push_speech_status(babble=stored.alphabet)
        if self.config.console_log:
            extra = ""
            if bias_word:
                extra += f" bias={bias_word!r}"
            if word_hit is not None:
                extra += (
                    f" word={word_hit.word!r} weight={float(word_hit.weight):.2f}"
                    f" hits={int(word_hit.hits)}"
                )
            log_console_line(
                f"speech_babble step={self._step} id={stored.utterance_id} "
                f"name={name} alphabet={stored.alphabet!r} steps={len(steps)}{extra}",
                log_path=self.config.log_path,
            )
        return True

    def _tick_speech_executor(self, info: dict[str, Any] | None = None) -> None:
        """Advance mouth playback; locomotion bus stays independent."""
        if self._interrupt_pending or self._father_waiting:
            if self.speech_executor.active:
                self.speech_executor.preempt("interrupt")
                self.speech_executor.clear()
                self._push_mouth(np.zeros(7, dtype=np.float32))
            return
        if not self.speech_executor.active:
            return
        frame = self.speech_executor.tick()
        if frame.active:
            self._push_mouth(frame.control)
            if isinstance(info, dict):
                info["speech_playback"] = True
                info["speech_name"] = frame.name
                info["speech_utterance_id"] = frame.utterance_id
        if frame.finished or not frame.active:
            self.speech_executor.clear()
            self._push_mouth(np.zeros(7, dtype=np.float32))

    def _tick_skill_executor(self, info: dict[str, Any] | None = None) -> bool:
        """Advance active skill playback; return True if skill owns the motor."""
        if not self.skill_executor.active and not (
            self.skill_executor.program is not None and not self.skill_executor.finished
        ):
            self._finish_skill_episode_if_needed()
            return False
        if self._interrupt_pending or self._father_waiting:
            self.skill_executor.preempt("interrupt")
            self._finish_skill_episode_if_needed()
            return False
        snap = info if isinstance(info, dict) else {}
        motion = abs(float(snap.get("speed", snap.get("pose_delta", 0.0)) or 0.0))
        if motion <= 1e-6:
            speed_fn = getattr(self.env, "agent_speed", None)
            if callable(speed_fn):
                try:
                    motion = abs(float(speed_fn() or 0.0))
                except Exception:
                    motion = 0.0
        novelty = float(snap.get("visual_novelty", snap.get("novelty", 0.0)) or 0.0)
        damage = float(snap.get("damage", snap.get("bitter", 0.0)) or 0.0)
        self.skill_executor.observe(motion=motion, novelty=novelty, damage=damage)
        frame = self.skill_executor.tick()
        if frame.active and self.act_controller is not None and self.motor is not None:
            self._push_motor_directive(
                mode="act_anything",
                plan_name=frame.plan_name,
                goal=frame.goal,
                reason=frame.reason,
                allow_pivot=True,
            )
            if isinstance(info, dict):
                info["skill_playback"] = True
                info["skill_name"] = frame.skill_name
                info["skill_step"] = frame.step_index
            return True
        self._finish_skill_episode_if_needed()
        return False

    def _apply_father_reply(self, reply: FatherReply) -> None:
        question = self._father_question
        if question is None:
            self._father_waiting = False
            return
        place_id = getattr(self._last_place_node, "id", None)
        lesson = lesson_from_reply(question, reply, place_id=place_id)
        if str(reply.answer or "").strip() and reply.source != "skip":
            self.father_lessons.add(lesson)
            self._father_lesson = lesson
            self._persist_father_lesson(lesson)
            self._handle_father_skill_promote(lesson)
            self._install_father_resume_goal(lesson)
        else:
            self._father_lesson = None
        self._father_waiting = False
        self._father_wait_started = None
        self.father_dialog.send_hide(request_id=int(question.request_id))
        if self.config.console_log:
            print(
                f"father_reply step={self._step} source={reply.source} "
                f"place={place_id} action={lesson.action} "
                f"skill={lesson.skill_name or '-'} "
                f"conf={float(lesson.confidence):.2f} answer={reply.answer[:120]}",
                flush=True,
            )

    def _poll_father_advocate(self) -> bool:
        """Return True when Father is still waiting on a human/LLM reply.

        Motor stays hard-stopped until a reply arrives. If the user focuses
        Display 3, wait forever. If they stay away (or no VIEW), auto Ask LLM
        after ``auto_llm_away_seconds`` (0 = immediately). Safe to call
        multiple times per step.
        """
        if not self._father_waiting or self._father_question is None:
            return False
        if int(getattr(self, "_father_polled_step", -1)) == int(self._step):
            return True
        self._father_polled_step = int(self._step)
        cfg = self.father_config
        incoming = self.father_dialog.pop_reply(int(self._father_request_id))
        if incoming is not None:
            source = str(incoming.get("source") or "human")
            if source == "ask_llm":
                reply = answer_father_question(
                    self.father_client,
                    self._father_question,
                    config=cfg,
                    source="llm",
                )
                self._apply_father_reply(reply)
                return False
            if source == "skip":
                self._apply_father_reply(
                    FatherReply(
                        request_id=int(self._father_request_id),
                        source="skip",
                        answer="",
                        lesson="",
                        risk="medium",
                    )
                )
                return False
            answer = str(incoming.get("answer") or "").strip()
            self._apply_father_reply(
                FatherReply(
                    request_id=int(self._father_request_id),
                    source="human",
                    answer=answer,
                    lesson=answer,
                    risk="medium",
                )
            )
            return False
        self._father_wait_steps += 1
        viewing = getattr(self.father_dialog, "user_on_display3", None)
        # User on Display 3 → unlimited time for the UI.
        if viewing is True:
            return True
        away_s = float(getattr(cfg, "auto_llm_away_seconds", 0.0))
        started = float(self._father_wait_started or time.monotonic())
        elapsed = time.monotonic() - started
        timed_out = False
        reason = "away"
        if away_s >= 0.0:
            # Away or unknown VIEW: auto Ask LLM after the grace window.
            timed_out = elapsed >= away_s
            reason = "away" if viewing is False else "no_view_signal"
        else:
            # Legacy absolute timeout when away-auto is disabled.
            timeout_s = float(getattr(cfg, "timeout_seconds", 0.0) or 0.0)
            if timeout_s > 0.0:
                timed_out = elapsed >= timeout_s
                reason = "timeout"
            elif int(getattr(cfg, "timeout_steps", 0) or 0) > 0:
                timed_out = self._father_wait_steps >= max(1, int(cfg.timeout_steps))
                reason = "timeout_steps"
        if timed_out:
            if self.config.console_log:
                print(
                    f"father_timeout step={self._step} auto_llm reason={reason} "
                    f"after {elapsed:.1f}s",
                    flush=True,
                )
            reply = answer_father_question(
                self.father_client,
                self._father_question,
                config=cfg,
                source="timeout_llm",
            )
            self._apply_father_reply(reply)
            return False
        return True

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
                stm_signals: dict[str, Any] = {
                    "stimulus_interrupt": True,
                    "interrupt_reason": self._interrupt_reason,
                    "cue": (self._interrupt_meta or {}).get("cue"),
                }
                if self._father_lesson is not None:
                    stm_signals.update(self._father_lesson.to_signals())
                self.memory.insert_embedding(
                    embedding=np.zeros(self.embedder.total_dim, dtype=np.float32),
                    signals=stm_signals,
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
                        f"si_saved_patch_stm step={self._step} " f"reason={self._interrupt_reason}",
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
        if self._father_waiting and self._father_question is not None:
            self.father_dialog.send_hide(request_id=int(self._father_question.request_id))
        self._father_waiting = False
        self._father_wait_steps = 0
        self._father_wait_started = None
        self._father_question = None
        self._interrupt_cooldown_until = int(self._step) + int(cfg.cooldown_steps)
        if self.mind_scheduler is not None:
            self.mind_scheduler.invalidate("freeze_cleared")
        self.conditioning.end_freeze()
        # Natural / early unlock: install last accept blend so motion resumes conditioned.
        # Structured Father lessons win on locomotion channels when confidence is high.
        if self._father_lesson is not None:
            self._install_father_resume_goal(self._father_lesson)
        father = self._father_lesson
        if father is not None and float(father.confidence) >= 0.25:
            if self._maybe_start_skill_from_lesson(father):
                self._resume_goal = None
                if self.motor is not None:
                    self._pending_action = self.motor.goal.copy()
                # Persist post-freeze conditioning (acks + hits) for later sessions.
                self._persist_conditioning_to_stm()
                return
        if self._resume_goal is not None and self.motor is not None:
            goal = np.asarray(self._resume_goal, dtype=np.float32).copy()
            plan_name = None
            intent = (
                self._last_mind_info.get("intent")
                if isinstance(self._last_mind_info.get("intent"), dict)
                else {}
            )
            if isinstance(intent, dict):
                plan_name = (intent.get("future") or {}).get("chosen_name")
            if father is not None and float(father.confidence) >= 0.25:
                fields = directive_fields_from_father_lesson(father)
                plan_name = str(fields["plan_name"])
                allow_pivot = bool(fields["allow_pivot"])
                resume_mode = str(fields["mode"])
                if resume_mode == "idle":
                    resume_mode = "still"
                if self.act_controller is not None:
                    self._push_motor_directive(
                        mode=resume_mode,
                        plan_name=plan_name,
                        goal=goal,
                        reason="si_clear_father",
                        allow_pivot=allow_pivot,
                    )
                else:
                    self.motor.set_goal(goal, allow_pivot=allow_pivot)
                    set_goal = getattr(self.env, "set_goal", None)
                    if callable(set_goal):
                        set_goal(goal)
            elif self.act_controller is not None:
                resume_mode = mode_for_plan(plan_name)
                if resume_mode == "idle":
                    resume_mode = "act_anything"
                self._push_motor_directive(
                    mode=resume_mode,
                    plan_name=str(plan_name or "explore"),
                    goal=goal,
                    reason="si_clear",
                )
            else:
                self.motor.set_goal(goal)
                set_goal = getattr(self.env, "set_goal", None)
                if callable(set_goal):
                    set_goal(goal)
            self._pending_action = self.motor.goal.copy()
        elif self._father_lesson is not None and float(self._father_lesson.confidence) >= 0.25:
            self._push_father_motor_directive(self._father_lesson)
            if self.motor is not None:
                self._pending_action = self.motor.goal.copy()
        elif self.act_controller is not None:
            self._push_motor_directive(
                mode="still",
                plan_name="stop",
                goal=STOP_GOAL.copy(),
                reason="si_clear",
            )
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
        if self.act_controller is not None:
            if self.act_controller.mode != "idle":
                self._push_motor_directive(
                    mode="idle",
                    plan_name="idle",
                    goal=STOP_GOAL.copy(),
                    reason=str(self._interrupt_reason or "stimulus_interrupt"),
                )
            action = self.act_controller.apply_tick(observation)
        else:
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
        info["father_waiting"] = bool(self._father_waiting)
        if self._father_waiting and self._father_question is not None:
            info["father_question"] = self._father_question.question
        info["motor_goal"] = self.motor.goal.tolist()
        info["motor_action"] = action.tolist()
        info["architecture"] = "B"
        info["mind_tick"] = bool(mind_tick)
        self._stamp_act_state(info)
        return action

    def _tick_interrupt_freeze(self) -> None:
        """Count down deep-thought freeze; clear + save STM when finished.

        Async: the pause also lasts until a deep thought on the new stimulus has been
        committed (capped by ``async_reframe_max_steps``), so the fly never resumes on
        a plan made before the stimulus.
        """
        if not self._interrupt_pending:
            return
        # Hold the freeze clock while waiting on the Father dialog.
        if self._poll_father_advocate():
            return
        self._interrupt_remaining = max(0, int(self._interrupt_remaining) - 1)
        if self._interrupt_remaining > 0:
            return
        if self.mind_scheduler is not None and self._deep_thoughts_done == 0:
            self._reframe_wait += 1
            if self._reframe_wait <= int(self.config.async_reframe_max_steps):
                return
        self._clear_stimulus_interrupt()

    def _capture_mind_inputs(
        self,
        observation: np.ndarray,
        *,
        sim_time: float | None = None,
        left_eye: np.ndarray | None = None,
        right_eye: np.ndarray | None = None,
    ) -> MindInputs:
        """Snapshot everything a thought reads from the env (the async worker never touches it)."""
        if left_eye is None:
            left_eye = getattr(self.env, "left_eye", None)
        if right_eye is None:
            right_eye = getattr(self.env, "right_eye", None)
        bearing_fn = getattr(self.env, "chem_bearings", None)
        pos_fn = getattr(self.env, "agent_position", None)
        chem_fn = getattr(self.env, "chem_vector", None)
        return MindInputs(
            observation=np.array(observation, copy=True),
            left_eye=None if left_eye is None else np.array(left_eye, copy=True),
            right_eye=None if right_eye is None else np.array(right_eye, copy=True),
            chem_bearings=dict(bearing_fn()) if callable(bearing_fn) else None,
            position=np.asarray(pos_fn(), dtype=np.float32).copy() if callable(pos_fn) else None,
            chem_vector=(
                np.asarray(chem_fn(), dtype=np.float32).copy() if callable(chem_fn) else None
            ),
            step=int(self._step),
            sim_time=float(sim_time if sim_time is not None else self._step * 0.05),
        )

    def _slow_call(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Run a slow mind call (LLM) with the pipeline lock released on the async worker."""
        if not getattr(self._mind_tls, "worker", False):
            return fn(*args, **kwargs)
        self._lock.release()
        try:
            result = fn(*args, **kwargs)
        finally:
            self._lock.acquire()
        # A new stimulus arrived meanwhile: stop before touching judgement / interest state.
        sched = self.mind_scheduler
        if sched is not None and sched.generation != self._mind_tls.generation:
            raise _StaleMindJob()
        return result

    def _run_mind_job(self, think: Any, inputs: MindInputs, generation: int) -> Any:
        """Async worker entry: think under the pipeline lock (released around LLM calls)."""
        with self._lock:
            self._mind_tls.worker = True
            self._mind_tls.generation = generation
            try:
                return think(inputs)
            except _StaleMindJob:
                return None
            finally:
                self._mind_tls.worker = False

    def _submit_mind_job(self, observation: np.ndarray) -> bool:
        """Start the next background thought when the worker is idle (deep while frozen)."""
        sched = self.mind_scheduler
        if sched is None or sched.busy:
            return False
        # Do not think (or move planning) until Father reply arrives.
        if self._father_waiting:
            return False
        deep = bool(self._interrupt_pending)
        min_steps = max(1, int(self.config.async_min_steps))
        if not deep and self._step - self._last_mind_submit_step < min_steps:
            return False
        inputs = self._capture_mind_inputs(observation)
        think = self._think_deep if deep else self._think_mind
        generation = sched.generation
        submitted = sched.submit(
            MIND_JOB_DEEP if deep else MIND_JOB_THINK,
            lambda: self._run_mind_job(think, inputs, generation),
            step=self._step,
        )
        if submitted:
            self._last_mind_submit_step = self._step
        return submitted

    def _poll_mind_job(self, observation: np.ndarray) -> MindJobResult | None:
        """Commit a finished background thought onto the motor (no action this call)."""
        sched = self.mind_scheduler
        if sched is None:
            return None
        result = sched.poll()
        if result is None or result.value is None:
            return None
        if result.kind == MIND_JOB_DEEP:
            if not self._interrupt_pending:
                return None
            self._commit_deep_thought(result.value, observation, act=False)
        else:
            if self._interrupt_pending:
                return None
            self._commit_mind(result.value, observation, act=False)
        return result

    def process_async_step(self, observation: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Async mind step: commit any finished thought, run the motor, submit the next.

        The motor acts every call without waiting for the mind. A stimulus interrupt
        armed by the motor tick discards the in-flight thought; the next job is a deep
        thought on the new stimulus while the fly is held still.
        """
        assert self.mind_scheduler is not None
        committed = self._poll_mind_job(observation)
        action, info = self.process_motor_only(observation)
        if committed is not None:
            info["mind_tick"] = True
            info["deep_thought"] = committed.kind == MIND_JOB_DEEP and bool(self._interrupt_pending)
        self._submit_mind_job(observation)
        info["async_mind"] = {
            **self.mind_scheduler.stats(),
            "lag_steps": None if committed is None else int(self._step - committed.submitted_step),
        }
        return action, info

    def _process_deep_thought(
        self,
        observation: np.ndarray,
        *,
        sim_time: float | None = None,
        left_eye: np.ndarray | None = None,
        right_eye: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Mind tick during interrupt freeze: heighten senses, run devil, no motor."""
        inputs = self._capture_mind_inputs(
            observation, sim_time=sim_time, left_eye=left_eye, right_eye=right_eye
        )
        action, info = self._commit_deep_thought(self._think_deep(inputs), observation)
        assert action is not None
        return action, info

    def _think_deep(self, inputs: MindInputs) -> DeepThought:
        """Deep-thought half that may run on the async worker (no env / motor access)."""
        left_eye = inputs.left_eye
        right_eye = inputs.right_eye
        sensory = self.signals.build(inputs.observation, left_eye=left_eye, right_eye=right_eye)
        cues = None
        visual_nov = None
        if left_eye is not None and right_eye is not None:
            cues = self._vision_cues(left_eye, right_eye)
            if cues is not None:
                sensory.summary.update(cues.to_dict())
            visual_nov = self.visual_novelty.observe(left_eye, right_eye)
            sensory.summary.update(visual_nov.to_dict())

        if inputs.chem_bearings:
            sensory.summary.update(inputs.chem_bearings)

        # Map feedback on raw transducers first; reported rates feed deep thought.
        pre_signal = sensory.to_jsonable()
        if cues is not None:
            pre_signal.update(cues.to_dict())
        if visual_nov is not None:
            pre_signal.update(visual_nov.to_dict())
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
        if self._father_lesson is not None:
            signal_payload.update(self._father_lesson.to_signals())
        elif self._father_waiting and self._father_question is not None:
            signal_payload["father_question"] = self._father_question.question
            signal_payload["father_waiting"] = True
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

        position = inputs.position
        chem_vec = inputs.chem_vector

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

        stm, thought, thought_rec = self._slow_call(
            self.thinking.think_on_embedding,
            embedding=parts.embedding,
            signals=signal_payload,
            affect=affect_payload,
            episode_id=self.config.experiment_id,
            step=inputs.step,
            sim_time=inputs.sim_time,
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
            "signals": signal_payload,
            "vision_cues": (
                place_info.get("vision_cues")
                if place_info.get("vision_cues") is not None
                else (cues.to_dict() if cues else None)
            ),
            "place": place_info.get("place"),
            "mapped_sensors": place_info.get("mapped_sensors"),
            "curiosity": curiosity_step.curiosity,
            "curiosity_gated": curiosity_step.gated_curiosity,
            "curiosity_gate": curiosity_step.gate,
            "explore_bias": explore_bias,
            "coverage": curiosity_step.novelty.coverage,
            "unique_cells": curiosity_step.novelty.unique_cells,
            "exploration_xp": curiosity_step.novelty.exploration_xp,
            "experience_novelty": curiosity_step.novelty.experience_novelty,
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
        if self._father_lesson is not None:
            info.update(self._father_lesson.to_signals())
        elif self._father_question is not None:
            info["father_question"] = self._father_question.question

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
        plan_name = None
        if isinstance(thought.intent, dict):
            plan_name = (thought.intent.get("future") or {}).get("chosen_name")
        return DeepThought(
            info=info,
            judged_goal=np.asarray(judged_goal, dtype=np.float32).copy(),
            l1_goal=np.asarray(l1_goal, dtype=np.float32).copy(),
            accepted=decision == "accept"
            and float(judgement.get("blend_weight", 0.0) or 0.0) > 0.0,
            decision=decision,
            early_unlock=bool(self.conditioning.confirmed()),
            plan_name=plan_name,
            position=position,
        )

    @staticmethod
    def _wall_hazard_for_blocked(info: dict[str, Any]) -> float:
        """Prefer raw wall cue so familiarity fade cannot disarm a real jam."""
        cues = info.get("vision_cues") if isinstance(info.get("vision_cues"), dict) else {}
        place = info.get("place") if isinstance(info.get("place"), dict) else {}
        for src in (cues, place):
            for key in ("wall_hazard_raw", "wall_hazard"):
                if src.get(key) is not None:
                    try:
                        return float(src[key])
                    except (TypeError, ValueError):
                        continue
        return 0.0

    def _observe_physical_into(
        self, info: dict[str, Any], position: np.ndarray | None, action: np.ndarray
    ) -> None:
        speed_fn = getattr(self.env, "agent_speed", None)
        speed = speed_fn() if callable(speed_fn) else None
        wall = self._wall_hazard_for_blocked(info)
        bore_phys = self.boredom.observe_physical(
            position=position,
            action=action,
            speed=speed,
            wall_hazard=wall,
        )
        info["boredom"] = bore_phys.boredom
        info["thought_stuck"] = bore_phys.thought_stuck
        info["physical_stuck"] = bore_phys.physical_stuck
        info["force_locomotion"] = bore_phys.force_locomotion
        info["blocked_streak"] = bore_phys.blocked_streak
        info["blocked_escape"] = bore_phys.escaping
        info["blocked_wall"] = wall

    def _commit_deep_thought(
        self, deep: DeepThought, observation: np.ndarray, *, act: bool = True
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """Main-loop half of deep thought: resume goal, early unlock, hard stop.

        ``act=False`` (async) only updates directives; the caller's motor tick acts.
        """
        info = deep.info
        self._deep_thoughts_done += 1
        if deep.accepted:
            self._resume_goal = deep.judged_goal.copy()
        action: np.ndarray | None = None
        if deep.early_unlock:
            info["stimulus_interrupt_early_unlock"] = True
            info["stimulus_interrupt_pending"] = False
            info["deep_thought"] = False
            # Clear freeze (installs resume goal on accept path via _resume_goal).
            self._clear_stimulus_interrupt()
            accept = deep.decision == "accept"
            if self.motor is not None:
                goal = deep.judged_goal if accept else deep.l1_goal
                if accept:
                    self._resume_goal = goal.copy()
                goal = self._apply_active_investigation(goal.copy(), info)
                plan_name = deep.plan_name
                if self.act_controller is not None:
                    resume_mode = mode_for_plan(plan_name)
                    if resume_mode == "idle":
                        resume_mode = "act_anything"
                    if not accept and abs(float(goal[0])) < 0.08 and abs(float(goal[1])) < 0.08:
                        resume_mode = "still"
                        plan_name = plan_name or "stop"
                    self._push_motor_directive(
                        mode=resume_mode,
                        plan_name=str(plan_name or resume_mode),
                        goal=goal,
                        reason="si_early_unlock",
                    )
                    if act:
                        action = self.act_controller.apply_tick(observation)
                else:
                    self.motor.set_goal(goal)
                    set_goal = getattr(self.env, "set_goal", None)
                    if callable(set_goal):
                        set_goal(goal)
                    if act:
                        action = self.motor.act(observation)
                info["architecture"] = "B"
                info["mind_tick"] = True
                info["stimulus_interrupt"] = False
                self._stamp_act_state(info)
                if action is not None:
                    self._motor_ticks += 1
                    info["motor_goal"] = self.motor.goal.tolist()
                    info["motor_action"] = action.tolist()
            else:
                action = deep.judged_goal.copy() if accept else STOP_GOAL.copy()
                info["architecture"] = "A"
                info["mind_tick"] = True
                info["motor_action"] = action.tolist()
                info["stimulus_interrupt"] = False
        elif self.motor is not None:
            if act:
                action = self._apply_hard_stop(info, observation, mind_tick=True)
                self._motor_ticks += 1
        else:
            action = STOP_GOAL.copy()
            info["architecture"] = "A"
            info["mind_tick"] = True
            info["motor_action"] = action.tolist()
        if action is not None:
            self._observe_physical_into(info, deep.position, action)
            self._pending_action = action.copy()
        self._last_mind_info = dict(info)
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
        if self.mind_scheduler is not None:
            self.mind_scheduler.shutdown()
        self.memory.close()
        self.visual_thinker.close()
        llm_close = getattr(self.thinking.llm, "close", None)
        if callable(llm_close):
            llm_close()
        pg_close = getattr(self.place_graph, "close", None)
        if callable(pg_close):
            pg_close()
        sc_close = getattr(self.skill_cortex, "close", None)
        if callable(sc_close):
            sc_close()
        if self.phrase_grounding is not None:
            self.phrase_grounding.close()
        if self.sentence_compose is not None:
            self.sentence_compose.close()
        ph_close = getattr(self.phrase_store, "close", None)
        if callable(ph_close):
            ph_close()
        sp_close = getattr(self.speech_cortex, "close", None)
        if callable(sp_close):
            sp_close()
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def _apply_sugar_orient(self, sensory: Any, info: dict[str, Any]) -> None:
        """Per-step orient: place-graph route home first, else sugar chemotaxis.

        Pivot is allowed (no forward floor) so the body can turn any angle. The
        mind-pushed goal is restored once both route and sugar search switch off.
        Local contact sugar / an active feed bout beats route so we do not retract
        the proboscis while standing on food (route otherwise forces p0.0).
        """
        info.pop("sugar_orient", None)
        info.pop("sugar_orient_mode", None)
        info.pop("route_home_orient", None)
        if self.motor is None:
            return
        held = self.act_controller is not None and self.act_controller.mode != "act_anything"
        summary = dict(sensory.summary)
        # Feed wins over route (and over held mind goals) so energy can rise.
        if self.sugar_search.feeding_now(summary):
            if self._orient_base is None:
                self._orient_base = self.motor.goal.copy()
            steer = self.sugar_search.steer(summary)
            if steer is not None:
                self.motor.set_goal(
                    orient_goal(self._orient_base, steer.forward, steer.turn, steer.proboscis),
                    allow_pivot=True,
                )
                info["sugar_orient"] = steer.to_list()
                info["sugar_orient_mode"] = (
                    f"feed({steer.ticks_left})" if steer.mode == "feed" else steer.mode
                )
                return
        route = self.route_home.state
        # Route-home always steers (even under a held mind goal) so odor invest cannot yank it.
        if route.active and route.turn is not None:
            if self._orient_base is None:
                self._orient_base = self.motor.goal.copy()
            self.motor.set_goal(
                orient_goal(self._orient_base, float(route.forward), float(route.turn), 0.0),
                allow_pivot=True,
            )
            info["route_home_orient"] = [float(route.forward), float(route.turn), 0.0]
            info["sugar_orient"] = info["route_home_orient"]
            if route.mode == "scan":
                info["sugar_orient_mode"] = (
                    f"scan({float(route.scan_yaw_degrees):+.0f}deg,{route.reason})"
                )
            else:
                info["sugar_orient_mode"] = f"route({route.hops_left})"
            info["route_home"] = route.to_dict()
            return
        steer = None if held else self.sugar_search.steer(summary)
        if steer is None:
            if self._orient_base is not None:
                self.motor.set_goal(self._orient_base)
                self._orient_base = None
            return
        if self._orient_base is None:
            self._orient_base = self.motor.goal.copy()
        self.motor.set_goal(
            orient_goal(self._orient_base, steer.forward, steer.turn, steer.proboscis),
            allow_pivot=True,
        )
        info["sugar_orient"] = steer.to_list()
        info["sugar_orient_mode"] = (
            f"feed({steer.ticks_left})" if steer.mode == "feed" else steer.mode
        )

    def _signals_after_mapped_feedback(
        self,
        sensory: Any,
        *,
        visual_nov: Any | None,
        place_info: dict[str, Any],
    ) -> dict[str, Any]:
        """Rebuild signal payload from sensory summary after mapped-sensor feedback."""
        signal_payload = (
            sensory.to_jsonable()
            if hasattr(sensory, "to_jsonable")
            else dict(getattr(sensory, "summary", {}) or {})
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
            raw["opening_strength"] = float(cues.opening_strength)
        elif "wall_hazard" in signal_payload:
            raw["wall_hazard"] = float(signal_payload.get("wall_hazard", 0.0) or 0.0)
        if "opening_strength" in signal_payload and "opening_strength" not in raw:
            raw["opening_strength"] = float(signal_payload.get("opening_strength", 0.0) or 0.0)
        # Landmark / novelty place splits (break same_place when meadow views match).
        if "visual_novelty" in signal_payload:
            raw["visual_novelty"] = float(signal_payload.get("visual_novelty", 0.0) or 0.0)
        if signal_payload.get("new_stimulus"):
            raw["new_stimulus"] = 1.0
        for channel in (
            "odor",
            "odor_left",
            "odor_right",
            "bitter",
            "sugar",
            "sugar_smell",
        ):
            if channel in signal_payload:
                raw[channel] = float(signal_payload.get(channel, 0.0) or 0.0)

        view = None
        if left_eye is not None and right_eye is not None:
            view = stereo_patch_embedding(left_eye, right_eye)

        smell = smell_vec
        if smell is None and "smell_vec" in signal_payload:
            smell = np.asarray(signal_payload.get("smell_vec"), dtype=np.float32)

        edge_props: dict[str, Any] = {}
        if cues is not None:
            edge_props["opening_bearing"] = float(cues.opening_bearing)
            edge_props["opening_strength"] = float(cues.opening_strength)
        turn = signal_payload.get("motor_turn")
        if turn is None and self.motor is not None:
            goal = getattr(self.motor, "goal", None)
            if goal is not None and len(goal) > 1:
                turn = float(goal[1])
        if turn is not None:
            edge_props["turn"] = float(np.clip(float(turn), -1.0, 1.0))

        # A jammed body stares at an unchanging (so "familiar") wall; keep its hazard raw.
        report = self.mapped_sensors.sense(
            raw=raw,
            view_vec=view,
            smell_vec=smell,
            step=int(self._step),
            now_ms=float(info.get("now_ms") or self._step),
            unattenuated=("wall_hazard",) if self.boredom.state().pushing_blocked else (),
            edge_props=edge_props or None,
        )
        for channel, reading in report.readings.items():
            signal_payload[channel] = float(reading.reported)
            signal_payload[f"{channel}_raw"] = float(reading.raw)
        if report.place is not None:
            self._last_place_node = report.place
            signal_payload["place_id"] = report.place.id
            signal_payload["place_mapped"] = not report.is_new
            signal_payload["place_familiarity"] = float(report.mean_familiarity)
            signal_payload["place_label"] = str(
                (report.place.props or {}).get("label") or report.label
            )
            # Anchor place id → Unity world XYZ (path viz + physical route steering).
            pose = None
            pos_fn = getattr(self.env, "agent_position", None)
            if callable(pos_fn):
                try:
                    pose = pos_fn()
                except Exception:
                    pose = None
            self.route_path.record_place_pose(report.place.id, pose)
            if pose is not None:
                try:
                    xyz = np.asarray(pose, dtype=np.float32).reshape(-1)
                    if xyz.size >= 3 and np.all(np.isfinite(xyz[:3])):
                        world = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
                        props = dict(report.place.props or {})
                        props["world_xyz"] = world
                        report.place.props = props
                        if self.place_graph is not None:
                            self.place_graph.refresh_place(
                                report.place.id,
                                report.place.view_vec,
                                props={"world_xyz": world},
                            )
                except Exception:
                    pass

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
            proposal = self._slow_call(
                propose_critique,
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
            bearing = float(
                signals.get("novelty_bearing", signals.get("target_bearing", 0.0)) or 0.0
            )
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
        # Place-graph route home wins over odor/interest investigation.
        route = info.get("route_home") if isinstance(info.get("route_home"), dict) else None
        if route is None:
            route = self._last_route_home
        if isinstance(route, dict) and route.get("mode") == "route":
            info["investigation"] = None
            info["interest_pending"] = len(self.interests.pending())
            return goal
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
        goal = self.visual_thinker.bias_goal(goal, tick=self._step, wall_hazard=float(wall_hazard))
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
        self._poll_skill_promote_ui()
        self._poll_phrase_grounding()
        if self._interrupt_pending and self._interrupt_remaining > 0:
            if self._poll_father_advocate() or self._father_waiting:
                info: dict[str, Any] = {
                    "father_waiting": True,
                    "father_question": (
                        self._father_question.question if self._father_question else ""
                    ),
                }
                if self._father_lesson is not None:
                    info.update(self._father_lesson.to_signals())
                action = self._apply_hard_stop(info, observation, mind_tick=False)
                return action, info
            return self._process_deep_thought(
                observation,
                sim_time=sim_time,
                left_eye=left_eye,
                right_eye=right_eye,
            )
        inputs = self._capture_mind_inputs(
            observation, sim_time=sim_time, left_eye=left_eye, right_eye=right_eye
        )
        action, info = self._commit_mind(self._think_mind(inputs), observation)
        assert action is not None
        return action, info

    def _think_mind(self, inputs: MindInputs) -> MindThought:
        """Mind-tick half that may run on the async worker (no env / motor access)."""
        left_eye = inputs.left_eye
        right_eye = inputs.right_eye
        sensory = self.signals.build(inputs.observation, left_eye=left_eye, right_eye=right_eye)
        cues = None
        visual_nov = None
        if left_eye is not None and right_eye is not None:
            cues = self._vision_cues(left_eye, right_eye)
            if cues is not None:
                sensory.summary.update(cues.to_dict())
            visual_nov = self.visual_novelty.observe(left_eye, right_eye)
            sensory.summary.update(visual_nov.to_dict())

        # Phase 6: chem gradient bearings from food-search envs (not XYZ in obs).
        if inputs.chem_bearings:
            sensory.summary.update(inputs.chem_bearings)

        if sensory.summary.get("internal_all_zero") and inputs.step == 0:
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
        search = self.sugar_search.observe(signal_payload)
        self._last_sugar_search = search.to_dict()
        if self.sugar_search.feeding_until_sated:
            signal_payload[FEEDING_UNTIL_SATED_KEY] = True
        if search.turn is not None:
            signal_payload["sugar_search_turn"] = search.turn
            signal_payload["sugar_search_mode"] = search.mode

        # Lay place-graph routing onto Unity: live fly pose for world bearing.
        pos_fn = getattr(self.env, "agent_position", None)
        yaw_fn = getattr(self.env, "agent_yaw_degrees", None)
        if callable(pos_fn):
            try:
                fly_xyz = pos_fn()
                if fly_xyz is not None:
                    xyz = np.asarray(fly_xyz, dtype=np.float32).reshape(-1)
                    if xyz.size >= 3 and np.all(np.isfinite(xyz[:3])):
                        # List — STM / LLM json.dumps reject ndarray.
                        signal_payload["fly_xyz"] = [
                            float(xyz[0]),
                            float(xyz[1]),
                            float(xyz[2]),
                        ]
            except Exception:
                pass
        if callable(yaw_fn):
            try:
                fly_yaw = yaw_fn()
                if fly_yaw is not None:
                    signal_payload["fly_yaw_deg"] = float(fly_yaw)
            except Exception:
                pass
        elif self.route_path.last_yaw_deg is not None:
            signal_payload["fly_yaw_deg"] = float(self.route_path.last_yaw_deg)

        route = self.route_home.observe(
            store=self.place_graph,
            place=self._last_place_node,
            signals=signal_payload,
        )
        self._last_route_home = route.to_dict()
        place_info["route_home"] = self._last_route_home
        if route.active and route.turn is not None:
            signal_payload["route_home_turn"] = float(route.turn)
            signal_payload["route_home_mode"] = route.mode
            signal_payload["route_home_forward"] = float(route.forward)
            # Suppress chemotaxis while following the remembered path into the box.
            signal_payload.pop("sugar_search_turn", None)
            signal_payload.pop("sugar_search_mode", None)

        affect = self.affect.estimate(sensory)
        parts = self.embedder.embed_parts(sensory, affect)
        embedding = parts.embedding

        position = inputs.position
        chem_vec = inputs.chem_vector

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
            explore_bias = float(np.clip(explore_bias + 0.55 * visual_nov.visual_novelty, 0.0, 1.0))
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
        if curiosity_step.gate > 0.05 or (
            visual_nov is not None and visual_nov.visual_novelty > 0.15
        ):
            viz_boost = float(visual_nov.visual_novelty) if visual_nov is not None else 0.0
            affect_payload["exploration"] = float(
                np.clip(
                    affect_payload.get("exploration", 0.0) + 0.35 * explore_bias + 0.4 * viz_boost,
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
        stm, thought, thought_rec = self._slow_call(
            self.thinking.think_on_embedding,
            embedding=embedding,
            signals=signal_payload,
            affect=affect_payload,
            episode_id=self.config.experiment_id,
            step=inputs.step,
            sim_time=inputs.sim_time,
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
            "signals": signal_payload,
            "vision_cues": (
                place_info.get("vision_cues")
                if place_info.get("vision_cues") is not None
                else (cues.to_dict() if cues else None)
            ),
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
            "experience_novelty": curiosity_step.novelty.experience_novelty,
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
            "sugar_smell": sensory.summary.get("sugar_smell"),
            "sugar_bearing": sensory.summary.get("sugar_bearing"),
            "sugar_search": self._last_sugar_search,
            "route_home": self._last_route_home,
            **_proboscis_feedback(sensory.summary),
            "antenna_debug": antenna_debug_payload(sensory),
        }

        goal: np.ndarray | None = None
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
                escape_sign=self.boredom.escape_sign,
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
            route = self._last_route_home if isinstance(self._last_route_home, dict) else None
            if route and route.get("mode") in ("route", "scan") and route.get("turn") is not None:
                # Lock motor goal to the remembered path or landmark scan pivot.
                goal = orient_goal(
                    goal,
                    float(route.get("forward", 0.55)),
                    float(route["turn"]),
                    0.0,
                )
                info["route_home_goal"] = True
                info["landmark_scan_goal"] = route.get("mode") == "scan"
            bore_now = self.boredom.state()
            goal = apply_blocked_escape_goal(
                goal,
                bore_now,
                config=self.boredom.config,
                escape_sign=self.boredom.escape_sign,
            )
            info["blocked_escape_goal"] = bool(bore_now.escaping)
        return MindThought(
            info=info,
            goal=goal,
            mind_action=np.asarray(mind_action, dtype=np.float32).copy(),
            plan_name=plan_name,
            sensory=sensory,
            position=position,
        )

    def _commit_mind(
        self, mind: MindThought, observation: np.ndarray, *, act: bool = True
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """Main-loop half of a mind tick: push the plan to the motor.

        ``act=False`` (async) only installs the directive; the caller's motor tick acts.
        """
        info = mind.info
        plan_name = mind.plan_name
        action: np.ndarray | None = None
        if self.motor is not None:
            assert mind.goal is not None
            goal = mind.goal
            escaping = bool(info.get("blocked_escape_goal"))
            # Skill playback owns the bus until done / preempted.
            skill_owns = False
            if escaping:
                self.skill_executor.preempt("blocked_escape")
                self._finish_skill_episode_if_needed()
            else:
                promoted = self._lookup_promoted_skill(str(plan_name or ""))
                if promoted is not None and not self.skill_executor.active:
                    self._start_skill_playback(promoted, candidate=False)
                skill_owns = self._tick_skill_executor(info)
                if not skill_owns:
                    senses = info.get("signals") if isinstance(info.get("signals"), dict) else {}
                    affect = info.get("affect") if isinstance(info.get("affect"), dict) else {}
                    if self._maybe_babble_skill(senses or {}, affect or {}):
                        skill_owns = True
            if not skill_owns:
                # Mind push: think result → act_anything / still directive.
                if self.act_controller is not None:
                    self._push_motor_directive(
                        mode="act_anything" if escaping else mode_for_plan(plan_name),
                        plan_name=str(plan_name or mode_for_plan(plan_name)),
                        goal=goal,
                        reason="blocked_escape" if escaping else "mind_think",
                        allow_pivot=escaping,
                    )
                else:
                    self.motor.set_goal(goal, allow_pivot=escaping)
                    set_goal = getattr(self.env, "set_goal", None)
                    if callable(set_goal):
                        set_goal(goal)
            self._orient_base = None
            info["architecture"] = "B"
            if act:
                if not skill_owns:
                    self._apply_sugar_orient(mind.sensory, info)
                if self.act_controller is not None:
                    action = self.act_controller.apply_tick(observation)
                else:
                    action = self.motor.act(observation)
                self._motor_ticks += 1
                info["motor_goal"] = self.motor.goal.tolist()
                info["motor_action"] = action.tolist()
                info["motor_tracking_error"] = self.motor.tracking_error()
            self._stamp_act_state(info)
        else:
            action = mind.mind_action
            info["architecture"] = "A"
        if action is not None:
            if act:
                self._observe_physical_into(info, mind.position, action)
            self._pending_action = action.copy()
        self._last_mind_info = dict(info)
        return action, info

    def process_motor_only(self, observation: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Motor tick without a new mind thought (architecture B).

        With act-state enabled: track the last mind-pushed directive only — no
        embed, curiosity, place map, or explore replan. Cheap SI cue arming may
        still push ``idle`` onto the same mind→motor bus.
        """
        self._poll_skill_promote_ui()
        info = dict(self._last_mind_info)
        info["mind_tick"] = False
        left_eye = getattr(self.env, "left_eye", None)
        right_eye = getattr(self.env, "right_eye", None)
        sensory = self.signals.build(observation, left_eye=left_eye, right_eye=right_eye)
        affect = self.affect.estimate(sensory)
        info["antenna_debug"] = antenna_debug_payload(sensory)
        position = None
        pos_fn = getattr(self.env, "agent_position", None)
        if callable(pos_fn):
            position = np.asarray(pos_fn(), dtype=np.float32)
        visual_nov = None
        if left_eye is not None and right_eye is not None:
            visual_nov = self.visual_novelty.observe(left_eye, right_eye)
            info["visual_novelty"] = visual_nov.visual_novelty
            info["novelty_bearing"] = visual_nov.novelty_bearing
            info["new_stimulus"] = visual_nov.new_stimulus

        summary = sensory.to_jsonable() if hasattr(sensory, "to_jsonable") else sensory.summary
        info["energy"] = summary.get("energy", sensory.summary.get("energy"))
        info["hunger"] = summary.get("hunger", sensory.summary.get("hunger"))
        info["sugar_smell"] = sensory.summary.get("sugar_smell")
        info["sugar_bearing"] = sensory.summary.get("sugar_bearing")
        info["sugar_search"] = self._last_sugar_search
        info["route_home"] = self._last_route_home
        info.update(_proboscis_feedback(sensory.summary))

        # Candidate novel-vs-STM → mind pushes idle (deep-thought freeze).
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
            if self.skill_executor.active:
                self.skill_executor.preempt("interrupt")
                self._finish_skill_episode_if_needed()
            action = self._apply_hard_stop(info, observation, mind_tick=False)
            self._motor_ticks += 1
            self._observe_physical_into(info, position, action)
            self._pending_action = action.copy()
            return action, info

        if self.motor is None:
            action = (
                self._pending_action
                if self._pending_action is not None
                else np.asarray([0.0, 0.0, 0.5, 0.5, 0.0], dtype=np.float32)
            )
            info["architecture"] = "A"
            self._observe_physical_into(info, position, action)
            self._pending_action = action.copy()
            return action, info

        skill_owns = self._tick_skill_executor(info)
        if not skill_owns:
            self._apply_sugar_orient(sensory, info)
        if self.act_controller is not None:
            action = self.act_controller.apply_tick(observation)
        else:
            action = self.motor.act(observation)
        self._motor_ticks += 1
        info["motor_goal"] = self.motor.goal.tolist()
        info["motor_action"] = action.tolist()
        info["motor_tracking_error"] = self.motor.tracking_error()
        info["architecture"] = "B"
        self._stamp_act_state(info)
        self._observe_physical_into(info, position, action)
        for key in (
            "boredom",
            "thought_stuck",
            "physical_stuck",
            "force_locomotion",
            "blocked_streak",
            "blocked_escape",
        ):
            self._last_mind_info[key] = info[key]
        if visual_nov is not None:
            self._last_mind_info["visual_novelty"] = visual_nov.visual_novelty
            self._last_mind_info["novelty_bearing"] = visual_nov.novelty_bearing
            self._last_mind_info["new_stimulus"] = visual_nov.new_stimulus
        if self.act_controller is not None:
            self._last_mind_info["act_mode"] = info.get("act_mode")
            self._last_mind_info["act_plan"] = info.get("act_plan")
            self._last_mind_info["act_reason"] = info.get("act_reason")
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
        self.route_home.reset()
        self._last_route_home = None
        self._last_place_node = None
        self.route_path.clear()
        self.mind_log.clear()
        self.visual_novelty.reset()
        self.visual_thinker.reset()
        self.judgement.reset()
        self.interests.reset()
        self.conditioning.reset()
        self._hydrate_conditioning_from_stm()
        self._history_baseline.reset()
        self._patch_history.reset()
        self._passive_smell.reset()
        self._status_reframe.reset()
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
        # First thought runs inline so the motor starts with a plan; then the mind is async.
        async_mind = self.mind_scheduler is not None and self.motor is not None
        self._last_mind_submit_step = 0
        self._deep_thoughts_done = 0
        self._reframe_wait = 0
        if self.config.console_log:
            print(
                "live_mind_waiting_first_tick "
                "(first [live …] line appears after the first mind/motor compute; "
                "then each Unity Academy decision)",
                flush=True,
            )
        while step_limit is None or self._step < step_limit:
            tick_start = time.perf_counter()
            with self._lock:
                think_interval = deep_interval if self._interrupt_pending else interval
                should_think = (
                    (self.motor is None)
                    or (self._step % think_interval == 0)
                    or not self._last_mind_info
                )
                if async_mind and self._last_mind_info:
                    action, info = self.process_async_step(obs)
                elif should_think and self._interrupt_pending:
                    if self._poll_father_advocate() or self._father_waiting:
                        info = {
                            "father_waiting": True,
                            "father_question": (
                                self._father_question.question if self._father_question else ""
                            ),
                            "stimulus_interrupt": True,
                        }
                        action = self._apply_hard_stop(info, obs, mind_tick=False)
                    else:
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
                        action[0] = (1.0 - blend) * float(action[0]) + blend * float(
                            explore_action[0]
                        )
                        action[1] = (1.0 - blend) * float(action[1]) + blend * float(
                            explore_action[1]
                        )
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
            # Queue route polyline + throttled place-graph map for Unity viz.
            place_id = getattr(self._last_place_node, "id", None)
            route = self._last_route_home if isinstance(self._last_route_home, dict) else None
            self.route_path.send_route(route, current_place_id=place_id)
            if self.place_graph is not None:
                home_id = (route or {}).get("home_id")
                path = (route or {}).get("path") or ()
                self.route_path.send_place_graph(
                    self.place_graph,
                    current_place_id=place_id,
                    home_id=home_id,
                    route_path=path,
                    fly_id=self.config.fly_id,
                    step=int(self._step),
                )
            # Display 3: human-readable mind / deep-thought panels.
            if last_info.get("mind_tick") or last_info.get("deep_thought"):
                self.mind_log.send_panel(
                    int(self._step),
                    last_info,
                    deep=bool(last_info.get("deep_thought")),
                )
            obs, _reward, terminated, truncated, step_info = self.env.step(action)
            if isinstance(step_info, dict):
                last_info = {**last_info, **step_info}
                self._cache_env_audio(step_info)
            # Parallel speech process (hear/babble every speech_interval; mouth every tick).
            self._speech_parallel_tick(last_info)
            with self._lock:
                self._step += 1
                self._tick_interrupt_freeze()
            if period is not None:
                elapsed = time.perf_counter() - tick_start
                sleep_for = period - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
            if terminated or truncated:
                break
        if self.mind_scheduler is not None:
            self.mind_scheduler.shutdown()
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
            "async_mind": self.mind_scheduler.stats() if async_mind else None,
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
