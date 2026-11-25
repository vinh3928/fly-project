"""Father advocate: question a novel interrupt and learn from the reply.

When stimulus interrupt arms deep thought, the Father asks what the situation
means and how to act. The answer may come from a human (Unity textbox) or an
LLM. Lessons are stored and re-injected when a similar cue returns.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from flybrain.motor.layer import clip_goal
from flybrain.thinking.devils_advocate import (
    HeuristicDevilClient,
    LlmClient,
    _call,
    make_llm_client,
)
from flybrain.thinking.stimulus_interrupt import STOP_GOAL

_FATHER_QUESTION_SYSTEM = (
    "You are the Father advocate for a simulated fly mind. "
    "A novel stimulus interrupted the fly into deep thought. "
    "Ask ONE short clarifying question that helps the fly learn what this "
    "situation means and what it should do next. "
    "Be wise and concrete, not poetic. Reply JSON only: "
    '{"question":"..."}'
)

_FATHER_ANSWER_SYSTEM = (
    "You are the Father advocate answering for a simulated fly mind. "
    "Given the interrupt cue and the question, give ONE short practical "
    "lesson the fly should remember (what it is, whether safe, what to do). "
    "Also choose a motor action the body should take when this lesson is reused: "
    "approach | avoid | taste | idle, a turn_bias in [-1,1] (neg=left, pos=right), "
    "and confidence in [0,1]. "
    "Reply JSON only: "
    '{"answer":"...","lesson":"...","risk":"low|medium|high",'
    '"action":"approach|avoid|taste|idle","turn_bias":0.0,"confidence":0.7}'
)

FATHER_ACTIONS = ("approach", "avoid", "taste", "idle")
_ACTION_ALIASES = {
    "retreat": "avoid",
    "flee": "avoid",
    "back": "avoid",
    "reverse": "avoid",
    "escape": "avoid",
    "advance": "approach",
    "seek": "approach",
    "explore": "approach",
    "forward": "approach",
    "feed": "taste",
    "sample": "taste",
    "probe": "taste",
    "stop": "idle",
    "wait": "idle",
    "freeze": "idle",
    "still": "idle",
}

_QUESTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "father_question",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
            "additionalProperties": False,
        },
    },
}

_ANSWER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "father_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "lesson": {"type": "string"},
                "risk": {"type": "string"},
                "action": {"type": "string"},
                "turn_bias": {"type": "number"},
                "confidence": {"type": "number"},
            },
            "required": ["answer", "lesson", "risk", "action", "turn_bias", "confidence"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class FatherAdvocateConfig:
    """HITL / LLM teaching during stimulus-interrupt deep thought."""

    # Opt-in: live runs enable via mind.yaml; unit tests stay interrupt-clean.
    enabled: bool = False
    # human_or_llm: Unity textbox + Ask-LLM button; llm: auto-answer (fake / headless)
    mode: str = "human_or_llm"
    # Seconds to wait for the user to focus Display 3 after a novelty ask.
    # If still away (or no VIEW signal), auto Ask LLM. While focused on Display 3,
    # wait forever for human / Ask LLM / Skip. Set <0 to disable away-auto.
    auto_llm_away_seconds: float = 5.0
    # Absolute wall-clock fallback when auto_llm_away_seconds < 0 (legacy).
    # Ignored while user_on_display3 is True. 0 = no absolute timeout.
    timeout_seconds: float = 0.0
    # Deprecated step-based timeout (ignored when timeout_seconds > 0).
    timeout_steps: int = 0
    max_question_chars: int = 220
    max_answer_chars: int = 420
    # Reuse mind llm.* unless overridden
    mock: bool | None = None
    max_lessons: int = 64

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> FatherAdvocateConfig:
        data = dict(raw or {})
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class FatherQuestion:
    request_id: int
    step: int
    cue: str
    reason: str
    question: str
    signals_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass
class FatherReply:
    request_id: int
    source: str  # human | llm | timeout_llm | skip
    answer: str
    lesson: str = ""
    risk: str = "medium"
    # Empty/None → inferred from answer text in lesson_from_reply.
    action: str = ""
    turn_bias: float | None = None
    confidence: float | None = None
    # Skill-cortex: name a promoted / new motor program (touch_self, dance, …).
    skill_name: str = ""
    promote_last_candidate: bool = False


@dataclass
class FatherLesson:
    cue: str
    question: str
    answer: str
    lesson: str
    risk: str
    source: str
    step: int
    place_id: int | str | None = None
    action: str = "idle"
    turn_bias: float = 0.0
    confidence: float = 0.5
    skill_name: str = ""
    promote_last_candidate: bool = False

    def to_signals(self) -> dict[str, Any]:
        out = {
            "father_cue": self.cue,
            "father_question": self.question,
            "father_answer": self.answer,
            "father_lesson": self.lesson or self.answer,
            "father_risk": self.risk,
            "father_source": self.source,
            "father_action": self.action,
            "father_turn_bias": float(self.turn_bias),
            "father_confidence": float(self.confidence),
        }
        if self.place_id is not None:
            out["father_place_id"] = self.place_id
        if self.skill_name:
            out["father_skill_name"] = self.skill_name
        if self.promote_last_candidate:
            out["father_promote_last_candidate"] = True
        return out


@dataclass
class FatherLessonStore:
    """Rolling lessons keyed loosely by interrupt cue."""

    max_lessons: int = 64
    _items: list[FatherLesson] = field(default_factory=list)

    def add(self, lesson: FatherLesson) -> None:
        self._items.append(lesson)
        while len(self._items) > int(self.max_lessons):
            self._items.pop(0)

    def latest_for_cue(self, cue: str) -> FatherLesson | None:
        key = str(cue or "").strip().lower()
        for item in reversed(self._items):
            if str(item.cue or "").strip().lower() == key:
                return item
        return None

    def latest_for_place_cue(
        self,
        place_id: int | str | None,
        cue: str,
    ) -> FatherLesson | None:
        """Prefer a lesson taught at this place for ``cue``; else any cue match."""
        if place_id is None:
            return self.latest_for_cue(cue)
        key = str(cue or "").strip().lower()
        pid = str(place_id)
        for item in reversed(self._items):
            if str(item.cue or "").strip().lower() != key:
                continue
            if item.place_id is not None and str(item.place_id) == pid:
                return item
        return self.latest_for_cue(cue)

    def as_list(self) -> list[dict[str, Any]]:
        return [
            {
                "cue": x.cue,
                "question": x.question,
                "answer": x.answer,
                "lesson": x.lesson,
                "risk": x.risk,
                "source": x.source,
                "step": x.step,
                "place_id": x.place_id,
                "action": x.action,
                "turn_bias": float(x.turn_bias),
                "confidence": float(x.confidence),
                "skill_name": x.skill_name,
                "promote_last_candidate": bool(x.promote_last_candidate),
            }
            for x in self._items
        ]


_SKILL_NAME_RE = re.compile(
    r"(?:keep\s+(?:this\s+)?(?:move|skill|dance|gait)\s+as|promote(?:\s+as)?|name\s+it)\s+"
    r"[\"']?([a-z][a-z0-9_]{1,40})[\"']?",
    flags=re.IGNORECASE,
)
_PROMOTE_LAST_RE = re.compile(
    r"\b(promote\s+last|keep\s+this\s+move|keep\s+the\s+candidate|yes\s+keep\s+it)\b",
    flags=re.IGNORECASE,
)


def normalize_skill_name(raw: str | None) -> str:
    s = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]", "", s)
    if len(s) < 2 or s in FATHER_ACTIONS or s in _ACTION_ALIASES:
        return ""
    return s[:40]


def parse_skill_promote_from_text(text: str) -> tuple[str, bool]:
    """Extract ``(skill_name, promote_last)`` from Father teaching text."""
    low = str(text or "")
    name = ""
    match = _SKILL_NAME_RE.search(low)
    if match:
        name = normalize_skill_name(match.group(1))
    promote_last = bool(_PROMOTE_LAST_RE.search(low)) or bool(name)
    if not name:
        # Bare skill tokens often used as action labels.
        for token in ("touch_self", "groom", "dance", "dance_spin"):
            if re.search(rf"\b{token}\b", low, flags=re.IGNORECASE):
                name = token
                break
    return name, promote_last


def normalize_father_action(raw: str | None) -> str:
    s = str(raw or "").strip().lower().replace("-", "_")
    if s in FATHER_ACTIONS:
        return s
    if s in _ACTION_ALIASES:
        return _ACTION_ALIASES[s]
    return "idle"


def normalize_father_risk(raw: str | None) -> str:
    risk = str(raw or "medium").strip().lower()
    return risk if risk in ("low", "medium", "high") else "medium"


def clamp01(value: float, default: float = 0.5) -> float:
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except (TypeError, ValueError):
        return float(default)


def clamp_unit(value: float, default: float = 0.0) -> float:
    try:
        return float(np.clip(float(value), -1.0, 1.0))
    except (TypeError, ValueError):
        return float(default)


def infer_father_structure(
    text: str,
    *,
    risk: str = "medium",
    cue: str = "",
    signals: dict[str, Any] | None = None,
    action: str | None = None,
    turn_bias: float | None = None,
    confidence: float | None = None,
) -> tuple[str, float, float, str]:
    """Fill structured motor fields from LLM/human text + cues."""
    risk_n = normalize_father_risk(risk)
    low = f"{text} {cue}".lower()
    snap = signals if isinstance(signals, dict) else {}
    bitter = float(snap.get("bitter", 0.0) or 0.0)
    sugar = float(snap.get("sugar", 0.0) or 0.0)
    wall = float(snap.get("wall_hazard", 0.0) or 0.0)

    if action is not None and str(action).strip():
        inferred = normalize_father_action(action)
    else:
        if bitter > 0.2 or any(
            w in low for w in ("toxin", "avoid", "retreat", "reverse", "back away", "danger")
        ):
            inferred = "avoid"
        elif any(w in low for w in ("taste", "proboscis", "sample", "feed", "sweet")):
            inferred = "taste"
        elif sugar > 0.2 or any(
            w in low for w in ("approach", "food", "sugar", "seek", "toward", "advance")
        ):
            inferred = "approach"
        elif wall > 0.35 or any(w in low for w in ("wall", "barrier", "opening", "turn")):
            inferred = "avoid"
        else:
            inferred = "idle"

    if turn_bias is None:
        turn = 0.0
        if inferred == "avoid":
            turn = 0.55 if "left" not in low else -0.55
            if "left" in low:
                turn = -0.6
            elif "right" in low:
                turn = 0.6
        elif "left" in low:
            turn = -0.45
        elif "right" in low:
            turn = 0.45
    else:
        turn = clamp_unit(turn_bias)

    if confidence is None:
        conf = 0.55
        if inferred in ("avoid", "taste", "approach"):
            conf = 0.72
        if risk_n == "high":
            conf = max(conf, 0.8)
        elif risk_n == "low":
            conf = max(conf, 0.65)
    else:
        conf = clamp01(confidence)

    if risk_n == "high" and inferred == "approach":
        inferred = "taste"
    return inferred, turn, conf, risk_n


def goal_from_father_structure(
    *,
    action: str,
    turn_bias: float = 0.0,
    confidence: float = 0.5,
    risk: str = "medium",
) -> np.ndarray:
    """Map structured Father lesson → 5-d motor goal."""
    act = normalize_father_action(action)
    conf = clamp01(confidence)
    turn = clamp_unit(turn_bias) * conf
    risk_n = normalize_father_risk(risk)
    if act == "avoid":
        fwd = -0.35 * conf - (0.2 if risk_n == "high" else 0.0)
        if abs(turn) < 0.05:
            turn = 0.55 * conf
        return clip_goal(np.asarray([fwd, turn, 0.5, 0.5, 0.0], dtype=np.float32))
    if act == "approach":
        fwd = 0.45 + 0.35 * conf
        return clip_goal(np.asarray([fwd, turn * 0.5, 0.5, 0.55, 0.0], dtype=np.float32))
    if act == "taste":
        fwd = 0.15 + 0.2 * conf
        return clip_goal(
            np.asarray(
                [fwd, turn * 0.3, 0.5, 0.4, 0.55 + 0.35 * conf],
                dtype=np.float32,
            )
        )
    return STOP_GOAL.copy()


def directive_fields_from_father_lesson(
    lesson: FatherLesson,
) -> dict[str, Any]:
    """Motor-directive kwargs derived from a structured lesson."""
    act = normalize_father_action(lesson.action)
    goal = goal_from_father_structure(
        action=act,
        turn_bias=float(lesson.turn_bias),
        confidence=float(lesson.confidence),
        risk=lesson.risk,
    )
    if act == "idle" or float(lesson.confidence) < 0.25:
        return {
            "mode": "still",
            "plan_name": "father_idle",
            "goal": STOP_GOAL.copy(),
            "allow_pivot": False,
            "reason": "father_lesson",
        }
    mode = "act_anything"
    allow_pivot = act == "avoid"
    return {
        "mode": mode,
        "plan_name": f"father_{act}",
        "goal": goal,
        "allow_pivot": allow_pivot,
        "reason": "father_lesson",
    }


def blend_goal_with_father(
    base: np.ndarray | None,
    lesson: FatherLesson,
) -> np.ndarray:
    """Let Father structure dominate forward/turn/proboscis by confidence."""
    father = goal_from_father_structure(
        action=lesson.action,
        turn_bias=float(lesson.turn_bias),
        confidence=float(lesson.confidence),
        risk=lesson.risk,
    )
    conf = clamp01(lesson.confidence)
    if base is None or normalize_father_action(lesson.action) == "idle":
        return father if conf >= 0.25 else (STOP_GOAL.copy() if base is None else clip_goal(base))
    g = clip_goal(base)
    # Replace locomotion channels; keep height/frequency soft-blended.
    g[0] = (1.0 - conf) * float(g[0]) + conf * float(father[0])
    g[1] = (1.0 - conf) * float(g[1]) + conf * float(father[1])
    g[2] = (1.0 - 0.35 * conf) * float(g[2]) + 0.35 * conf * float(father[2])
    g[3] = (1.0 - 0.35 * conf) * float(g[3]) + 0.35 * conf * float(father[3])
    g[4] = (1.0 - conf) * float(g[4]) + conf * float(father[4])
    return clip_goal(g)


def _trim(text: str, limit: int) -> str:
    s = " ".join(str(text or "").split())
    if len(s) > limit:
        s = s[: max(0, limit - 1)] + "…"
    return s


def _parse_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}


def heuristic_father_question(*, cue: str, reason: str, signals: dict[str, Any]) -> str:
    """Deterministic question when mock / offline."""
    cue_s = str(cue or "novelty").replace("_", " ")
    sugar = float(signals.get("sugar", 0.0) or 0.0)
    bitter = float(signals.get("bitter", 0.0) or 0.0)
    wall = float(signals.get("wall_hazard", 0.0) or 0.0)
    odor = float(signals.get("odor", signals.get("sugar_smell", 0.0)) or 0.0)
    if bitter > 0.2:
        return f"This {cue_s} tastes bitter — is it toxin, and should I retreat?"
    if sugar > 0.2 or odor > 0.25:
        return f"This {cue_s} smells like food — is it safe sugar I should approach?"
    if wall > 0.35:
        return f"This {cue_s} feels like a wall — should I turn toward an opening?"
    if "hunger" in str(reason) or "energy" in str(reason):
        return "My needs changed — should I keep exploring or seek remembered food?"
    return f"What is this new {cue_s}, and what should I do about it?"


def heuristic_father_answer(*, cue: str, question: str, signals: dict[str, Any]) -> FatherReply:
    """Deterministic teaching reply when mock / offline."""
    _ = question
    bitter = float(signals.get("bitter", 0.0) or 0.0)
    sugar = float(signals.get("sugar", 0.0) or 0.0)
    wall = float(signals.get("wall_hazard", 0.0) or 0.0)
    if bitter > 0.2:
        text = "Treat as toxin: stop tasting, reverse, and avoid this place."
        risk = "high"
        action = "avoid"
        turn_bias = 0.65
        confidence = 0.85
    elif sugar > 0.2:
        text = "Likely food: approach carefully, taste, then feed if sweet."
        risk = "low"
        action = "taste"
        turn_bias = 0.0
        confidence = 0.75
    elif wall > 0.35:
        text = "Solid barrier: turn toward openings; do not push forward."
        risk = "medium"
        action = "avoid"
        turn_bias = 0.55
        confidence = 0.7
    else:
        text = f"Note the {cue or 'stimulus'}, sample briefly, then resume with caution."
        risk = "medium"
        action = "approach"
        turn_bias = 0.0
        confidence = 0.55
    action, turn_bias, confidence, risk = infer_father_structure(
        text,
        risk=risk,
        cue=cue,
        signals=signals,
        action=action,
        turn_bias=turn_bias,
        confidence=confidence,
    )
    return FatherReply(
        request_id=0,
        source="llm",
        answer=text,
        lesson=text,
        risk=risk,
        action=action,
        turn_bias=turn_bias,
        confidence=confidence,
    )


def propose_father_question(
    client: LlmClient | None,
    *,
    request_id: int,
    step: int,
    cue: str,
    reason: str,
    signals: dict[str, Any] | None = None,
    config: FatherAdvocateConfig | None = None,
) -> FatherQuestion:
    cfg = config or FatherAdvocateConfig()
    snap = {
        k: signals[k]
        for k in (
            "energy",
            "hunger",
            "sugar",
            "bitter",
            "odor",
            "wall_hazard",
            "visual_novelty",
            "sugar_smell",
        )
        if isinstance(signals, dict) and k in signals
    }
    question = heuristic_father_question(cue=cue, reason=reason, signals=snap)
    if client is not None:
        user = json.dumps(
            {"cue": cue, "reason": reason, "signals": snap},
            sort_keys=True,
        )
        try:
            raw = _call(client, _FATHER_QUESTION_SYSTEM, user, _QUESTION_SCHEMA)
            parsed = _parse_json_object(raw)
            q = str(parsed.get("question") or "").strip()
            if q:
                question = q
        except Exception:  # noqa: BLE001 - never block interrupt on LLM failure
            pass
    return FatherQuestion(
        request_id=int(request_id),
        step=int(step),
        cue=str(cue or ""),
        reason=str(reason or ""),
        question=_trim(question, int(cfg.max_question_chars)),
        signals_snapshot=snap,
    )


def answer_father_question(
    client: LlmClient | None,
    question: FatherQuestion,
    *,
    config: FatherAdvocateConfig | None = None,
    source: str = "llm",
) -> FatherReply:
    cfg = config or FatherAdvocateConfig()
    fallback = heuristic_father_answer(
        cue=question.cue,
        question=question.question,
        signals=question.signals_snapshot,
    )
    fallback.request_id = int(question.request_id)
    fallback.source = source
    if client is None:
        skill_name, promote_last = parse_skill_promote_from_text(
            f"{fallback.answer} {fallback.lesson}"
        )
        return FatherReply(
            request_id=int(question.request_id),
            source=source,
            answer=_trim(fallback.answer, int(cfg.max_answer_chars)),
            lesson=_trim(fallback.lesson, int(cfg.max_answer_chars)),
            risk=fallback.risk,
            action=fallback.action,
            turn_bias=float(fallback.turn_bias),
            confidence=float(fallback.confidence),
            skill_name=skill_name,
            promote_last_candidate=promote_last,
        )
    user = json.dumps(
        {
            "cue": question.cue,
            "reason": question.reason,
            "question": question.question,
            "signals": question.signals_snapshot,
        },
        sort_keys=True,
    )
    try:
        raw = _call(client, _FATHER_ANSWER_SYSTEM, user, _ANSWER_SCHEMA)
        parsed = _parse_json_object(raw)
        answer = str(parsed.get("answer") or parsed.get("lesson") or "").strip()
        lesson = str(parsed.get("lesson") or answer).strip()
        if not answer:
            return fallback
        action, turn_bias, confidence, risk = infer_father_structure(
            answer,
            risk=str(parsed.get("risk") or fallback.risk),
            cue=question.cue,
            signals=question.signals_snapshot,
            action=parsed.get("action"),
            turn_bias=parsed.get("turn_bias"),
            confidence=parsed.get("confidence"),
        )
        skill_name = normalize_skill_name(parsed.get("skill_name"))
        promote_last = bool(parsed.get("promote_last_candidate"))
        if not skill_name:
            skill_name, promote_from_text = parse_skill_promote_from_text(f"{answer} {lesson}")
            promote_last = promote_last or promote_from_text
        return FatherReply(
            request_id=int(question.request_id),
            source=source,
            answer=_trim(answer, int(cfg.max_answer_chars)),
            lesson=_trim(lesson, int(cfg.max_answer_chars)),
            risk=risk,
            action=action,
            turn_bias=turn_bias,
            confidence=confidence,
            skill_name=skill_name,
            promote_last_candidate=promote_last,
        )
    except Exception:  # noqa: BLE001
        return fallback


def make_father_client(
    *,
    mock: bool = True,
    llm_kwargs: dict[str, Any] | None = None,
) -> LlmClient:
    """Father uses a sync client (question/answer are infrequent)."""
    kwargs = dict(llm_kwargs or {})
    # Prefer deterministic mock; otherwise reuse OpenAI-compatible client without background.
    # mind.yaml llm.background must not win — Father asks/answers inline once per interrupt.
    kwargs.pop("background", None)
    if mock:
        return HeuristicDevilClient()
    return make_llm_client(mock=False, background=False, **kwargs)


def lesson_from_reply(
    question: FatherQuestion,
    reply: FatherReply,
    *,
    place_id: int | str | None = None,
) -> FatherLesson:
    action, turn_bias, confidence, risk = infer_father_structure(
        reply.lesson or reply.answer,
        risk=reply.risk,
        cue=question.cue,
        signals=question.signals_snapshot,
        action=reply.action,
        turn_bias=reply.turn_bias,
        confidence=reply.confidence,
    )
    skill_name = normalize_skill_name(reply.skill_name)
    promote_last = bool(reply.promote_last_candidate)
    if not skill_name:
        skill_name, promote_from_text = parse_skill_promote_from_text(
            f"{reply.answer} {reply.lesson}"
        )
        promote_last = promote_last or promote_from_text
    # If Father names a skill as the action label, keep motor action idle and
    # let skill playback own the body (place lesson still stores skill_name).
    if skill_name and normalize_father_action(reply.action) == "idle":
        action = "idle"
    return FatherLesson(
        cue=question.cue,
        question=question.question,
        answer=reply.answer,
        lesson=reply.lesson or reply.answer,
        risk=risk,
        source=reply.source,
        step=int(question.step),
        place_id=place_id,
        action=action,
        turn_bias=turn_bias,
        confidence=confidence,
        skill_name=skill_name,
        promote_last_candidate=promote_last,
    )
