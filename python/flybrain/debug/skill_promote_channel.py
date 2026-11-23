"""ML-Agents SideChannel: skill-promote queue UI for Unity Display 4.

HITL: Python SHOWs candidates; Unity replies accept / ask_llm / refuse / test.
Ask LLM auto-adjudicates accept vs refuse (plus a name) then advances the queue.

Python → Unity SHOW:
  int32 version (=1)
  int32 kind (=0 SHOW)
  int32 request_id
  int32 step
  int32 queue_count
  int32 allow_llm (0/1)
  string skill_id
  string name
  string tags
  string score
  string steps_count
  string suggested_name

Python → Unity HIDE:
  int32 version, kind (=2), request_id

Unity → Python REPLY:
  int32 version, kind (=1), request_id
  int32 source (0=accept, 1=ask_llm, 2=refuse)
  string name

Guid must match ``SkillPromoteDebugChannel.cs``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

SKILL_PROMOTE_CHANNEL_ID = UUID("f6a7b8c9-0d1e-4f2a-3b4c-5d6e7f8091a2")

PROTOCOL_VERSION = 1
KIND_SHOW = 0
KIND_REPLY = 1
KIND_HIDE = 2

SOURCE_ACCEPT = 0
SOURCE_ASK_LLM = 1
SOURCE_REFUSE = 2
SOURCE_TEST = 3

SOURCE_NAMES = {
    SOURCE_ACCEPT: "accept",
    SOURCE_ASK_LLM: "ask_llm",
    SOURCE_REFUSE: "refuse",
    SOURCE_TEST: "test",
}


def _ascii(text: str) -> str:
    return str(text or "").encode("ascii", "replace").decode("ascii")


def pack_show_message(
    *,
    request_id: int,
    step: int,
    queue_count: int,
    skill_id: str,
    name: str,
    tags: str,
    score: str,
    steps_count: str,
    suggested_name: str = "",
    allow_llm: bool = True,
) -> bytes:
    import struct

    parts = [
        struct.pack(
            "<iiiiii",
            int(PROTOCOL_VERSION),
            int(KIND_SHOW),
            int(request_id),
            int(step),
            int(queue_count),
            1 if allow_llm else 0,
        )
    ]
    for value in (skill_id, name, tags, score, steps_count, suggested_name):
        raw = _ascii(value).encode("ascii")
        parts.append(struct.pack("<i", len(raw)))
        parts.append(raw)
    return b"".join(parts)


def pack_hide_message(*, request_id: int) -> bytes:
    import struct

    return struct.pack("<iii", int(PROTOCOL_VERSION), int(KIND_HIDE), int(request_id))


def parse_reply_message(raw: bytes) -> dict[str, Any]:
    import struct

    if len(raw) < 16:
        return {
            "version": 0,
            "kind": KIND_REPLY,
            "request_id": 0,
            "source": "refuse",
            "name": "",
        }
    version, kind, request_id, source = struct.unpack_from("<iiii", raw, 0)
    (n,) = struct.unpack_from("<i", raw, 16)
    name = raw[20 : 20 + max(0, n)].decode("ascii", "replace")
    return {
        "version": int(version),
        "kind": int(kind),
        "request_id": int(request_id),
        "source": SOURCE_NAMES.get(int(source), "accept"),
        "name": name,
    }


def suggest_skill_name(*, tags: list[str] | None = None, current_name: str = "") -> str:
    """Neutral name hint for Ask-LLM / offline — HITL Accept is the source of truth.

    Builds a short snake_case label from primitive tags (forward/turn/extend/…)
    when present; otherwise strips a ``candidate_`` prefix. Does not invent
    dance/touch labels.
    """
    skip = {"babble", "combo", "candidate", "move", "skill", "father", "display4"}
    prims: list[str] = []
    for raw in tags or []:
        t = str(raw or "").strip().lower().replace("-", "_")
        t = "".join(ch for ch in t if ch.isalnum() or ch == "_")
        if not t or t in skip or t in prims:
            continue
        prims.append(t)
        if len(prims) >= 3:
            break
    if prims:
        return "_".join(prims)[:48]
    base = str(current_name or "skill").strip().lower().replace("-", "_")
    base = "".join(ch for ch in base if ch.isalnum() or ch == "_")
    if base.startswith("candidate_") and len(base) > 10:
        return base[len("candidate_") :][:48] or "move"
    return base or "move"


@dataclass
class SkillAdjudication:
    """LLM / heuristic verdict for Display 4 Ask LLM."""

    decision: str  # "accept" | "refuse"
    name: str = ""
    reason: str = ""
    source: str = "heuristic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "name": self.name,
            "reason": self.reason,
            "source": self.source,
        }


def summarize_skill_for_judge(program: Any) -> dict[str, Any]:
    """Compact motion summary for adjudication prompts / heuristics."""
    import numpy as np

    tags = list(getattr(program, "tags", None) or [])
    steps = list(getattr(program, "steps", None) or [])
    goals = []
    durations = []
    for step in steps:
        g = getattr(step, "goal", None)
        if g is not None:
            goals.append(np.asarray(g, dtype=np.float32).reshape(-1)[:5])
        durations.append(int(getattr(step, "duration_steps", 0) or 0))
    if goals:
        mat = np.stack(goals, axis=0)
        mean = mat.mean(axis=0)
        span = mat.max(axis=0) - mat.min(axis=0)
        motion = float(np.mean(np.abs(mat[:, :2])))
        variety = float(np.mean(span[:2]))
        prob = float(np.mean(mat[:, 4])) if mat.shape[1] > 4 else 0.0
    else:
        mean = np.zeros(5, dtype=np.float32)
        span = np.zeros(5, dtype=np.float32)
        motion = 0.0
        variety = 0.0
        prob = 0.0
    return {
        "skill_id": str(getattr(program, "skill_id", "") or ""),
        "name": str(getattr(program, "name", "") or ""),
        "tags": tags,
        "score": float(getattr(program, "score", 0.0) or 0.0),
        "n_steps": len(steps),
        "total_duration": int(sum(durations)),
        "mean_forward": float(mean[0]),
        "mean_turn": float(mean[1]),
        "mean_proboscis": float(prob),
        "motion": motion,
        "variety": variety,
        "span_forward": float(span[0]),
        "span_turn": float(span[1]),
    }


def heuristic_adjudicate_skill(
    summary: dict[str, Any],
    *,
    min_score: float = 0.4,
) -> SkillAdjudication:
    """Deterministic accept/refuse used for mock LLM and parse fallbacks."""
    score = float(summary.get("score") or 0.0)
    motion = float(summary.get("motion") or 0.0)
    variety = float(summary.get("variety") or 0.0)
    n_steps = int(summary.get("n_steps") or 0)
    name = suggest_skill_name(
        tags=list(summary.get("tags") or []),
        current_name=str(summary.get("name") or ""),
    )
    if n_steps <= 0:
        return SkillAdjudication(
            decision="refuse",
            name="",
            reason="empty program",
            source="heuristic",
        )
    if score < float(min_score) * 0.75 and motion < 0.12 and variety < 0.15:
        return SkillAdjudication(
            decision="refuse",
            name="",
            reason=f"low score={score:.2f} and weak motion/variety",
            source="heuristic",
        )
    if score >= float(min_score) or (motion >= 0.2 and variety >= 0.2):
        return SkillAdjudication(
            decision="accept",
            name=name,
            reason=f"score={score:.2f} motion={motion:.2f} variety={variety:.2f}",
            source="heuristic",
        )
    if score < float(min_score):
        return SkillAdjudication(
            decision="refuse",
            name="",
            reason=f"score={score:.2f} below min={float(min_score):.2f}",
            source="heuristic",
        )
    return SkillAdjudication(
        decision="accept",
        name=name,
        reason=f"borderline keep score={score:.2f}",
        source="heuristic",
    )


def parse_adjudication_text(raw: str, *, fallback: SkillAdjudication) -> SkillAdjudication:
    """Parse LLM JSON (or accept/refuse keywords) into an adjudication."""
    import json
    import re

    text = str(raw or "").strip()
    if not text:
        return fallback
    # Prefer fenced / bare JSON object.
    match = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
    blob = match.group(0) if match else text
    try:
        data = json.loads(blob)
    except Exception:
        data = None
    if isinstance(data, dict):
        decision = str(data.get("decision") or data.get("verdict") or "").strip().lower()
        if decision in ("promote", "keep", "yes", "accept"):
            decision = "accept"
        elif decision in ("reject", "retire", "no", "refuse", "deny"):
            decision = "refuse"
        if decision not in ("accept", "refuse"):
            return fallback
        name = str(data.get("name") or data.get("skill_name") or "").strip()
        reason = str(data.get("reason") or data.get("rationale") or "").strip()
        if decision == "accept" and not name:
            name = fallback.name
        return SkillAdjudication(
            decision=decision,
            name=name if decision == "accept" else "",
            reason=reason or fallback.reason,
            source="llm",
        )
    low = text.lower()
    if any(k in low for k in ("refuse", "reject", "retire", "discard")):
        return SkillAdjudication(
            decision="refuse",
            name="",
            reason=text[:160],
            source="llm",
        )
    if any(k in low for k in ("accept", "promote", "keep")):
        return SkillAdjudication(
            decision="accept",
            name=fallback.name,
            reason=text[:160],
            source="llm",
        )
    return fallback


def adjudicate_skill_promote(
    program: Any,
    *,
    client: Any | None = None,
    min_score: float = 0.4,
    mock: bool = True,
) -> SkillAdjudication:
    """Judge whether Display 4 should accept or refuse a candidate skill.

    When ``mock`` or no client, uses a deterministic heuristic. Otherwise asks
    the LLM for JSON ``{decision, name, reason}`` and falls back on parse miss.
    """
    summary = summarize_skill_for_judge(program)
    fallback = heuristic_adjudicate_skill(summary, min_score=min_score)
    if mock or client is None or not hasattr(client, "complete"):
        return fallback
    system = (
        "You adjudicate fly motor skill candidates for a skill library. "
        "Decide accept or refuse. Accept only if the motion looks intentional, "
        "reusable, and non-damaging. If accept, invent a short snake_case name "
        "from the primitives (not dance/touch stereotypes unless clearly fitting). "
        "Reply with JSON only: "
        '{"decision":"accept"|"refuse","name":"snake_case_or_empty","reason":"short"}'
    )
    user = (
        "Candidate skill summary:\n"
        f"{summary}\n"
        f"Minimum keep score hint: {float(min_score):.2f}\n"
        "Adjudicate now."
    )
    try:
        raw = client.complete(system, user)
    except Exception as exc:
        out = SkillAdjudication(
            decision=fallback.decision,
            name=fallback.name,
            reason=f"llm_error:{exc}; {fallback.reason}",
            source="heuristic",
        )
        return out
    return parse_adjudication_text(str(raw or ""), fallback=fallback)


class SkillPromoteChannel:
    """Bidirectional SideChannel wrapper; safe no-op without mlagents_envs."""

    def __init__(self) -> None:
        self._impl: Any = None
        self._OutgoingMessage: Any = None
        self._pending_replies: list[dict[str, Any]] = []
        self.last_show: dict[str, Any] | None = None
        try:
            from mlagents_envs.side_channel.side_channel import (
                IncomingMessage,
                OutgoingMessage,
                SideChannel,
            )

            channel_id = SKILL_PROMOTE_CHANNEL_ID
            outer = self

            class _Impl(SideChannel):
                def __init__(self) -> None:
                    super().__init__(channel_id)

                def on_message_received(self, msg: IncomingMessage) -> None:
                    version = int(msg.read_int32())
                    kind = int(msg.read_int32())
                    request_id = int(msg.read_int32())
                    if kind != KIND_REPLY:
                        return
                    source = int(msg.read_int32())
                    name = str(msg.read_string() or "")
                    outer._pending_replies.append(
                        {
                            "version": version,
                            "kind": kind,
                            "request_id": request_id,
                            "source": SOURCE_NAMES.get(source, "accept"),
                            "name": name,
                        }
                    )

            self._impl = _Impl()
            self._OutgoingMessage = OutgoingMessage
        except Exception:
            self._impl = None
            self._OutgoingMessage = None

    @property
    def side_channel(self) -> Any:
        return self._impl

    def send_show(
        self,
        *,
        request_id: int,
        step: int,
        queue_count: int,
        skill_id: str,
        name: str,
        tags: str = "",
        score: str = "0",
        steps_count: str = "0",
        suggested_name: str = "",
        allow_llm: bool = True,
    ) -> None:
        payload = {
            "request_id": int(request_id),
            "step": int(step),
            "queue_count": int(queue_count),
            "skill_id": _ascii(skill_id),
            "name": _ascii(name),
            "tags": _ascii(tags),
            "score": _ascii(score),
            "steps_count": _ascii(steps_count),
            "suggested_name": _ascii(suggested_name),
            "allow_llm": bool(allow_llm),
        }
        self.last_show = payload
        if self._impl is None or self._OutgoingMessage is None:
            return
        msg = self._OutgoingMessage()
        msg.write_int32(int(PROTOCOL_VERSION))
        msg.write_int32(int(KIND_SHOW))
        msg.write_int32(int(request_id))
        msg.write_int32(int(step))
        msg.write_int32(int(queue_count))
        msg.write_int32(1 if allow_llm else 0)
        for value in (
            skill_id,
            name,
            tags,
            score,
            steps_count,
            suggested_name,
        ):
            msg.write_string(_ascii(value))
        self._impl.queue_message_to_send(msg)

    def send_hide(self, *, request_id: int) -> None:
        if self._impl is None or self._OutgoingMessage is None:
            return
        msg = self._OutgoingMessage()
        msg.write_int32(int(PROTOCOL_VERSION))
        msg.write_int32(int(KIND_HIDE))
        msg.write_int32(int(request_id))
        self._impl.queue_message_to_send(msg)

    def pop_reply(self, request_id: int | None = None) -> dict[str, Any] | None:
        if not self._pending_replies:
            return None
        if request_id is None:
            return self._pending_replies.pop(0)
        for i, item in enumerate(self._pending_replies):
            if int(item.get("request_id", -1)) == int(request_id):
                return self._pending_replies.pop(i)
        return None

    def inject_reply_for_tests(
        self,
        *,
        request_id: int,
        source: str = "accept",
        name: str = "",
    ) -> None:
        self._pending_replies.append(
            {
                "version": PROTOCOL_VERSION,
                "kind": KIND_REPLY,
                "request_id": int(request_id),
                "source": source,
                "name": _ascii(name),
            }
        )

    def clear(self) -> None:
        self._pending_replies.clear()
        self.last_show = None
