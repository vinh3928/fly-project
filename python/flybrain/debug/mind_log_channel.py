"""ML-Agents SideChannel: human-readable mind / deep-thought logs for Unity Display 3.

Python → Unity message:
  int32 version (=1)
  int32 kind     (0=mind, 1=deep)
  int32 step
  string text    (ASCII; non-ASCII replaced)

Guid must match ``MindLogDebugChannel.cs``.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

# Fixed Guid — must match MindLogDebugChannel.cs
MIND_LOG_CHANNEL_ID = UUID("d4e5f6a7-8b9c-4d0e-1f2a-3b4c5d6e7f80")

PROTOCOL_VERSION = 1
KIND_MIND = 0
KIND_DEEP = 1
MAX_THOUGHT_CHARS = 420
MAX_DEVIL_CHARS = 280
MAX_PANEL_CHARS = 1800


def _ascii(text: str) -> str:
    """SideChannel strings are ASCII-only."""
    return str(text or "").encode("ascii", "replace").decode("ascii")


def _clean(text: Any, limit: int) -> str:
    s = " ".join(str(text or "").split())
    if len(s) > limit:
        s = s[: max(0, limit - 1)] + "…"
    return _ascii(s)


def format_mind_panel(step: int, info: dict[str, Any] | None, *, deep: bool = False) -> str:
    """Human-readable mind / deep-thought block for Display 3."""
    data = info if isinstance(info, dict) else {}
    intent = data.get("intent") if isinstance(data.get("intent"), dict) else {}
    future = intent.get("future") if isinstance(intent.get("future"), dict) else {}
    plan = str(future.get("chosen_name") or data.get("plan_name") or "-")
    energy = data.get("energy")
    hunger = data.get("hunger")
    e_s = f"{float(energy):.2f}" if energy is not None else "-"
    h_s = f"{float(hunger):.2f}" if hunger is not None else "-"

    judgement = data.get("judgement") if isinstance(data.get("judgement"), dict) else {}
    judge = str(judgement.get("decision") or "-")
    if judge == "accept" and judgement.get("blend_weight") is not None:
        judge = f"accept(w={float(judgement['blend_weight']):.2f})"

    route = data.get("route_home") if isinstance(data.get("route_home"), dict) else {}
    route_s = "-"
    if route:
        mode = str(route.get("mode") or "off")
        if mode in ("route", "scan", "handoff"):
            route_s = f"{mode} hops={int(route.get('hops_left') or 0)}"
            reason = route.get("reason")
            if reason:
                route_s += f" ({reason})"

    cue = ""
    if deep:
        meta = data.get("interrupt_meta") if isinstance(data.get("interrupt_meta"), dict) else {}
        si = data.get("interrupt_debug") if isinstance(data.get("interrupt_debug"), dict) else {}
        cue_name = meta.get("cue") or si.get("cue") or data.get("interrupt_reason") or ""
        if cue_name:
            cue = f" cue={cue_name}"

    father = ""
    if data.get("father_waiting"):
        father = f"father?: {_clean(data.get('father_question'), 160) or '-'}"
    elif data.get("father_answer") or data.get("father_lesson"):
        father = f"father: {_clean(data.get('father_lesson') or data.get('father_answer'), 200)}"

    tag = "DEEP" if deep else "MIND"
    lines = [
        f"[{tag} {int(step)}]{cue}",
        f"plan={plan}  E={e_s} H={h_s}  judge={judge}",
        f"route={route_s}",
        f"thought: {_clean(data.get('refined_thought'), MAX_THOUGHT_CHARS) or '-'}",
        f"devil: {_clean(data.get('devil_critique'), MAX_DEVIL_CHARS) or '-'}",
    ]
    if father:
        lines.append(father)
    panel = "\n".join(lines)
    if len(panel) > MAX_PANEL_CHARS:
        panel = panel[: MAX_PANEL_CHARS - 1] + "…"
    return panel


def pack_mind_log_message(
    text: str,
    *,
    kind: int = KIND_MIND,
    step: int = 0,
) -> bytes:
    """Pack a mind-log payload (for tests; live path uses OutgoingMessage)."""
    import struct

    body = _ascii(text).encode("ascii")
    return (
        struct.pack("<iii", int(PROTOCOL_VERSION), int(kind), int(step))
        + struct.pack("<i", len(body))
        + body
    )


def parse_mind_log_message(raw: bytes) -> dict[str, Any]:
    """Parse a packed mind-log payload (mirrors Unity IncomingMessage)."""
    import struct

    if len(raw) < 12:
        return {"version": 0, "kind": KIND_MIND, "step": 0, "text": ""}
    version, kind, step = struct.unpack_from("<iii", raw, 0)
    (n,) = struct.unpack_from("<i", raw, 12)
    text = raw[16 : 16 + max(0, n)].decode("ascii", "replace")
    return {
        "version": int(version),
        "kind": int(kind),
        "step": int(step),
        "text": text,
    }


class MindLogDebugChannel:
    """SideChannel wrapper; safe no-op if mlagents_envs is unavailable."""

    def __init__(self) -> None:
        self._impl: Any = None
        self._OutgoingMessage: Any = None
        self.last_text: str = ""
        self.last_kind: int = KIND_MIND
        self.last_step: int = 0
        try:
            from mlagents_envs.side_channel.side_channel import (
                IncomingMessage,
                OutgoingMessage,
                SideChannel,
            )

            channel_id = MIND_LOG_CHANNEL_ID

            class _Impl(SideChannel):
                def __init__(self) -> None:
                    super().__init__(channel_id)

                def on_message_received(self, msg: IncomingMessage) -> None:
                    # Display 3 is one-way Python → Unity for now.
                    del msg

            self._impl = _Impl()
            self._OutgoingMessage = OutgoingMessage
        except Exception:
            self._impl = None
            self._OutgoingMessage = None

    @property
    def side_channel(self) -> Any:
        return self._impl

    def send_panel(
        self,
        step: int,
        info: dict[str, Any] | None,
        *,
        deep: bool = False,
    ) -> str:
        """Format + queue a mind/deep panel for the next ``env.step``."""
        kind = KIND_DEEP if deep else KIND_MIND
        text = format_mind_panel(int(step), info, deep=deep)
        return self.send_text(text, kind=kind, step=int(step))

    def send_text(self, text: str, *, kind: int = KIND_MIND, step: int = 0) -> str:
        clean = _ascii(text)
        self.last_text = clean
        self.last_kind = int(kind)
        self.last_step = int(step)
        if self._impl is not None and self._OutgoingMessage is not None:
            msg = self._OutgoingMessage()
            msg.write_int32(int(PROTOCOL_VERSION))
            msg.write_int32(int(kind))
            msg.write_int32(int(step))
            msg.write_string(clean)
            self._impl.queue_message_to_send(msg)
        return clean

    def clear(self) -> None:
        self.send_text("", kind=KIND_MIND, step=0)
        self.last_text = ""
