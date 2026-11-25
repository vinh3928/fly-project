"""ML-Agents SideChannel: Father-advocate ask/reply dialog for Unity Display 3.

Python → Unity ASK:
  int32 version (=1)
  int32 kind (=0 ASK)
  int32 request_id
  int32 step
  int32 allow_llm (0/1)
  string cue
  string question

Python → Unity HIDE:
  int32 version
  int32 kind (=2 HIDE)
  int32 request_id

Unity → Python REPLY:
  int32 version
  int32 kind (=1 REPLY)
  int32 request_id
  int32 source (0=human, 1=ask_llm, 2=skip)
  string answer

Unity → Python VIEW:
  int32 version, kind (=3 VIEW), request_id
  int32 viewing (0/1) — 1 when Game view focuses Display 3

Guid must match ``FatherDialogDebugChannel.cs``.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

FATHER_DIALOG_CHANNEL_ID = UUID("e5f6a7b8-9c0d-4e1f-2a3b-4c5d6e7f8091")

PROTOCOL_VERSION = 1
KIND_ASK = 0
KIND_REPLY = 1
KIND_HIDE = 2
KIND_VIEW = 3

SOURCE_HUMAN = 0
SOURCE_ASK_LLM = 1
SOURCE_SKIP = 2

SOURCE_NAMES = {
    SOURCE_HUMAN: "human",
    SOURCE_ASK_LLM: "ask_llm",
    SOURCE_SKIP: "skip",
}


def _ascii(text: str) -> str:
    return str(text or "").encode("ascii", "replace").decode("ascii")


def pack_ask_message(
    *,
    request_id: int,
    step: int,
    cue: str,
    question: str,
    allow_llm: bool = True,
) -> bytes:
    import struct

    cue_b = _ascii(cue).encode("ascii")
    q_b = _ascii(question).encode("ascii")
    return (
        struct.pack(
            "<iiiii",
            int(PROTOCOL_VERSION),
            int(KIND_ASK),
            int(request_id),
            int(step),
            1 if allow_llm else 0,
        )
        + struct.pack("<i", len(cue_b))
        + cue_b
        + struct.pack("<i", len(q_b))
        + q_b
    )


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
            "source": "skip",
            "answer": "",
        }
    version, kind, request_id, source = struct.unpack_from("<iiii", raw, 0)
    (n,) = struct.unpack_from("<i", raw, 16)
    answer = raw[20 : 20 + max(0, n)].decode("ascii", "replace")
    return {
        "version": int(version),
        "kind": int(kind),
        "request_id": int(request_id),
        "source": SOURCE_NAMES.get(int(source), "human"),
        "answer": answer,
    }


class FatherDialogChannel:
    """Bidirectional SideChannel wrapper; safe no-op without mlagents_envs."""

    def __init__(self) -> None:
        self._impl: Any = None
        self._OutgoingMessage: Any = None
        self._pending_replies: list[dict[str, Any]] = []
        self.last_ask: dict[str, Any] | None = None
        # None = no VIEW yet; True/False from Unity Display 3 focus.
        self.user_on_display3: bool | None = None
        try:
            from mlagents_envs.side_channel.side_channel import (
                IncomingMessage,
                OutgoingMessage,
                SideChannel,
            )

            channel_id = FATHER_DIALOG_CHANNEL_ID
            outer = self

            class _Impl(SideChannel):
                def __init__(self) -> None:
                    super().__init__(channel_id)

                def on_message_received(self, msg: IncomingMessage) -> None:
                    version = int(msg.read_int32())
                    kind = int(msg.read_int32())
                    request_id = int(msg.read_int32())
                    if kind == KIND_VIEW:
                        viewing = int(msg.read_int32())
                        outer.user_on_display3 = bool(viewing)
                        return
                    if kind != KIND_REPLY:
                        return
                    source = int(msg.read_int32())
                    answer = str(msg.read_string() or "")
                    outer._pending_replies.append(
                        {
                            "version": version,
                            "kind": kind,
                            "request_id": request_id,
                            "source": SOURCE_NAMES.get(source, "human"),
                            "answer": answer,
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

    def send_ask(
        self,
        *,
        request_id: int,
        step: int,
        cue: str,
        question: str,
        allow_llm: bool = True,
    ) -> None:
        payload = {
            "request_id": int(request_id),
            "step": int(step),
            "cue": _ascii(cue),
            "question": _ascii(question),
            "allow_llm": bool(allow_llm),
        }
        self.last_ask = payload
        self.user_on_display3 = None
        if self._impl is None or self._OutgoingMessage is None:
            return
        msg = self._OutgoingMessage()
        msg.write_int32(int(PROTOCOL_VERSION))
        msg.write_int32(int(KIND_ASK))
        msg.write_int32(int(request_id))
        msg.write_int32(int(step))
        msg.write_int32(1 if allow_llm else 0)
        msg.write_string(_ascii(cue))
        msg.write_string(_ascii(question))
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
        answer: str,
        source: str = "human",
    ) -> None:
        """Unit-test helper when no Unity is connected."""
        self._pending_replies.append(
            {
                "version": PROTOCOL_VERSION,
                "kind": KIND_REPLY,
                "request_id": int(request_id),
                "source": source,
                "answer": _ascii(answer),
            }
        )

    def inject_view_for_tests(self, *, request_id: int, viewing: bool) -> None:
        _ = request_id
        self.user_on_display3 = bool(viewing)

    def clear(self) -> None:
        self._pending_replies.clear()
        self.last_ask = None
        self.user_on_display3 = None
