"""ML-Agents SideChannel: Unity ear bands in, articulatory mouth + speech status out.

Python ← Unity AUDIO:
  float list: [kind=0, band0..band15, rms, voiced]

Python → Unity MOUTH:
  int kind=1, then 7 float articulatory controls

Python → Unity STATUS:
  int kind=2, string hear_alphabet, string babble_alphabet

Guid must match ``SpeechDebugChannel.cs``.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import numpy as np

from flybrain.speech.articulatory import ARTICULATORY_DIM, AUDIO_BAND_DIM, clip_articulatory

SPEECH_CHANNEL_ID = UUID("b7c8d9e0-1f2a-4b3c-9d4e-5f6a7b8c9d0e")

KIND_AUDIO = 0
KIND_MOUTH = 1
KIND_STATUS = 2


def _ascii(text: str | None) -> str:
    raw = str(text or "")
    return raw.encode("ascii", "replace").decode("ascii")


class SpeechChannel:
    """Thin wrapper: last audio features + send mouth / status to Unity."""

    def __init__(self) -> None:
        self._side = None
        self._OutgoingMessage: Any = None
        self.last_bands: list[float] = [0.0] * AUDIO_BAND_DIM
        self.last_rms: float = 0.0
        self.last_voiced: float = 0.0
        self.last_hear_alphabet: str = ""
        self.last_babble_alphabet: str = ""
        try:
            from mlagents_envs.side_channel.outgoing_message import OutgoingMessage
            from mlagents_envs.side_channel.side_channel import SideChannel
        except ImportError:
            return

        channel_id = SPEECH_CHANNEL_ID
        outer = self

        class _SpeechSide(SideChannel):
            def __init__(self) -> None:
                super().__init__(channel_id)

            def on_message_received(self, msg: Any) -> None:
                try:
                    data = list(msg.read_float32_list())
                except Exception:
                    return
                if not data:
                    return
                kind = int(data[0])
                if kind != KIND_AUDIO:
                    return
                need = 1 + AUDIO_BAND_DIM + 2
                if len(data) < need:
                    return
                outer.last_bands = [float(x) for x in data[1 : 1 + AUDIO_BAND_DIM]]
                outer.last_rms = float(data[1 + AUDIO_BAND_DIM])
                outer.last_voiced = float(data[2 + AUDIO_BAND_DIM])

        self._side = _SpeechSide()
        self._OutgoingMessage = OutgoingMessage

    @property
    def side_channel(self):
        return self._side

    def audio_info(self) -> dict[str, Any]:
        return {
            "audio_bands": list(self.last_bands),
            "audio_rms": float(self.last_rms),
            "audio_voiced": float(self.last_voiced),
        }

    def send_mouth(self, control: np.ndarray | list[float] | None) -> None:
        if self._side is None or self._OutgoingMessage is None:
            return
        ctrl = clip_articulatory(control)
        if ctrl.shape[0] != ARTICULATORY_DIM:
            return
        try:
            msg = self._OutgoingMessage()
            msg.write_int32(int(KIND_MOUTH))
            for value in ctrl.tolist():
                msg.write_float32(float(value))
            self._side.queue_message_to_send(msg)
        except Exception:
            pass

    def send_status(
        self,
        hear: str | None = None,
        babble: str | None = None,
    ) -> None:
        """Push latest speech_hear / speech_babble alphabet labels to Unity overlay."""
        if hear is not None:
            self.last_hear_alphabet = str(hear)
        if babble is not None:
            self.last_babble_alphabet = str(babble)
        if self._side is None or self._OutgoingMessage is None:
            return
        try:
            msg = self._OutgoingMessage()
            msg.write_int32(int(KIND_STATUS))
            msg.write_string(_ascii(self.last_hear_alphabet))
            msg.write_string(_ascii(self.last_babble_alphabet))
            self._side.queue_message_to_send(msg)
        except Exception:
            pass
