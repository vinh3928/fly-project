"""Playback sequencer for speech articulatory programs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from flybrain.speech.articulatory import ARTICULATORY_DIM, clip_articulatory
from flybrain.speech.babble import BabbleStep


@dataclass
class SpeechPlaybackFrame:
    active: bool
    finished: bool
    control: np.ndarray
    step_index: int
    reason: str = "speech_playback"
    utterance_id: str = ""
    name: str = ""


@dataclass
class SpeechExecutor:
    """Advance timed articulatory steps on the mouth bus."""

    steps: list[BabbleStep] = field(default_factory=list)
    utterance_id: str = ""
    name: str = ""
    step_index: int = 0
    ticks_in_step: int = 0
    preempted: bool = False

    @property
    def active(self) -> bool:
        return bool(self.steps) and not self.preempted and not self.finished

    @property
    def finished(self) -> bool:
        if self.preempted or not self.steps:
            return True
        return self.step_index >= len(self.steps)

    def start(
        self,
        steps: list[BabbleStep],
        *,
        utterance_id: str = "",
        name: str = "",
    ) -> None:
        self.steps = list(steps or [])
        self.utterance_id = str(utterance_id or "")
        self.name = str(name or "")
        self.step_index = 0
        self.ticks_in_step = 0
        self.preempted = False

    def preempt(self, reason: str = "preempt") -> None:
        _ = reason
        if self.steps:
            self.preempted = True

    def clear(self) -> None:
        self.steps = []
        self.utterance_id = ""
        self.name = ""
        self.step_index = 0
        self.ticks_in_step = 0
        self.preempted = False

    def tick(self) -> SpeechPlaybackFrame:
        if not self.active:
            return SpeechPlaybackFrame(
                active=False,
                finished=True,
                control=np.zeros(ARTICULATORY_DIM, dtype=np.float32),
                step_index=self.step_index,
                reason="idle",
                utterance_id=self.utterance_id,
                name=self.name,
            )
        step = self.steps[self.step_index]
        ctrl = clip_articulatory(step.control)
        self.ticks_in_step += 1
        if self.ticks_in_step >= max(1, int(step.duration_steps)):
            self.step_index += 1
            self.ticks_in_step = 0
        return SpeechPlaybackFrame(
            active=True,
            finished=self.finished,
            control=ctrl,
            step_index=max(0, self.step_index - (0 if self.ticks_in_step else 1)),
            reason="speech_playback",
            utterance_id=self.utterance_id,
            name=self.name,
        )

    def episode_stats(self) -> dict[str, Any]:
        return {
            "completed": bool(self.finished and not self.preempted and self.steps),
            "utterance_id": self.utterance_id,
            "name": self.name,
            "steps": len(self.steps),
        }
