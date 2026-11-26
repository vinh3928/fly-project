"""LLM agent: compose a short sentence from recalled phrases + drives.

Runs off the motor hot path (thread pool), like PhraseGroundingService.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from flybrain.memory.grounded_phrase import (
    GroundedPhraseConfig,
    GroundedSentence,
)
from flybrain.thinking.phrase_labeler import PhraseLlmClient
from flybrain.thinking.visual_thinker import VisualThinkerConfig


def _clamp01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


_SYSTEM_PROMPT = (
    "You speak for an embodied fruit fly in a chamber world. Given recalled "
    "grounded phrases (what it sees/hears) and drive levels (hunger, novelty, "
    "curiosity, survival risk), write ONE short English sentence (max ~12 "
    "words) that a fly might mean right now. Reply with ONE compact JSON "
    'object only, keys: "sentence" (string), "confidence" (0..1). Prefer '
    "concrete, present-tense wording. Use the phrases; do not invent places "
    "or objects that are not listed."
)


@dataclass
class SentenceLabel:
    sentence: str
    confidence: float
    reason: str = "llm"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentence": self.sentence,
            "confidence": float(self.confidence),
            "reason": self.reason,
        }


def drive_snapshot(
    signals: dict[str, Any] | None, affect: dict[str, Any] | None = None
) -> dict[str, float]:
    """Pull the drive keys used for sentence compose."""
    senses = dict(signals or {})
    aff = dict(affect or {})
    out: dict[str, float] = {}
    for key in (
        "hunger",
        "novelty",
        "visual_novelty",
        "experience_novelty",
        "survival_risk",
        "tendency_curiosity",
        "curiosity_drive",
    ):
        if key in senses:
            try:
                out[key] = _clamp01(float(senses[key]))
            except (TypeError, ValueError):
                continue
    if "curiosity" not in out and "tendency_curiosity" in out:
        out["curiosity"] = out["tendency_curiosity"]
    elif "curiosity" not in out and "curiosity_drive" in out:
        out["curiosity"] = out["curiosity_drive"]
    for key in ("fear", "joy", "exploration"):
        if key in aff:
            try:
                out[key] = _clamp01(float(aff[key]))
            except (TypeError, ValueError):
                continue
    # Prefer a single novelty number for prompts/heuristics.
    if "novelty" not in out:
        for alt in ("visual_novelty", "experience_novelty"):
            if alt in out:
                out["novelty"] = out[alt]
                break
    return out


def heuristic_sentence(
    phrases: list[str],
    drive: dict[str, float] | None = None,
) -> SentenceLabel:
    """Offline / mock compose from phrase list + drives."""
    parts = [" ".join(str(p).lower().split()) for p in phrases if str(p).strip()]
    parts = [p for p in parts if p]
    if not parts:
        return SentenceLabel(sentence="silence.", confidence=0.2, reason="heuristic")
    drives = dict(drive or {})
    hunger = float(drives.get("hunger", 0.0) or 0.0)
    novelty = float(drives.get("novelty", drives.get("visual_novelty", 0.0)) or 0.0)
    risk = float(drives.get("survival_risk", drives.get("fear", 0.0)) or 0.0)
    sugar = next((p for p in parts if "sugar" in p or "pad" in p), None)
    if hunger >= 0.45 and sugar is not None:
        return SentenceLabel(
            sentence=f"I want the {sugar}.",
            confidence=0.55,
            reason="heuristic",
        )
    if risk >= 0.55:
        focus = parts[0]
        return SentenceLabel(
            sentence=f"I fear the {focus}.",
            confidence=0.5,
            reason="heuristic",
        )
    if novelty >= 0.45 and len(parts) >= 2:
        return SentenceLabel(
            sentence=f"I notice {parts[0]} and {parts[1]}.",
            confidence=0.5,
            reason="heuristic",
        )
    if len(parts) >= 2:
        return SentenceLabel(
            sentence=f"{parts[0]} and {parts[1]}.",
            confidence=0.4,
            reason="heuristic",
        )
    return SentenceLabel(sentence=f"{parts[0]}.", confidence=0.35, reason="heuristic")


def parse_sentence_label(text: str, *, fallback: SentenceLabel) -> SentenceLabel:
    match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    if not match:
        return fallback
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return fallback
    sentence = " ".join(str(data.get("sentence") or "").split())
    if not sentence:
        return fallback
    if sentence[-1] not in ".!?":
        sentence = sentence + "."
    conf = _clamp01(float(data.get("confidence", fallback.confidence) or fallback.confidence))
    return SentenceLabel(sentence=sentence[:160], confidence=conf, reason="llm")


def build_compose_prompt(
    *,
    phrases: list[str],
    drive: dict[str, float],
    place_id: str | None,
) -> str:
    phrase_s = ", ".join(repr(p) for p in phrases)
    drive_s = ", ".join(f"{k}={float(v):.2f}" for k, v in sorted(drive.items()))
    return (
        f"phrases=[{phrase_s}]\n"
        f"drives={{{drive_s}}}\n"
        f"place_id={place_id!r}\n"
        "Compose one short sentence JSON."
    )


@dataclass
class _Job:
    draft: GroundedSentence
    left: np.ndarray | None
    right: np.ndarray | None


class SentenceComposeService:
    """Async compose + store; poll() commits onto the skill Neo4j phrase store."""

    def __init__(
        self,
        config: GroundedPhraseConfig,
        store: Any,
        *,
        client: PhraseLlmClient | None = None,
        visual_cfg: VisualThinkerConfig | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self._client = client
        self._visual_cfg = visual_cfg or VisualThinkerConfig(mock=True)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sentence_compose")
        self._future: Future[tuple[GroundedSentence, SentenceLabel]] | None = None
        self._last_compose_step = -10_000
        self.last_label: SentenceLabel | None = None
        self.last_stored: GroundedSentence | None = None

    @property
    def is_pending(self) -> bool:
        return self._future is not None and not self._future.done()

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def poll(self) -> GroundedSentence | None:
        if self._future is None or not self._future.done():
            return None
        fut = self._future
        self._future = None
        try:
            draft, label = fut.result()
        except Exception:
            return None
        self.last_label = label
        if (
            float(label.confidence) < float(self.config.sentence_min_confidence)
            or not label.sentence
        ):
            return None
        draft.sentence = label.sentence
        draft.confidence = float(label.confidence)
        draft.score = max(float(draft.score), float(label.confidence))
        stored = self.store.upsert_sentence(draft)
        self.last_stored = stored
        return stored

    def maybe_submit(
        self,
        *,
        step: int,
        draft: GroundedSentence,
        left_eye: np.ndarray | None = None,
        right_eye: np.ndarray | None = None,
    ) -> bool:
        if not self.config.enabled or not self.config.sentence_compose or self.store is None:
            return False
        if len(draft.phrases) < int(self.config.sentence_min_phrases):
            return False
        if self.is_pending:
            return False
        if int(step) - int(self._last_compose_step) < int(self.config.sentence_cooldown_steps):
            return False
        self._last_compose_step = int(step)
        job = _Job(
            draft=GroundedSentence.from_dict(draft.to_dict()),
            left=None if left_eye is None else np.array(left_eye, copy=True),
            right=None if right_eye is None else np.array(right_eye, copy=True),
        )
        self._future = self._pool.submit(self._run_job, job)
        return True

    def _run_job(self, job: _Job) -> tuple[GroundedSentence, SentenceLabel]:
        fallback = heuristic_sentence(job.draft.phrases, job.draft.drive)
        prompt = build_compose_prompt(
            phrases=job.draft.phrases,
            drive=job.draft.drive,
            place_id=job.draft.place_id,
        )
        if self._client is None:
            return job.draft, fallback
        images: list[np.ndarray] = []
        if self.config.use_vision:
            if job.left is not None:
                images.append(job.left)
            if job.right is not None:
                images.append(job.right)
        try:
            raw = self._client.label(system=_SYSTEM_PROMPT, prompt=prompt, images=images)
            label = parse_sentence_label(raw, fallback=fallback)
        except Exception:
            label = fallback
        return job.draft, label
