"""LLM agent: map eye frames + hear bands + alphabet → single/multi-word phrase.

Runs off the motor hot path (thread pool), like VisualThinker.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from flybrain.memory.grounded_phrase import GroundedPhrase, GroundedPhraseConfig
from flybrain.thinking.visual_thinker import (
    OllamaVisionClient,
    VisualLlmClient,
    VisualThinkerConfig,
)


def _clamp01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


_SYSTEM_PROMPT = (
    "You name what an embodied fruit fly is experiencing. You get left/right "
    "eye frames (optional), what it just heard (band energies / RMS), and the "
    "letters of its current utterance (babble or hear alphabet). Reply with "
    "ONE compact JSON object only, keys: "
    '"phrase" (1-6 English words naming the scene+sound+utterance meaning), '
    '"confidence" (0..1). Prefer concrete words (sugar pad, wind, bee buzz, '
    "chamber song). If unclear, use a short onomatopoeia or the alphabet itself."
)


@dataclass
class PhraseLabel:
    phrase: str
    confidence: float
    reason: str = "llm"

    def to_dict(self) -> dict[str, Any]:
        return {
            "phrase": self.phrase,
            "confidence": float(self.confidence),
            "reason": self.reason,
        }


class PhraseLlmClient(Protocol):
    def label(
        self,
        *,
        system: str,
        prompt: str,
        images: list[np.ndarray],
    ) -> str: ...


class VisionPhraseClient:
    """Reuse Ollama vision chat completions for phrase labeling."""

    def __init__(self, vision: VisualLlmClient, image_size: int = 128) -> None:
        self._vision = vision
        self._image_size = int(image_size)

    def label(self, *, system: str, prompt: str, images: list[np.ndarray]) -> str:
        # OllamaVisionClient.inspect matches this signature.
        return self._vision.inspect(system=system, prompt=prompt, images=images)


def make_phrase_client(
    *,
    mock: bool,
    visual_cfg: VisualThinkerConfig | None = None,
) -> PhraseLlmClient | None:
    if mock:
        return None
    cfg = visual_cfg or VisualThinkerConfig(mock=False)
    live = OllamaVisionClient(cfg)
    return VisionPhraseClient(live, image_size=cfg.image_size)


def heuristic_phrase_label(
    *,
    alphabet: str,
    rms: float,
    voiced: float,
    cue: str = "",
) -> PhraseLabel:
    """Offline / mock label from alphabet + audio energy."""
    letters = "".join(ch for ch in str(alphabet or "").lower() if ch.isalpha())
    cue_l = str(cue or "").lower()
    if "sugar" in cue_l or "pad" in cue_l:
        phrase = "sugar pad"
        conf = 0.55
    elif float(rms) >= 0.35 and float(voiced) < 0.35:
        phrase = "chamber song" if letters else "loud hum"
        conf = 0.45
    elif float(rms) >= 0.2:
        phrase = f"{letters} sound" if len(letters) >= 2 else "soft noise"
        conf = 0.4
    elif len(letters) >= 2:
        # Space multi-char alphabet into a pseudo-word token.
        phrase = letters if len(letters) <= 8 else letters[:8]
        conf = 0.35
    else:
        phrase = letters or "silence"
        conf = 0.2
    return PhraseLabel(phrase=phrase, confidence=conf, reason="heuristic")


def parse_phrase_label(text: str, *, fallback: PhraseLabel) -> PhraseLabel:
    match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    if not match:
        return fallback
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return fallback
    phrase = " ".join(str(data.get("phrase") or "").lower().split())
    if not phrase:
        return fallback
    conf = _clamp01(float(data.get("confidence", fallback.confidence) or fallback.confidence))
    return PhraseLabel(phrase=phrase[:80], confidence=conf, reason="llm")


def build_label_prompt(
    *,
    alphabet: str,
    rms: float,
    voiced: float,
    bands: list[float] | None,
    cue: str,
    place_id: str | None,
) -> str:
    band_s = ",".join(f"{float(x):.2f}" for x in (bands or [])[:16])
    return (
        f"utterance_alphabet={alphabet!r}\n"
        f"audio_rms={float(rms):.3f} voiced={float(voiced):.3f}\n"
        f"audio_bands=[{band_s}]\n"
        f"cue={cue!r} place_id={place_id!r}\n"
        "Name the grounded meaning as phrase + confidence JSON."
    )


@dataclass
class _Job:
    draft: GroundedPhrase
    left: np.ndarray | None
    right: np.ndarray | None
    bands: list[float]
    cue: str


class PhraseGroundingService:
    """Async label + store; poll() commits results onto the skill Neo4j store."""

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
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="phrase_ground")
        self._future: Future[tuple[GroundedPhrase, PhraseLabel]] | None = None
        self._last_label_step = -10_000
        self.last_label: PhraseLabel | None = None
        self.last_stored: GroundedPhrase | None = None
        self.last_match: GroundedPhrase | None = None

    @property
    def is_pending(self) -> bool:
        return self._future is not None and not self._future.done()

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def poll(self) -> GroundedPhrase | None:
        if self._future is None or not self._future.done():
            return None
        fut = self._future
        self._future = None
        try:
            phrase, label = fut.result()
        except Exception:
            return None
        self.last_label = label
        if float(label.confidence) < float(self.config.min_confidence) or not label.phrase:
            return None
        phrase.phrase = label.phrase
        phrase.confidence = float(label.confidence)
        phrase.score = max(float(phrase.score), float(label.confidence))
        stored = self.store.upsert_phrase(phrase)
        self.last_stored = stored
        return stored

    def maybe_submit(
        self,
        *,
        step: int,
        draft: GroundedPhrase,
        left_eye: np.ndarray | None,
        right_eye: np.ndarray | None,
        bands: list[float] | None = None,
        cue: str = "",
    ) -> bool:
        if not self.config.enabled or self.store is None:
            return False
        if self.is_pending:
            return False
        if int(step) - int(self._last_label_step) < int(self.config.cooldown_steps):
            return False
        self._last_label_step = int(step)
        job = _Job(
            draft=GroundedPhrase.from_dict(draft.to_dict()),
            left=None if left_eye is None else np.array(left_eye, copy=True),
            right=None if right_eye is None else np.array(right_eye, copy=True),
            bands=[float(x) for x in (bands or [])],
            cue=str(cue or ""),
        )
        self._future = self._pool.submit(self._run_job, job)
        return True

    def _run_job(self, job: _Job) -> tuple[GroundedPhrase, PhraseLabel]:
        fallback = heuristic_phrase_label(
            alphabet=job.draft.alphabet,
            rms=job.draft.rms,
            voiced=job.draft.voiced,
            cue=job.cue,
        )
        prompt = build_label_prompt(
            alphabet=job.draft.alphabet,
            rms=job.draft.rms,
            voiced=job.draft.voiced,
            bands=job.bands,
            cue=job.cue,
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
            label = parse_phrase_label(raw, fallback=fallback)
        except Exception:
            label = fallback
        return job.draft, label
