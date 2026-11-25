"""Speech babble: sample short articulatory sequences (no ML speech model)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from flybrain.speech.articulatory import ARTICULATORY_DIM, clip_articulatory

PRIMITIVE_SILENCE = "silence"
PRIMITIVE_AH = "ah"
PRIMITIVE_EE = "ee"
PRIMITIVE_OO = "oo"
PRIMITIVE_MM = "mm"
PRIMITIVE_BA = "ba"
PRIMITIVE_PA = "pa"
PRIMITIVE_SS = "ss"

PRIMITIVES: tuple[str, ...] = (
    PRIMITIVE_SILENCE,
    PRIMITIVE_AH,
    PRIMITIVE_EE,
    PRIMITIVE_OO,
    PRIMITIVE_MM,
    PRIMITIVE_BA,
    PRIMITIVE_PA,
    PRIMITIVE_SS,
)

# Rough inverse of articulatory→letter so weighted words can drive mouth poses.
LETTER_TO_PRIMITIVE: dict[str, str] = {
    "a": PRIMITIVE_AH,
    "e": PRIMITIVE_EE,
    "i": PRIMITIVE_EE,
    "o": PRIMITIVE_OO,
    "u": PRIMITIVE_OO,
    "m": PRIMITIVE_MM,
    "b": PRIMITIVE_BA,
    "p": PRIMITIVE_PA,
    "s": PRIMITIVE_SS,
    "t": PRIMITIVE_SS,
    "n": PRIMITIVE_MM,
    "l": PRIMITIVE_AH,
}


@dataclass
class SpeechBabbleConfig:
    enabled: bool = True
    min_novelty: float = 0.25
    max_survival_risk: float = 0.45
    max_hunger: float = 0.55
    cooldown_steps: int = 160
    max_steps: int = 6
    min_steps: int = 2
    duration_range: tuple[int, int] = (4, 16)
    # P(ignore Word.weight and sample pure random primitives).
    word_bias_explore: float = 0.35

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> SpeechBabbleConfig:
        data = dict(raw or {})
        dur = data.get("duration_range", [4, 16])
        if isinstance(dur, (list, tuple)) and len(dur) >= 2:
            lo, hi = int(dur[0]), int(dur[1])
        else:
            lo, hi = 4, 16
        if hi < lo:
            lo, hi = hi, lo
        explore = float(data.get("word_bias_explore", 0.35))
        explore = float(np.clip(explore, 0.0, 1.0))
        return cls(
            enabled=bool(data.get("enabled", True)),
            min_novelty=float(data.get("min_novelty", 0.25)),
            max_survival_risk=float(data.get("max_survival_risk", 0.45)),
            max_hunger=float(data.get("max_hunger", 0.55)),
            cooldown_steps=max(0, int(data.get("cooldown_steps", 160))),
            max_steps=max(1, int(data.get("max_steps", 6))),
            min_steps=max(1, int(data.get("min_steps", 2))),
            duration_range=(lo, hi),
            word_bias_explore=explore,
        )


@dataclass
class SpeechBabbleGate:
    ok: bool
    reason: str = ""


def babble_gate(
    *,
    config: SpeechBabbleConfig,
    step: int,
    last_babble_step: int,
    executor_active: bool,
    novelty: float,
    survival_risk: float,
    hunger: float,
    explore_tendency: float = 0.5,
) -> SpeechBabbleGate:
    if not config.enabled:
        return SpeechBabbleGate(False, "disabled")
    if executor_active:
        return SpeechBabbleGate(False, "executor_active")
    if int(step) - int(last_babble_step) < int(config.cooldown_steps):
        return SpeechBabbleGate(False, "cooldown")
    if float(survival_risk) > float(config.max_survival_risk):
        return SpeechBabbleGate(False, "survival")
    if float(hunger) > float(config.max_hunger):
        return SpeechBabbleGate(False, "hunger")
    if float(novelty) < float(config.min_novelty) and float(explore_tendency) < 0.55:
        return SpeechBabbleGate(False, "low_novelty")
    return SpeechBabbleGate(True, "ok")


def _ctrl(
    jaw: float,
    lip: float,
    tf: float,
    th: float,
    voice: float,
    pitch: float = 0.0,
    amp: float = 0.7,
) -> np.ndarray:
    return clip_articulatory([jaw, lip, tf, th, voice, pitch, amp])


def primitive_control(name: str, rng: np.random.Generator) -> np.ndarray:
    key = str(name or "").strip().lower()
    if key == PRIMITIVE_SILENCE:
        return _ctrl(0.2, 0.2, 0.3, 0.3, 0.0, 0.0, 0.0)
    if key == PRIMITIVE_AH:
        return _ctrl(0.85, 0.7, 0.4, 0.35, 0.9, float(rng.uniform(-0.2, 0.2)), 0.85)
    if key == PRIMITIVE_EE:
        return _ctrl(0.45, 0.65, 0.85, 0.85, 0.9, float(rng.uniform(-0.1, 0.3)), 0.8)
    if key == PRIMITIVE_OO:
        return _ctrl(0.4, 0.35, 0.2, 0.75, 0.9, float(rng.uniform(-0.3, 0.1)), 0.8)
    if key == PRIMITIVE_MM:
        return _ctrl(0.25, 0.05, 0.4, 0.4, 0.85, 0.0, 0.55)
    if key == PRIMITIVE_BA:
        return _ctrl(0.35, 0.1, 0.4, 0.45, 0.8, 0.0, 0.75)
    if key == PRIMITIVE_PA:
        return _ctrl(0.35, 0.08, 0.4, 0.45, 0.15, 0.0, 0.7)
    if key == PRIMITIVE_SS:
        return _ctrl(0.3, 0.4, 0.85, 0.25, 0.2, 0.2, 0.65)
    return _ctrl(0.5, 0.5, 0.5, 0.5, 0.6, 0.0, 0.5)


@dataclass
class BabbleStep:
    control: np.ndarray
    duration_steps: int
    primitive: str


def _normalize_word_weights(
    word_weights: Sequence[tuple[str, float]] | None,
) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    if not word_weights:
        return out
    for raw_word, raw_w in word_weights:
        word = "".join(ch for ch in str(raw_word or "").lower() if ch.isalpha())
        weight = float(raw_w)
        if not word or weight <= 0.0:
            continue
        out.append((word, weight))
    return out


def _pick_weighted_word(
    rng: np.random.Generator, words: list[tuple[str, float]]
) -> str | None:
    if not words:
        return None
    weights = np.asarray([w for _, w in words], dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        return None
    probs = weights / total
    idx = int(rng.choice(len(words), p=probs))
    return words[idx][0]


def _duration(rng: np.random.Generator, config: SpeechBabbleConfig) -> int:
    lo, hi = config.duration_range
    return max(1, int(rng.integers(lo, hi + 1)))


def _steps_from_word(
    word: str,
    rng: np.random.Generator,
    config: SpeechBabbleConfig,
) -> list[BabbleStep]:
    letters = [ch for ch in str(word).lower() if ch.isalpha()]
    if not letters:
        return []
    # Fit into configured babble length bounds.
    max_n = max(1, int(config.max_steps))
    min_n = max(1, min(int(config.min_steps), max_n))
    if len(letters) > max_n:
        letters = letters[:max_n]
    while len(letters) < min_n:
        # Pad with a light vowel so short words still meet min_steps.
        letters.append("a")
    steps: list[BabbleStep] = []
    voiced = [p for p in PRIMITIVES if p != PRIMITIVE_SILENCE]
    for ch in letters:
        prim = LETTER_TO_PRIMITIVE.get(ch)
        if prim is None:
            prim = str(voiced[int(rng.integers(0, len(voiced)))])
        steps.append(
            BabbleStep(
                control=primitive_control(prim, rng),
                duration_steps=_duration(rng, config),
                primitive=prim,
            )
        )
    return steps


def _steps_random(
    rng: np.random.Generator, config: SpeechBabbleConfig
) -> list[BabbleStep]:
    n = int(rng.integers(config.min_steps, config.max_steps + 1))
    steps: list[BabbleStep] = []
    for _ in range(n):
        prim = str(PRIMITIVES[int(rng.integers(0, len(PRIMITIVES)))])
        steps.append(
            BabbleStep(
                control=primitive_control(prim, rng),
                duration_steps=_duration(rng, config),
                primitive=prim,
            )
        )
    return steps


def sample_babble_utterance(
    rng: np.random.Generator,
    *,
    config: SpeechBabbleConfig | None = None,
    word_weights: Sequence[tuple[str, float]] | None = None,
) -> tuple[str, list[BabbleStep], list[str]]:
    """Return (anonymous_name, steps, tags).

    When ``word_weights`` has positive entries, sample a target word with
    probability proportional to weight (unless ``word_bias_explore`` fires),
    then drive articulatory primitives from that word's letters.
    """
    cfg = config or SpeechBabbleConfig()
    tags = ["babble", "speech"]
    words = _normalize_word_weights(word_weights)
    target: str | None = None
    explore = float(np.clip(cfg.word_bias_explore, 0.0, 1.0))
    if words and float(rng.random()) >= explore:
        target = _pick_weighted_word(rng, words)
    if target:
        steps = _steps_from_word(target, rng, cfg)
        tags.extend(["word_bias", f"word:{target}"])
        tags.extend(s.primitive for s in steps)
    else:
        steps = _steps_random(rng, cfg)
        tags.extend(s.primitive for s in steps)
    if not steps:
        steps = _steps_random(rng, cfg)
    name = f"candidate_utter_{int(rng.integers(0, 1_000_000)):06d}"
    return name, steps, list(dict.fromkeys(tags))


def expand_controls(steps: list[BabbleStep]) -> list[np.ndarray]:
    """Expand timed babble steps into a per-tick articulatory sequence."""
    out: list[np.ndarray] = []
    for step in steps:
        ctrl = clip_articulatory(step.control)
        for _ in range(max(1, int(step.duration_steps))):
            out.append(ctrl.copy())
    if not out:
        out.append(np.zeros(ARTICULATORY_DIM, dtype=np.float32))
    return out
