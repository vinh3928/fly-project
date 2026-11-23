"""Motor babbling: sample open combos of timed 5-d goal primitives.

Naming is left to HITL (Display 4 Accept / Ask LLM / Father). Babble only
proposes anonymous candidate programs — no dance/touch hard labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import numpy as np

from flybrain.memory.skill_cortex import (
    SkillBabbleConfig,
    SkillContext,
    SkillProgram,
    SkillStep,
)
from flybrain.motor.layer import clip_goal

# Atomic motor primitives mixed into candidate programs.
PRIMITIVE_PAUSE = "pause"
PRIMITIVE_FORWARD = "forward"
PRIMITIVE_BACK = "back"
PRIMITIVE_TURN_L = "turn_left"
PRIMITIVE_TURN_R = "turn_right"
PRIMITIVE_PIVOT = "pivot"
PRIMITIVE_EXTEND = "extend"
PRIMITIVE_CROUCH = "crouch"
PRIMITIVE_RAISE = "raise"

PRIMITIVES: tuple[str, ...] = (
    PRIMITIVE_PAUSE,
    PRIMITIVE_FORWARD,
    PRIMITIVE_BACK,
    PRIMITIVE_TURN_L,
    PRIMITIVE_TURN_R,
    PRIMITIVE_PIVOT,
    PRIMITIVE_EXTEND,
    PRIMITIVE_CROUCH,
    PRIMITIVE_RAISE,
)

# Backward-compatible aliases (old motif API → still accepted as a bias hint only).
MOTIF_GROOM = "groom"
MOTIF_DANCE = "dance"
MOTIF_TOUCH = "touch_self"


def _rand_duration(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _goal(
    forward: float = 0.0,
    turn: float = 0.0,
    height: float = 0.5,
    freq: float = 0.5,
    proboscis: float = 0.0,
) -> np.ndarray:
    return clip_goal(np.asarray([forward, turn, height, freq, proboscis], dtype=np.float32))


def primitive_goal(name: str, rng: np.random.Generator) -> np.ndarray:
    """Map a primitive name to a 5-d motor goal (with light randomness)."""
    key = str(name or "").strip().lower()
    if key == PRIMITIVE_PAUSE:
        return _goal(0.0, float(rng.uniform(-0.08, 0.08)), 0.5, 0.25, 0.0)
    if key == PRIMITIVE_FORWARD:
        return _goal(float(rng.uniform(0.35, 0.85)), float(rng.uniform(-0.2, 0.2)), 0.5, 0.7, 0.0)
    if key == PRIMITIVE_BACK:
        return _goal(
            float(rng.uniform(-0.55, -0.2)), float(rng.uniform(-0.15, 0.15)), 0.5, 0.55, 0.0
        )
    if key == PRIMITIVE_TURN_L:
        return _goal(
            float(rng.uniform(0.05, 0.35)), float(rng.uniform(-0.95, -0.45)), 0.5, 0.65, 0.0
        )
    if key == PRIMITIVE_TURN_R:
        return _goal(float(rng.uniform(0.05, 0.35)), float(rng.uniform(0.45, 0.95)), 0.5, 0.65, 0.0)
    if key == PRIMITIVE_PIVOT:
        sign = -1.0 if rng.random() < 0.5 else 1.0
        return _goal(0.05, float(sign * rng.uniform(0.55, 1.0)), 0.5, 0.5, 0.0)
    if key == PRIMITIVE_EXTEND:
        return _goal(
            float(rng.uniform(-0.05, 0.15)),
            float(rng.uniform(-0.25, 0.25)),
            float(rng.uniform(0.35, 0.65)),
            float(rng.uniform(0.2, 0.45)),
            float(rng.uniform(0.65, 1.0)),
        )
    if key == PRIMITIVE_CROUCH:
        return _goal(0.05, float(rng.uniform(-0.2, 0.2)), float(rng.uniform(0.15, 0.35)), 0.4, 0.0)
    if key == PRIMITIVE_RAISE:
        return _goal(0.05, float(rng.uniform(-0.2, 0.2)), float(rng.uniform(0.65, 0.9)), 0.45, 0.0)
    # Unknown → mild explore nudge.
    return _goal(float(rng.uniform(0.1, 0.4)), float(rng.uniform(-0.3, 0.3)), 0.5, 0.55, 0.0)


def _bias_weights(hint: str | None) -> np.ndarray:
    """Optional soft bias from legacy motif names; default is uniform over primitives."""
    w = np.ones(len(PRIMITIVES), dtype=np.float64)
    hint_l = str(hint or "").strip().lower()
    if hint_l in (MOTIF_GROOM, MOTIF_TOUCH, "groom", "touch"):
        for i, p in enumerate(PRIMITIVES):
            if p in (PRIMITIVE_PAUSE, PRIMITIVE_EXTEND, PRIMITIVE_CROUCH, PRIMITIVE_RAISE):
                w[i] = 3.0
            elif p in (PRIMITIVE_FORWARD, PRIMITIVE_BACK):
                w[i] = 0.4
    elif hint_l in (MOTIF_DANCE, "dance", "spin"):
        for i, p in enumerate(PRIMITIVES):
            if p in (PRIMITIVE_TURN_L, PRIMITIVE_TURN_R, PRIMITIVE_PIVOT, PRIMITIVE_FORWARD):
                w[i] = 3.0
            elif p == PRIMITIVE_EXTEND:
                w[i] = 0.3
    w = w / w.sum()
    return w


def sample_combo_program(
    rng: np.random.Generator,
    *,
    min_steps: int,
    max_steps: int,
    duration_range: tuple[int, int],
    bias: str | None = None,
) -> tuple[list[SkillStep], list[str]]:
    """Build a random sequence of motor primitives → timed goals + tag list."""
    n = int(rng.integers(min_steps, max_steps + 1))
    lo, hi = duration_range
    weights = _bias_weights(bias)
    picks = list(rng.choice(PRIMITIVES, size=n, replace=True, p=weights))
    # Avoid a whole program of identical pauses.
    if all(p == PRIMITIVE_PAUSE for p in picks) and n > 1:
        picks[0] = PRIMITIVE_FORWARD
        picks[-1] = str(rng.choice([PRIMITIVE_TURN_L, PRIMITIVE_TURN_R, PRIMITIVE_EXTEND]))
    steps: list[SkillStep] = []
    for prim in picks:
        steps.append(
            SkillStep(
                goal=primitive_goal(prim, rng),
                duration_steps=_rand_duration(rng, lo, hi),
            )
        )
    # Stable unique tags in appearance order (for UI / Ask-LLM hints only).
    tags = list(dict.fromkeys(["babble", "combo", *[str(p) for p in picks]]))
    return steps, tags


def sample_groom_program(
    rng: np.random.Generator,
    *,
    min_steps: int,
    max_steps: int,
    duration_range: tuple[int, int],
) -> list[SkillStep]:
    """Compat wrapper: biased combo (extend/pause heavy), not a fixed motif."""
    steps, _ = sample_combo_program(
        rng,
        min_steps=min_steps,
        max_steps=max_steps,
        duration_range=duration_range,
        bias=MOTIF_TOUCH,
    )
    return steps


def sample_dance_program(
    rng: np.random.Generator,
    *,
    min_steps: int,
    max_steps: int,
    duration_range: tuple[int, int],
) -> list[SkillStep]:
    """Compat wrapper: biased combo (turn/pivot heavy), not a fixed motif."""
    steps, _ = sample_combo_program(
        rng,
        min_steps=min_steps,
        max_steps=max_steps,
        duration_range=duration_range,
        bias=MOTIF_DANCE,
    )
    return steps


def sample_babble_program(
    rng: np.random.Generator | None = None,
    *,
    config: SkillBabbleConfig | None = None,
    motif: str | None = None,
) -> tuple[str, list[SkillStep], list[str]]:
    """Return ``(temp_name, steps, tags)`` as an open primitive combo.

    ``motif`` is an optional soft bias only (legacy API). Names stay anonymous
    (`candidate_move_*`); HITL / Ask LLM / Father supply the real label.
    """
    cfg = config or SkillBabbleConfig()
    gen = rng or np.random.default_rng()
    lo, hi = cfg.duration_range
    steps, tags = sample_combo_program(
        gen,
        min_steps=cfg.min_steps,
        max_steps=cfg.max_steps,
        duration_range=(lo, hi),
        bias=motif,
    )
    # Light noise so candidates are not identical.
    noisy: list[SkillStep] = []
    for step in steps:
        g = np.asarray(step.goal, dtype=np.float32).copy()
        g[0] = float(np.clip(g[0] + gen.normal(0.0, 0.05), -1.0, 1.0))
        g[1] = float(np.clip(g[1] + gen.normal(0.0, 0.08), -1.0, 1.0))
        g[2] = float(np.clip(g[2] + gen.normal(0.0, 0.04), 0.0, 1.0))
        g[3] = float(np.clip(g[3] + gen.normal(0.0, 0.04), 0.0, 1.0))
        g[4] = float(np.clip(g[4] + gen.normal(0.0, 0.05), 0.0, 1.0))
        noisy.append(SkillStep(goal=g, duration_steps=step.duration_steps))
    name = f"candidate_move_{uuid4().hex[:8]}"
    return name, noisy, tags


@dataclass
class BabbleGateResult:
    ok: bool
    reason: str = ""


def babble_gate(
    *,
    config: SkillBabbleConfig,
    step: int,
    last_babble_step: int,
    executor_active: bool,
    novelty: float,
    survival_risk: float,
    hunger: float,
    explore_tendency: float = 0.0,
    library_score_weak: bool = True,
) -> BabbleGateResult:
    """Whether motor babbling may start a new candidate skill."""
    if not config.enabled:
        return BabbleGateResult(False, "disabled")
    if executor_active:
        return BabbleGateResult(False, "executor_active")
    if int(step) - int(last_babble_step) < int(config.cooldown_steps):
        return BabbleGateResult(False, "cooldown")
    if float(survival_risk) > float(config.max_survival_risk):
        return BabbleGateResult(False, "survival")
    if float(hunger) > float(config.max_hunger):
        return BabbleGateResult(False, "hunger")
    novelty_ok = float(novelty) >= float(config.min_novelty)
    explore_ok = float(explore_tendency) >= 0.45
    if not (novelty_ok or explore_ok or library_score_weak):
        return BabbleGateResult(False, "low_drive")
    return BabbleGateResult(True, "ok")


def score_babble_episode(
    *,
    motion_variance: float,
    damage_spike: bool,
    novelty_during: float,
    completed: bool,
) -> float:
    """Lightweight episode score in [0, 1]."""
    if damage_spike:
        return 0.05
    base = 0.25
    if completed:
        base += 0.2
    base += 0.35 * float(np.clip(motion_variance, 0.0, 1.0))
    base += 0.2 * float(np.clip(novelty_during, 0.0, 1.0))
    return float(np.clip(base, 0.0, 1.0))


def build_candidate_skill(
    *,
    fly_id: str,
    steps: list[SkillStep],
    name: str,
    tags: list[str],
    cue: str = "",
    place_id: str | None = None,
    step: int = 0,
    skill_id: str | None = None,
) -> SkillProgram:
    return SkillProgram(
        skill_id=str(skill_id or uuid4()),
        name=str(name),
        status="candidate",
        steps=list(steps),
        tags=list(tags),
        source="babble",
        fly_id=str(fly_id),
        score=0.0,
        context=SkillContext(cue=str(cue or ""), place_id=place_id, step=int(step)),
    )


def signals_for_gate(
    signals: dict[str, Any] | None, affect: dict[str, Any] | None
) -> dict[str, float]:
    snap = signals if isinstance(signals, dict) else {}
    aff = affect if isinstance(affect, dict) else {}
    return {
        "novelty": float(
            snap.get("visual_novelty", snap.get("novelty", snap.get("experience_novelty", 0.0)))
            or 0.0
        ),
        "survival_risk": float(aff.get("survival_risk", snap.get("survival_risk", 0.0)) or 0.0),
        "hunger": float(snap.get("hunger", 0.0) or 0.0),
        "explore_tendency": float(
            snap.get("tendency_curiosity", snap.get("curiosity", 0.0)) or 0.0
        ),
    }
