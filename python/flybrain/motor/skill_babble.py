"""Motor babbling: sample short touch/dance-style programs outside the L1 library."""

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

MOTIF_GROOM = "groom"
MOTIF_DANCE = "dance"
MOTIF_TOUCH = "touch_self"


def _rand_duration(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def sample_groom_program(
    rng: np.random.Generator,
    *,
    min_steps: int,
    max_steps: int,
    duration_range: tuple[int, int],
) -> list[SkillStep]:
    """Stop-ish + proboscis / height wiggle (self-touch / grooming motif)."""
    n = int(rng.integers(min_steps, max_steps + 1))
    lo, hi = duration_range
    steps: list[SkillStep] = []
    for i in range(n):
        turn = float(rng.uniform(-0.35, 0.35))
        height = float(rng.uniform(0.35, 0.75))
        freq = float(rng.uniform(0.15, 0.45))
        prob = float(rng.uniform(0.55, 1.0)) if i % 2 == 0 else float(rng.uniform(0.2, 0.6))
        steps.append(
            SkillStep(
                goal=clip_goal(np.asarray([0.05, turn, height, freq, prob], dtype=np.float32)),
                duration_steps=_rand_duration(rng, lo, hi),
            )
        )
    return steps


def sample_dance_program(
    rng: np.random.Generator,
    *,
    min_steps: int,
    max_steps: int,
    duration_range: tuple[int, int],
) -> list[SkillStep]:
    """Alternating turn pulses + frequency peaks (dance motif)."""
    n = int(rng.integers(max(min_steps, 4), max_steps + 1))
    lo, hi = duration_range
    steps: list[SkillStep] = []
    for i in range(n):
        sign = -1.0 if i % 2 == 0 else 1.0
        turn = float(sign * rng.uniform(0.45, 0.95))
        fwd = float(rng.uniform(0.15, 0.45))
        freq = float(rng.uniform(0.55, 0.95))
        height = float(rng.uniform(0.4, 0.7))
        steps.append(
            SkillStep(
                goal=clip_goal(np.asarray([fwd, turn, height, freq, 0.0], dtype=np.float32)),
                duration_steps=_rand_duration(rng, lo, max(lo, hi // 2 + 2)),
            )
        )
    return steps


def sample_babble_program(
    rng: np.random.Generator | None = None,
    *,
    config: SkillBabbleConfig | None = None,
    motif: str | None = None,
) -> tuple[str, list[SkillStep], list[str]]:
    """Return ``(temp_name, steps, tags)`` biased toward groom/dance motifs + noise."""
    cfg = config or SkillBabbleConfig()
    gen = rng or np.random.default_rng()
    choice = str(motif or "").strip().lower()
    if choice not in (MOTIF_GROOM, MOTIF_DANCE, MOTIF_TOUCH):
        choice = str(gen.choice([MOTIF_GROOM, MOTIF_DANCE, MOTIF_TOUCH]))
    lo, hi = cfg.duration_range
    if choice in (MOTIF_GROOM, MOTIF_TOUCH):
        steps = sample_groom_program(
            gen,
            min_steps=cfg.min_steps,
            max_steps=cfg.max_steps,
            duration_range=(lo, hi),
        )
        tags = [MOTIF_GROOM, MOTIF_TOUCH]
        name = f"candidate_{MOTIF_TOUCH}_{uuid4().hex[:6]}"
    else:
        steps = sample_dance_program(
            gen,
            min_steps=cfg.min_steps,
            max_steps=cfg.max_steps,
            duration_range=(lo, hi),
        )
        tags = [MOTIF_DANCE]
        name = f"candidate_{MOTIF_DANCE}_{uuid4().hex[:6]}"
    # Light noise on each goal so candidates are not identical.
    noisy: list[SkillStep] = []
    for step in steps:
        g = np.asarray(step.goal, dtype=np.float32).copy()
        g[0] = float(np.clip(g[0] + gen.normal(0.0, 0.05), -1.0, 1.0))
        g[1] = float(np.clip(g[1] + gen.normal(0.0, 0.08), -1.0, 1.0))
        g[2] = float(np.clip(g[2] + gen.normal(0.0, 0.04), 0.0, 1.0))
        g[3] = float(np.clip(g[3] + gen.normal(0.0, 0.04), 0.0, 1.0))
        g[4] = float(np.clip(g[4] + gen.normal(0.0, 0.05), 0.0, 1.0))
        noisy.append(SkillStep(goal=g, duration_steps=step.duration_steps))
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
