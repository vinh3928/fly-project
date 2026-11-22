"""Motor babble sampler + gate + episode score."""

from __future__ import annotations

import numpy as np

from flybrain.memory.skill_cortex import InMemorySkillCortex, SkillBabbleConfig
from flybrain.motor.skill_babble import (
    MOTIF_DANCE,
    MOTIF_TOUCH,
    babble_gate,
    build_candidate_skill,
    sample_babble_program,
    score_babble_episode,
)


def test_sample_babble_bounds_and_motifs() -> None:
    rng = np.random.default_rng(0)
    cfg = SkillBabbleConfig(min_steps=3, max_steps=6, duration_range=(8, 16))
    name, steps, tags = sample_babble_program(rng, config=cfg, motif=MOTIF_TOUCH)
    assert "touch" in name or MOTIF_TOUCH in tags
    assert 3 <= len(steps) <= 6
    for step in steps:
        g = np.asarray(step.goal, dtype=np.float32)
        assert g.shape == (5,)
        assert -1.0 <= float(g[0]) <= 1.0
        assert -1.0 <= float(g[1]) <= 1.0
        assert 0.0 <= float(g[2]) <= 1.0
        assert 0.0 <= float(g[4]) <= 1.0
        assert 8 <= int(step.duration_steps) <= 16

    name2, steps2, tags2 = sample_babble_program(rng, config=cfg, motif=MOTIF_DANCE)
    assert MOTIF_DANCE in tags2
    assert len(steps2) >= 4


def test_babble_gate_conditions() -> None:
    cfg = SkillBabbleConfig(
        enabled=True,
        min_novelty=0.35,
        max_survival_risk=0.45,
        max_hunger=0.55,
        cooldown_steps=200,
    )
    assert babble_gate(
        config=cfg,
        step=500,
        last_babble_step=0,
        executor_active=False,
        novelty=0.5,
        survival_risk=0.1,
        hunger=0.1,
    ).ok
    assert not babble_gate(
        config=cfg,
        step=50,
        last_babble_step=0,
        executor_active=False,
        novelty=0.5,
        survival_risk=0.1,
        hunger=0.1,
    ).ok
    assert babble_gate(
        config=cfg,
        step=50,
        last_babble_step=0,
        executor_active=False,
        novelty=0.5,
        survival_risk=0.1,
        hunger=0.1,
    ).reason == "cooldown"
    assert not babble_gate(
        config=cfg,
        step=500,
        last_babble_step=0,
        executor_active=True,
        novelty=0.5,
        survival_risk=0.1,
        hunger=0.1,
    ).ok
    assert not babble_gate(
        config=cfg,
        step=500,
        last_babble_step=0,
        executor_active=False,
        novelty=0.5,
        survival_risk=0.9,
        hunger=0.1,
    ).ok


def test_score_and_candidate_write() -> None:
    score = score_babble_episode(
        motion_variance=0.4,
        damage_spike=False,
        novelty_during=0.5,
        completed=True,
    )
    assert 0.4 <= score <= 1.0
    assert (
        score_babble_episode(
            motion_variance=0.9,
            damage_spike=True,
            novelty_during=1.0,
            completed=True,
        )
        < 0.1
    )
    rng = np.random.default_rng(1)
    name, steps, tags = sample_babble_program(rng, motif=MOTIF_DANCE)
    program = build_candidate_skill(
        fly_id="fly_b",
        steps=steps,
        name=name,
        tags=tags,
        cue="novelty",
        place_id="p1",
        step=12,
    )
    store = InMemorySkillCortex()
    stored = store.upsert_skill(program)
    assert stored.status == "candidate"
    assert stored.source == "babble"
    assert store.get_skill(stored.skill_id) is not None
