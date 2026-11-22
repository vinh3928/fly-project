"""SkillExecutor step advance, goals, and preempt."""

from __future__ import annotations

import numpy as np

from flybrain.memory.skill_cortex import SkillProgram, SkillStep
from flybrain.motor.skill_executor import SkillExecutor


def _prog() -> SkillProgram:
    return SkillProgram(
        skill_id="ex-1",
        name="dance_spin",
        status="promoted",
        steps=[
            SkillStep(
                goal=np.asarray([0.3, 0.9, 0.5, 0.8, 0.0], dtype=np.float32), duration_steps=2
            ),
            SkillStep(
                goal=np.asarray([0.3, -0.9, 0.5, 0.8, 0.0], dtype=np.float32), duration_steps=2
            ),
        ],
        tags=["dance"],
        source="babble",
        fly_id="fly_x",
    )


def test_executor_advances_and_finishes() -> None:
    ex = SkillExecutor()
    ex.start(_prog())
    assert ex.active
    f0 = ex.tick()
    assert f0.active
    assert f0.plan_name == "skill_dance_spin"
    assert float(f0.goal[1]) > 0.5
    f1 = ex.tick()
    assert f1.active
    assert f1.step_index == 1 or f1.step_index == 0
    # Exhaust remaining duration.
    frames = [ex.tick() for _ in range(6)]
    assert any(f.finished for f in frames) or ex.finished
    assert ex.finished


def test_executor_preempt() -> None:
    ex = SkillExecutor()
    ex.start(_prog())
    ex.tick()
    ex.preempt("interrupt")
    assert ex.finished
    assert not ex.active
    frame = ex.tick()
    assert frame.finished
    assert frame.reason == "skill_idle"


def test_executor_episode_stats() -> None:
    ex = SkillExecutor()
    ex.start(_prog())
    ex.observe(motion=0.4, novelty=0.6, damage=0.0)
    ex.observe(motion=0.5, novelty=0.4, damage=0.0)
    while not ex.finished:
        ex.tick()
    stats = ex.episode_stats()
    assert stats["skill_id"] == "ex-1"
    assert stats["completed"] is True
    assert float(stats["novelty_during"]) > 0.0
