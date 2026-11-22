"""Father promote + skill playback from place lesson action / skill_name."""

from __future__ import annotations

import numpy as np

from flybrain.env import FakeFlyConfig, FakeFlyEnv
from flybrain.memory.place_graph import PlaceGraphConfig
from flybrain.memory.skill_cortex import (
    InMemorySkillCortex,
    SkillCortexConfig,
    SkillProgram,
    SkillStep,
)
from flybrain.pipeline.live_mind import LiveMindConfig, LiveMindPipeline
from flybrain.thinking.father_advocate import (
    FatherAdvocateConfig,
    FatherLesson,
    FatherQuestion,
    FatherReply,
    lesson_from_reply,
    parse_skill_promote_from_text,
)
from flybrain.thinking.stimulus_interrupt import StimulusInterruptConfig


def test_parse_skill_promote_from_text() -> None:
    name, promote = parse_skill_promote_from_text("Keep this move as dance_spin")
    assert name == "dance_spin"
    assert promote is True
    name2, promote2 = parse_skill_promote_from_text("promote last candidate")
    assert promote2 is True
    name3, _ = parse_skill_promote_from_text("Try a touch_self grooming bout.")
    assert name3 == "touch_self"


def test_lesson_from_reply_carries_skill_name() -> None:
    q = FatherQuestion(
        request_id=1,
        step=3,
        cue="novelty",
        reason="novel:novelty",
        question="What is this?",
    )
    reply = FatherReply(
        request_id=1,
        source="human",
        answer="Keep this move as touch_self and stay still after.",
        lesson="Keep this move as touch_self",
        risk="low",
        action="idle",
        turn_bias=0.0,
        confidence=0.8,
    )
    lesson = lesson_from_reply(q, reply, place_id="p1")
    assert lesson.skill_name == "touch_self"
    assert lesson.promote_last_candidate is True


def test_father_promote_and_playback() -> None:
    cfg = LiveMindConfig(
        backend="in_memory",
        llm_mock=True,
        console_log=False,
        async_mind=False,
        mind_interval=4,
        enable_motor_layer=True,
        fly_id="fly_skill",
        stimulus_interrupt=StimulusInterruptConfig(enabled=False),
        father_advocate=FatherAdvocateConfig(enabled=False),
        place_graph=PlaceGraphConfig(enabled=False, backend="in_memory"),
        skill_cortex=SkillCortexConfig(
            enabled=True,
            backend="in_memory",
            promote_min_score=0.3,
            babble={"enabled": False},
        ),
    )
    pipe = LiveMindPipeline(FakeFlyEnv(FakeFlyConfig(seed=3)), config=cfg)
    try:
        assert isinstance(pipe.skill_cortex, InMemorySkillCortex)
        cand = SkillProgram(
            skill_id="cand-1",
            name="candidate_dance_x",
            status="candidate",
            steps=[
                SkillStep(
                    goal=np.asarray([0.2, 0.7, 0.5, 0.8, 0.0], dtype=np.float32),
                    duration_steps=3,
                ),
                SkillStep(
                    goal=np.asarray([0.2, -0.7, 0.5, 0.8, 0.0], dtype=np.float32),
                    duration_steps=3,
                ),
            ],
            tags=["dance"],
            source="babble",
            fly_id="fly_skill",
            score=0.6,
        )
        pipe.skill_cortex.upsert_skill(cand)
        pipe._last_babble_skill_id = "cand-1"
        pipe._pending_skill_promote_id = "cand-1"

        lesson = FatherLesson(
            cue="novelty",
            question="Keep?",
            answer="Keep this move as dance_spin",
            lesson="Keep this move as dance_spin",
            risk="low",
            source="human",
            step=1,
            place_id=None,
            action="idle",
            turn_bias=0.0,
            confidence=0.9,
            skill_name="dance_spin",
            promote_last_candidate=True,
        )
        pipe._handle_father_skill_promote(lesson)
        got = pipe.skill_cortex.get_by_name(
            "dance_spin", fly_id="fly_skill", status="promoted"
        )
        assert got is not None
        assert got.status == "promoted"
        names = [n for n, _ in pipe.thinking.extra_action_candidates]
        assert "dance_spin" in names

        assert pipe._maybe_start_skill_from_lesson(
            FatherLesson(
                cue="novelty",
                question="",
                answer="",
                lesson="",
                risk="low",
                source="place",
                step=2,
                action="idle",
                confidence=0.9,
                skill_name="dance_spin",
            )
        )
        assert pipe.skill_executor.active
        assert pipe.skill_executor.program is not None
        assert pipe.skill_executor.program.name == "dance_spin"
    finally:
        pipe.close()
