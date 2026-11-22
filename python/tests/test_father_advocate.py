"""Father advocate question / reply / lesson store."""

from __future__ import annotations

import numpy as np

from flybrain.debug.father_dialog_channel import (
    FatherDialogChannel,
    pack_ask_message,
    pack_hide_message,
    parse_reply_message,
)
from flybrain.thinking.father_advocate import (
    FatherAdvocateConfig,
    FatherLesson,
    FatherLessonStore,
    FatherQuestion,
    FatherReply,
    answer_father_question,
    directive_fields_from_father_lesson,
    goal_from_father_structure,
    heuristic_father_question,
    infer_father_structure,
    lesson_from_reply,
    make_father_client,
    propose_father_question,
)


def test_infer_structure_from_human_text() -> None:
    action, turn, conf, risk = infer_father_structure(
        "Back away slowly; this is toxin.",
        risk="high",
        cue="bitter",
    )
    assert action == "avoid"
    assert turn != 0.0
    assert conf >= 0.7
    assert risk == "high"


def test_goal_from_father_structure_actions() -> None:
    avoid = goal_from_father_structure(action="avoid", turn_bias=0.5, confidence=0.8, risk="high")
    assert float(avoid[0]) < 0.0
    assert abs(float(avoid[1])) > 0.1
    taste = goal_from_father_structure(action="taste", confidence=0.8, risk="low")
    assert float(taste[4]) >= 0.5
    approach = goal_from_father_structure(action="approach", confidence=0.7, risk="low")
    assert float(approach[0]) > 0.4


def test_human_reply_infers_structure_into_lesson() -> None:
    q = FatherQuestion(
        request_id=1,
        step=3,
        cue="smell_novelty",
        reason="novel",
        question="Is this food?",
        signals_snapshot={"sugar": 0.4},
    )
    reply = FatherReply(
        request_id=1,
        source="human",
        answer="Sugar plume: approach slowly and taste before committing.",
        lesson="Sugar plume: approach slowly and taste before committing.",
        risk="medium",
    )
    lesson = lesson_from_reply(q, reply, place_id=9)
    assert lesson.action in ("taste", "approach")
    assert lesson.confidence >= 0.5
    fields = directive_fields_from_father_lesson(lesson)
    assert fields["plan_name"].startswith("father_")
    assert float(fields["goal"][0]) > 0.0 or float(fields["goal"][4]) > 0.0


def test_heuristic_question_mentions_cue() -> None:
    q = heuristic_father_question(
        cue="smell_novelty",
        reason="novel",
        signals={"sugar": 0.5, "bitter": 0.0},
    )
    assert "food" in q.lower() or "sugar" in q.lower()


def test_propose_and_answer_mock() -> None:
    client = make_father_client(mock=True)
    cfg = FatherAdvocateConfig(mock=True)
    question = propose_father_question(
        client,
        request_id=3,
        step=10,
        cue="bitter",
        reason="novel",
        signals={"bitter": 0.6},
        config=cfg,
    )
    assert question.request_id == 3
    assert question.question
    reply = answer_father_question(client, question, config=cfg, source="llm")
    assert reply.answer
    assert reply.risk in ("low", "medium", "high")
    lesson = lesson_from_reply(question, reply)
    store = FatherLessonStore(max_lessons=4)
    store.add(lesson)
    assert store.latest_for_cue("bitter") is not None
    assert store.latest_for_cue("other") is None


def test_father_dialog_pack_parse_roundtrip() -> None:
    raw = pack_ask_message(
        request_id=7,
        step=42,
        cue="visual_novelty",
        question="What is this?",
        allow_llm=True,
    )
    assert len(raw) > 20
    hide = pack_hide_message(request_id=7)
    assert len(hide) == 12

    # Simulate Unity reply bytes: version, kind, request_id, source, string
    import struct

    answer = b"Back away slowly"
    reply_raw = struct.pack("<iiii", 1, 1, 7, 0) + struct.pack("<i", len(answer)) + answer
    parsed = parse_reply_message(reply_raw)
    assert parsed["request_id"] == 7
    assert parsed["source"] == "human"
    assert "Back away" in parsed["answer"]


def test_father_dialog_channel_inject() -> None:
    ch = FatherDialogChannel()
    ch.send_ask(
        request_id=1,
        step=0,
        cue="wall",
        question="Is this a wall?",
        allow_llm=True,
    )
    assert ch.last_ask is not None
    ch.inject_reply_for_tests(request_id=1, answer="Yes, turn left.", source="human")
    got = ch.pop_reply(1)
    assert got is not None
    assert got["answer"].startswith("Yes")


def test_live_mind_father_auto_llm_on_fake(monkeypatch) -> None:
    """Fake env has no Unity channel → Father auto-answers via LLM/heuristic."""
    from flybrain.env import FakeFlyConfig, FakeFlyEnv
    from flybrain.pipeline.live_mind import LiveMindConfig, LiveMindPipeline

    cfg = LiveMindConfig(
        backend="in_memory",
        llm_mock=True,
        console_log=False,
        async_mind=False,
        mind_interval=4,
        father_advocate=FatherAdvocateConfig(enabled=True, mode="llm", mock=True),
    )
    # Force interrupt-friendly low load / novelty by stubbing evaluate if needed —
    # instead call _begin_father_advocate directly after a fake arm.
    pipe = LiveMindPipeline(FakeFlyEnv(FakeFlyConfig(seed=1)), config=cfg)
    pipe._interrupt_reason = "novel_smell"
    pipe._interrupt_meta = {"cue": "smell_novelty"}
    pipe._begin_father_advocate({"odor": 0.4, "sugar": 0.0, "bitter": 0.0})
    assert pipe._father_waiting is False
    assert pipe._father_lesson is not None
    assert pipe._father_lesson.answer
    assert pipe.father_lessons.latest_for_cue("smell_novelty") is not None
    pipe.close()


def test_live_mind_father_human_reply_via_channel() -> None:
    from flybrain.debug.father_dialog_channel import FatherDialogChannel
    from flybrain.env import FakeFlyConfig, FakeFlyEnv
    from flybrain.pipeline.live_mind import LiveMindConfig, LiveMindPipeline

    ch = FatherDialogChannel()
    # Pretend Unity is connected so mode human_or_llm waits.
    ch._impl = object()  # type: ignore[attr-defined]
    ch._OutgoingMessage = None

    class _Env(FakeFlyEnv):
        @property
        def father_dialog_channel(self):  # noqa: D401
            return ch

    cfg = LiveMindConfig(
        backend="in_memory",
        llm_mock=True,
        console_log=False,
        async_mind=False,
        father_advocate=FatherAdvocateConfig(
            enabled=True, mode="human_or_llm", mock=True, timeout_seconds=60.0
        ),
    )
    pipe = LiveMindPipeline(_Env(FakeFlyConfig(seed=2)), config=cfg)
    pipe._interrupt_reason = "novel"
    pipe._interrupt_meta = {"cue": "visual_novelty"}
    pipe._begin_father_advocate({"visual_novelty": 0.5})
    assert pipe._father_waiting is True
    assert pipe._father_wait_started is not None
    ch.inject_reply_for_tests(
        request_id=pipe._father_request_id,
        answer="Explore slowly; it may be an opening.",
        source="human",
    )
    assert pipe._poll_father_advocate() is False
    assert pipe._father_lesson is not None
    assert "opening" in pipe._father_lesson.answer.lower()
    pipe.close()


def test_father_timeout_auto_llm(monkeypatch) -> None:
    """After timeout_seconds with no human reply, Father auto-answers via LLM."""
    import time as time_mod

    from flybrain.debug.father_dialog_channel import FatherDialogChannel
    from flybrain.env import FakeFlyConfig, FakeFlyEnv
    from flybrain.pipeline.live_mind import LiveMindConfig, LiveMindPipeline

    ch = FatherDialogChannel()
    ch._impl = object()  # type: ignore[attr-defined]
    ch._OutgoingMessage = None

    class _Env(FakeFlyEnv):
        @property
        def father_dialog_channel(self):
            return ch

    cfg = LiveMindConfig(
        backend="in_memory",
        llm_mock=True,
        console_log=False,
        async_mind=False,
        father_advocate=FatherAdvocateConfig(
            enabled=True, mode="human_or_llm", mock=True, timeout_seconds=60.0
        ),
    )
    pipe = LiveMindPipeline(_Env(FakeFlyConfig(seed=3)), config=cfg)
    pipe._interrupt_reason = "novel"
    pipe._interrupt_meta = {"cue": "smell_novelty"}
    pipe._begin_father_advocate({"odor": 0.5})
    assert pipe._father_waiting is True
    # Pretend 61s elapsed without a Unity reply.
    pipe._father_wait_started = time_mod.monotonic() - 61.0
    pipe._father_polled_step = -1
    assert pipe._poll_father_advocate() is False
    assert pipe._father_waiting is False
    assert pipe._father_lesson is not None
    assert pipe._father_lesson.source == "timeout_llm"
    pipe.close()


def test_father_lesson_persists_on_place_and_reuses() -> None:
    """Father reply attaches HAS_LESSON; same place+cue reuses without re-ask."""
    from flybrain.env import FakeFlyConfig, FakeFlyEnv
    from flybrain.memory.place_graph import InMemoryPlaceGraph
    from flybrain.pipeline.live_mind import LiveMindConfig, LiveMindPipeline
    from flybrain.phase4 import PATCH_VEC_DIM

    cfg = LiveMindConfig(
        backend="in_memory",
        llm_mock=True,
        console_log=False,
        async_mind=False,
        father_advocate=FatherAdvocateConfig(enabled=True, mode="llm", mock=True),
    )
    pipe = LiveMindPipeline(FakeFlyEnv(FakeFlyConfig(seed=4)), config=cfg)
    graph = InMemoryPlaceGraph()
    place = graph.upsert_place(
        np.zeros(PATCH_VEC_DIM, dtype=np.float32),
        fly_id=cfg.fly_id,
        step=1,
    )
    pipe.place_graph = graph
    pipe._last_place_node = place
    pipe._interrupt_reason = "novel_smell"
    pipe._interrupt_meta = {"cue": "smell_novelty"}
    pipe._begin_father_advocate({"odor": 0.4, "sugar": 0.4})
    assert pipe._father_lesson is not None
    assert pipe._father_lesson.place_id == place.id
    assert pipe._father_lesson.action in ("taste", "approach", "avoid", "idle")
    lessons = graph.list_lessons(place.id, cue="smell_novelty")
    assert len(lessons) == 1
    assert lessons[0].answer
    assert lessons[0].action == pipe._father_lesson.action

    # Clear RAM store so reuse must come from the place graph.
    pipe.father_lessons._items.clear()
    pipe._father_lesson = None
    pipe._interrupt_reason = "novel_smell"
    pipe._interrupt_meta = {"cue": "smell_novelty"}
    pipe._begin_father_advocate({"odor": 0.4})
    assert pipe._father_waiting is False
    assert pipe._father_lesson is not None
    assert pipe._father_lesson.place_id == place.id
    assert pipe._father_lesson.action == lessons[0].action
    pipe.close()


def test_father_structure_binds_motor_on_interrupt_clear() -> None:
    """On SI clear, structured Father avoid lesson pushes a reverse/turn goal."""
    from flybrain.env import FakeFlyConfig, FakeFlyEnv
    from flybrain.motor import MotorLayerConfig
    from flybrain.pipeline.live_mind import LiveMindConfig, LiveMindPipeline

    cfg = LiveMindConfig(
        backend="in_memory",
        llm_mock=True,
        console_log=False,
        async_mind=False,
        enable_motor_layer=True,
        motor=MotorLayerConfig(act_state_enabled=True, mode="heuristic"),
        father_advocate=FatherAdvocateConfig(enabled=True, mode="llm", mock=True),
    )
    pipe = LiveMindPipeline(FakeFlyEnv(FakeFlyConfig(seed=5)), config=cfg)
    assert pipe.motor is not None
    assert pipe.act_controller is not None
    pipe._father_lesson = FatherLesson(
        cue="bitter",
        question="Toxin?",
        answer="Back away; high bitter.",
        lesson="Back away; high bitter.",
        risk="high",
        source="human",
        step=1,
        action="avoid",
        turn_bias=0.6,
        confidence=0.9,
    )
    pipe._resume_goal = None
    pipe._interrupt_pending = True
    pipe._clear_stimulus_interrupt()
    assert pipe.act_controller.current.plan_name == "father_avoid"
    assert float(pipe.motor.goal[0]) < 0.0
    assert abs(float(pipe.motor.goal[1])) > 0.1
    pipe.close()
