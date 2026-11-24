"""Display 4 skill-promote queue: channel + Accept/Refuse/Ask LLM adjudicate."""

from __future__ import annotations

import numpy as np

from flybrain.debug.skill_promote_channel import (
    KIND_SHOW,
    PROTOCOL_VERSION,
    SkillPromoteChannel,
    adjudicate_skill_promote,
    heuristic_adjudicate_skill,
    pack_show_message,
    parse_adjudication_text,
    parse_reply_message,
    suggest_skill_name,
    summarize_skill_for_judge,
)
from flybrain.env import FakeFlyConfig, FakeFlyEnv
from flybrain.memory.place_graph import PlaceGraphConfig
from flybrain.memory.skill_cortex import (
    SkillCortexConfig,
    SkillProgram,
    SkillStep,
)
from flybrain.pipeline.live_mind import LiveMindConfig, LiveMindPipeline
from flybrain.thinking.father_advocate import FatherAdvocateConfig
from flybrain.thinking.stimulus_interrupt import StimulusInterruptConfig


def _cand(
    *,
    skill_id: str,
    tags: list[str],
    score: float,
    fly_id: str,
    goals: list[list[float]] | None = None,
) -> SkillProgram:
    seq = goals or [[0.2, 0.6, 0.5, 0.7, 0.0], [0.1, -0.5, 0.5, 0.6, 0.0]]
    return SkillProgram(
        skill_id=skill_id,
        name=f"candidate_move_{skill_id}",
        status="candidate",
        steps=[SkillStep(goal=np.asarray(g, dtype=np.float32), duration_steps=2) for g in seq],
        tags=tags,
        source="babble",
        fly_id=fly_id,
        score=score,
    )


def test_suggest_skill_name_from_primitive_tags() -> None:
    assert suggest_skill_name(tags=["babble", "combo", "turn_left", "extend"]) == "turn_left_extend"
    assert suggest_skill_name(tags=["forward", "pivot"], current_name="candidate_move_ab12") == (
        "forward_pivot"
    )
    assert suggest_skill_name(tags=[], current_name="candidate_move_deadbeef") == "move_deadbeef"


def test_heuristic_adjudicate_accept_and_refuse() -> None:
    strong = summarize_skill_for_judge(
        _cand(
            skill_id="ok",
            tags=["babble", "combo", "turn_right", "forward"],
            score=0.8,
            fly_id="fly_x",
        )
    )
    ok = heuristic_adjudicate_skill(strong, min_score=0.4)
    assert ok.decision == "accept"
    assert ok.name == "turn_right_forward"

    weak = summarize_skill_for_judge(
        _cand(
            skill_id="bad",
            tags=["babble", "combo", "pause"],
            score=0.1,
            fly_id="fly_x",
            goals=[[0.0, 0.0, 0.5, 0.2, 0.0], [0.01, 0.0, 0.5, 0.2, 0.0]],
        )
    )
    no = heuristic_adjudicate_skill(weak, min_score=0.4)
    assert no.decision == "refuse"


def test_parse_adjudication_json() -> None:
    fallback = heuristic_adjudicate_skill(
        {"score": 0.8, "motion": 0.4, "variety": 0.3, "n_steps": 2, "tags": ["pivot"], "name": "x"},
        min_score=0.4,
    )
    parsed = parse_adjudication_text(
        '{"decision":"refuse","name":"","reason":"too twitchy"}',
        fallback=fallback,
    )
    assert parsed.decision == "refuse"
    assert parsed.source == "llm"
    assert "twitchy" in parsed.reason

    keep = parse_adjudication_text(
        '{"decision":"accept","name":"side_probe","reason":"clear turn bout"}',
        fallback=fallback,
    )
    assert keep.decision == "accept"
    assert keep.name == "side_probe"


def test_adjudicate_mock_uses_heuristic() -> None:
    program = _cand(
        skill_id="m1",
        tags=["babble", "combo", "extend", "pause"],
        score=0.7,
        fly_id="fly_x",
        goals=[[0.0, 0.1, 0.4, 0.3, 0.9], [0.05, -0.1, 0.45, 0.3, 0.2]],
    )
    verdict = adjudicate_skill_promote(program, client=None, min_score=0.4, mock=True)
    assert verdict.decision == "accept"
    assert verdict.source == "heuristic"
    assert verdict.name == "extend_pause"


def test_pack_show_and_parse_reply_roundtrip_shape() -> None:
    raw = pack_show_message(
        request_id=3,
        step=10,
        queue_count=2,
        skill_id="s1",
        name="candidate_move_ab",
        tags="forward,turn_left",
        score="0.70",
        steps_count="4",
        suggested_name="forward_turn_left",
        allow_llm=True,
    )
    assert raw[:4]  # non-empty
    import struct

    version, kind, request_id = struct.unpack_from("<iii", raw, 0)
    assert version == PROTOCOL_VERSION
    assert kind == KIND_SHOW
    assert request_id == 3

    name_b = b"side_probe"
    reply = struct.pack("<iiii", 1, 1, 3, 0) + struct.pack("<i", len(name_b)) + name_b
    parsed = parse_reply_message(reply)
    assert parsed["source"] == "accept"
    assert parsed["name"] == "side_probe"
    assert parsed["request_id"] == 3


def test_display4_accept_promotes_with_name() -> None:
    ch = SkillPromoteChannel()

    class _Env(FakeFlyEnv):
        @property
        def skill_promote_channel(self):
            return ch

    cfg = LiveMindConfig(
        backend="in_memory",
        llm_mock=True,
        console_log=False,
        async_mind=False,
        mind_interval=4,
        enable_motor_layer=True,
        fly_id="fly_promote_ui",
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
    pipe = LiveMindPipeline(_Env(FakeFlyConfig(seed=1)), config=cfg)
    try:
        pipe.skill_cortex.upsert_skill(
            _cand(
                skill_id="ui-cand-1",
                tags=["babble", "combo", "turn_right", "forward"],
                score=0.8,
                fly_id="fly_promote_ui",
            )
        )
        pipe._enqueue_skill_promote("ui-cand-1")
        assert pipe._skill_promote_showing is True
        assert ch.last_show is not None
        assert ch.last_show["suggested_name"] == "turn_right_forward"

        ch.inject_reply_for_tests(
            request_id=int(pipe._skill_promote_request_id),
            source="accept",
            name="wiggle_probe",
        )
        pipe._poll_skill_promote_ui()
        got = pipe.skill_cortex.get_by_name(
            "wiggle_probe", fly_id="fly_promote_ui", status="promoted"
        )
        assert got is not None
        assert got.status == "promoted"
        assert pipe._skill_promote_queue == []
        assert pipe._skill_promote_showing is False
    finally:
        pipe.close()


def test_display4_ask_llm_auto_accepts_strong_candidate() -> None:
    ch = SkillPromoteChannel()

    class _Env(FakeFlyEnv):
        @property
        def skill_promote_channel(self):
            return ch

    cfg = LiveMindConfig(
        backend="in_memory",
        llm_mock=True,
        console_log=False,
        async_mind=False,
        mind_interval=4,
        enable_motor_layer=True,
        fly_id="fly_promote_llm",
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
    pipe = LiveMindPipeline(_Env(FakeFlyConfig(seed=3)), config=cfg)
    try:
        pipe.skill_cortex.upsert_skill(
            _cand(
                skill_id="llm-ok",
                tags=["babble", "combo", "pivot", "forward"],
                score=0.75,
                fly_id="fly_promote_llm",
            )
        )
        pipe._enqueue_skill_promote("llm-ok")
        req = int(pipe._skill_promote_request_id)
        ch.inject_reply_for_tests(request_id=req, source="ask_llm", name="")
        pipe._poll_skill_promote_ui()
        got = pipe.skill_cortex.get_skill("llm-ok")
        assert got is not None and got.status == "promoted"
        assert got.name == "pivot_forward"
        assert pipe._skill_promote_queue == []
        assert pipe._skill_promote_showing is False
    finally:
        pipe.close()


def test_display4_ask_llm_auto_refuses_weak_candidate() -> None:
    ch = SkillPromoteChannel()

    class _Env(FakeFlyEnv):
        @property
        def skill_promote_channel(self):
            return ch

    cfg = LiveMindConfig(
        backend="in_memory",
        llm_mock=True,
        console_log=False,
        async_mind=False,
        mind_interval=4,
        enable_motor_layer=True,
        fly_id="fly_promote_llm2",
        stimulus_interrupt=StimulusInterruptConfig(enabled=False),
        father_advocate=FatherAdvocateConfig(enabled=False),
        place_graph=PlaceGraphConfig(enabled=False, backend="in_memory"),
        skill_cortex=SkillCortexConfig(
            enabled=True,
            backend="in_memory",
            promote_min_score=0.4,
            babble={"enabled": False},
        ),
    )
    pipe = LiveMindPipeline(_Env(FakeFlyConfig(seed=4)), config=cfg)
    try:
        pipe.skill_cortex.upsert_skill(
            _cand(
                skill_id="llm-bad",
                tags=["babble", "combo", "pause"],
                score=0.1,
                fly_id="fly_promote_llm2",
                goals=[[0.0, 0.0, 0.5, 0.2, 0.0], [0.0, 0.02, 0.5, 0.2, 0.0]],
            )
        )
        pipe.skill_cortex.upsert_skill(
            _cand(
                skill_id="llm-next",
                tags=["babble", "combo", "turn_left"],
                score=0.7,
                fly_id="fly_promote_llm2",
            )
        )
        pipe._enqueue_skill_promote("llm-bad")
        pipe._enqueue_skill_promote("llm-next")
        req = int(pipe._skill_promote_request_id)
        ch.inject_reply_for_tests(request_id=req, source="ask_llm", name="")
        pipe._poll_skill_promote_ui()
        bad = pipe.skill_cortex.get_skill("llm-bad")
        assert bad is not None and bad.status == "retired"
        assert pipe._skill_promote_queue == ["llm-next"]
        assert pipe._skill_promote_showing is True
    finally:
        pipe.close()


def test_display4_auto_ask_llm_when_no_view_signal() -> None:
    """No Unity VIEW → Python immediate auto-adjudicate (grace 0)."""
    ch = SkillPromoteChannel()

    class _Env(FakeFlyEnv):
        @property
        def skill_promote_channel(self):
            return ch

    cfg = LiveMindConfig(
        backend="in_memory",
        llm_mock=True,
        console_log=False,
        async_mind=False,
        mind_interval=4,
        enable_motor_layer=True,
        fly_id="fly_promote_auto",
        stimulus_interrupt=StimulusInterruptConfig(enabled=False),
        father_advocate=FatherAdvocateConfig(enabled=False),
        place_graph=PlaceGraphConfig(enabled=False, backend="in_memory"),
        skill_cortex=SkillCortexConfig(
            enabled=True,
            backend="in_memory",
            promote_min_score=0.3,
            auto_llm_away_seconds=0.0,
            babble={"enabled": False},
        ),
    )
    pipe = LiveMindPipeline(_Env(FakeFlyConfig(seed=5)), config=cfg)
    try:
        pipe.skill_cortex.upsert_skill(
            _cand(
                skill_id="auto-ok",
                tags=["babble", "combo", "forward", "pivot"],
                score=0.8,
                fly_id="fly_promote_auto",
            )
        )
        pipe._enqueue_skill_promote("auto-ok")
        # Do not inject VIEW — Python owns fallback when focus is unknown.
        pipe._poll_skill_promote_ui()
        got = pipe.skill_cortex.get_skill("auto-ok")
        assert got is not None and got.status == "promoted"
        assert pipe._skill_promote_queue == []
    finally:
        pipe.close()


def test_display4_python_defers_when_unity_reports_away() -> None:
    """Unity reported away → wait for Unity ask_llm; do not double-fire."""
    ch = SkillPromoteChannel()

    class _Env(FakeFlyEnv):
        @property
        def skill_promote_channel(self):
            return ch

    cfg = LiveMindConfig(
        backend="in_memory",
        llm_mock=True,
        console_log=False,
        async_mind=False,
        mind_interval=4,
        enable_motor_layer=True,
        fly_id="fly_promote_defer",
        stimulus_interrupt=StimulusInterruptConfig(enabled=False),
        father_advocate=FatherAdvocateConfig(enabled=False),
        place_graph=PlaceGraphConfig(enabled=False, backend="in_memory"),
        skill_cortex=SkillCortexConfig(
            enabled=True,
            backend="in_memory",
            promote_min_score=0.3,
            auto_llm_away_seconds=0.0,
            babble={"enabled": False},
        ),
    )
    pipe = LiveMindPipeline(_Env(FakeFlyConfig(seed=8)), config=cfg)
    try:
        pipe.skill_cortex.upsert_skill(
            _cand(
                skill_id="wait-unity",
                tags=["babble", "combo", "forward", "pivot"],
                score=0.8,
                fly_id="fly_promote_defer",
            )
        )
        pipe._enqueue_skill_promote("wait-unity")
        req = int(pipe._skill_promote_request_id)
        ch.inject_view_for_tests(request_id=req, viewing=False)
        pipe._poll_skill_promote_ui()
        got = pipe.skill_cortex.get_skill("wait-unity")
        assert got is not None and got.status == "candidate"
        assert pipe._skill_promote_queue == ["wait-unity"]
        # Unity-style ask_llm then resolves.
        ch.inject_reply_for_tests(request_id=req, source="ask_llm", name="")
        pipe._poll_skill_promote_ui()
        got = pipe.skill_cortex.get_skill("wait-unity")
        assert got is not None and got.status == "promoted"
    finally:
        pipe.close()


def test_display4_auto_ask_llm_disabled_when_grace_negative() -> None:
    ch = SkillPromoteChannel()

    class _Env(FakeFlyEnv):
        @property
        def skill_promote_channel(self):
            return ch

    cfg = LiveMindConfig(
        backend="in_memory",
        llm_mock=True,
        console_log=False,
        async_mind=False,
        mind_interval=4,
        enable_motor_layer=True,
        fly_id="fly_promote_off",
        stimulus_interrupt=StimulusInterruptConfig(enabled=False),
        father_advocate=FatherAdvocateConfig(enabled=False),
        place_graph=PlaceGraphConfig(enabled=False, backend="in_memory"),
        skill_cortex=SkillCortexConfig(
            enabled=True,
            backend="in_memory",
            promote_min_score=0.3,
            auto_llm_away_seconds=-1.0,
            babble={"enabled": False},
        ),
    )
    pipe = LiveMindPipeline(_Env(FakeFlyConfig(seed=7)), config=cfg)
    try:
        pipe.skill_cortex.upsert_skill(
            _cand(
                skill_id="stay-cand",
                tags=["babble", "combo", "forward"],
                score=0.8,
                fly_id="fly_promote_off",
            )
        )
        pipe._enqueue_skill_promote("stay-cand")
        pipe._poll_skill_promote_ui()
        got = pipe.skill_cortex.get_skill("stay-cand")
        assert got is not None and got.status == "candidate"
        assert pipe._skill_promote_queue == ["stay-cand"]
    finally:
        pipe.close()


def test_display4_no_auto_ask_llm_while_viewing() -> None:
    ch = SkillPromoteChannel()

    class _Env(FakeFlyEnv):
        @property
        def skill_promote_channel(self):
            return ch

    cfg = LiveMindConfig(
        backend="in_memory",
        llm_mock=True,
        console_log=False,
        async_mind=False,
        mind_interval=4,
        enable_motor_layer=True,
        fly_id="fly_promote_view",
        stimulus_interrupt=StimulusInterruptConfig(enabled=False),
        father_advocate=FatherAdvocateConfig(enabled=False),
        place_graph=PlaceGraphConfig(enabled=False, backend="in_memory"),
        skill_cortex=SkillCortexConfig(
            enabled=True,
            backend="in_memory",
            promote_min_score=0.3,
            auto_llm_away_seconds=0.01,
            babble={"enabled": False},
        ),
    )
    pipe = LiveMindPipeline(_Env(FakeFlyConfig(seed=6)), config=cfg)
    try:
        pipe.skill_cortex.upsert_skill(
            _cand(
                skill_id="hold-me",
                tags=["babble", "combo", "forward"],
                score=0.8,
                fly_id="fly_promote_view",
            )
        )
        pipe._enqueue_skill_promote("hold-me")
        ch.inject_view_for_tests(request_id=int(pipe._skill_promote_request_id), viewing=True)
        pipe._skill_promote_shown_at = __import__("time").monotonic() - 10.0
        pipe._poll_skill_promote_ui()
        got = pipe.skill_cortex.get_skill("hold-me")
        assert got is not None and got.status == "candidate"
        assert pipe._skill_promote_queue == ["hold-me"]
        assert pipe._skill_promote_showing is True
    finally:
        pipe.close()


def test_display4_refuse_retires_and_test_keeps_head() -> None:
    ch = SkillPromoteChannel()

    class _Env(FakeFlyEnv):
        @property
        def skill_promote_channel(self):
            return ch

    cfg = LiveMindConfig(
        backend="in_memory",
        llm_mock=True,
        console_log=False,
        async_mind=False,
        mind_interval=4,
        enable_motor_layer=True,
        fly_id="fly_promote_ui2",
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
    pipe = LiveMindPipeline(_Env(FakeFlyConfig(seed=2)), config=cfg)
    try:
        for sid, tags in (
            ("a1", ["babble", "combo", "extend", "pause"]),
            ("a2", ["babble", "combo", "pivot", "forward"]),
        ):
            pipe.skill_cortex.upsert_skill(
                _cand(skill_id=sid, tags=tags, score=0.6, fly_id="fly_promote_ui2")
            )
            pipe._enqueue_skill_promote(sid)

        assert len(pipe._skill_promote_queue) == 2
        req_test = int(pipe._skill_promote_request_id)
        ch.inject_reply_for_tests(request_id=req_test, source="test", name="")
        pipe._poll_skill_promote_ui()
        assert pipe._skill_promote_queue[0] == "a1"
        assert pipe.skill_executor.active or pipe.skill_executor.program is not None

        req2 = int(pipe._skill_promote_request_id)
        ch.inject_reply_for_tests(request_id=req2, source="refuse", name="")
        pipe._poll_skill_promote_ui()
        retired = pipe.skill_cortex.get_skill("a1")
        assert retired is not None and retired.status == "retired"
        assert pipe._skill_promote_queue == ["a2"]
        assert pipe._skill_promote_showing is True
    finally:
        pipe.close()
