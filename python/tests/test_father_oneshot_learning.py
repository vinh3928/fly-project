"""One-shot Father-advocate teaching loop.

Covers the reinforce path:
  novel interrupt → hard-stop motor → Father ask → human/LLM reply → lesson stored
  → same cue reuses the lesson (no second popup wait).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from flybrain.debug.father_dialog_channel import FatherDialogChannel
from flybrain.env import FakeFlyConfig, FakeFlyEnv
from flybrain.pipeline.live_mind import LiveMindConfig, LiveMindPipeline
from flybrain.thinking.father_advocate import FatherAdvocateConfig
from flybrain.thinking.stimulus_interrupt import STOP_GOAL, StimulusInterruptConfig


def _pipeline_with_unity_dialog(
    *,
    seed: int = 7,
    timeout_seconds: float = 60.0,
    # OnesHot human-wait tests: disable away-auto so missing VIEW does not
    # instant-Ask-LLM (production yaml uses 0.0 = Ask LLM when not on Display 3).
    auto_llm_away_seconds: float = -1.0,
) -> tuple[LiveMindPipeline, FatherDialogChannel]:
    ch = FatherDialogChannel()
    # Pretend ML-Agents SideChannel is connected so human_or_llm waits on Unity.
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
        mind_interval=4,
        enable_motor_layer=True,
        fly_id="fly_father_oneshot",
        stimulus_interrupt=StimulusInterruptConfig(
            enabled=True,
            cooldown_steps=0,
            deep_thought_steps=20,
            deep_thought_mind_interval=4,
        ),
        father_advocate=FatherAdvocateConfig(
            enabled=True,
            mode="human_or_llm",
            mock=True,
            timeout_seconds=timeout_seconds,
            auto_llm_away_seconds=auto_llm_away_seconds,
        ),
        place_graph=None,
    )
    # Disable place graph explicitly (None may still load from defaults via from_yaml only).
    from flybrain.memory.place_graph import PlaceGraphConfig

    cfg.place_graph = PlaceGraphConfig(enabled=False, backend="in_memory")
    pipe = LiveMindPipeline(_Env(FakeFlyConfig(seed=seed)), config=cfg)
    return pipe, ch


def _arm_novel_interrupt(pipe: LiveMindPipeline, *, cue: str = "smell_novelty") -> None:
    pipe._interrupt_pending = True
    pipe._interrupt_remaining = 40
    pipe._interrupt_reason = f"novel:{cue}"
    pipe._interrupt_meta = {"cue": cue, "reason": f"novel:{cue}"}
    pipe._push_motor_directive(
        mode="idle",
        plan_name="idle",
        goal=STOP_GOAL.copy(),
        reason="stimulus_interrupt",
    )
    pipe._begin_father_advocate(
        {
            "odor": 0.55,
            "sugar": 0.0,
            "bitter": 0.0,
            "wall_hazard": 0.1,
            "energy": 0.8,
            "hunger": 0.2,
        }
    )


def test_oneshot_human_teach_then_reuse() -> None:
    """One teaching reply becomes a lasting lesson for that cue."""
    pipe, ch = _pipeline_with_unity_dialog()
    try:
        _arm_novel_interrupt(pipe, cue="smell_novelty")
        assert pipe._father_waiting is True
        assert pipe._father_question is not None
        assert ch.last_ask is not None
        assert "smell" in ch.last_ask["cue"] or ch.last_ask["question"]
        ch.inject_view_for_tests(request_id=pipe._father_request_id, viewing=True)

        # Motor must stay hard-stopped while waiting for Father.
        obs = np.zeros(197, dtype=np.float32)
        action, info = pipe.process_observation(obs)
        assert info.get("father_waiting") is True
        assert float(action[0]) == pytest.approx(0.0, abs=1e-5)
        assert float(action[1]) == pytest.approx(0.0, abs=1e-5)

        # One-shot human teaching reply (simulates Unity textbox Submit).
        ch.inject_reply_for_tests(
            request_id=pipe._father_request_id,
            answer="Sugar plume: approach slowly and taste before committing.",
            source="human",
        )
        pipe._father_polled_step = -1
        assert pipe._poll_father_advocate() is False
        assert pipe._father_waiting is False
        assert pipe._father_lesson is not None
        assert pipe._father_lesson.source == "human"
        assert "sugar" in pipe._father_lesson.answer.lower()
        lesson = pipe.father_lessons.latest_for_cue("smell_novelty")
        assert lesson is not None
        assert lesson.answer == pipe._father_lesson.answer

        # Same cue again → reuse lesson, no Unity wait.
        pipe._interrupt_pending = True
        pipe._interrupt_remaining = 20
        pipe._interrupt_reason = "novel:smell_novelty"
        pipe._interrupt_meta = {"cue": "smell_novelty"}
        pipe._begin_father_advocate({"odor": 0.6})
        assert pipe._father_waiting is False
        assert pipe._father_lesson is not None
        assert pipe._father_lesson.answer == lesson.answer
    finally:
        pipe.close()


def test_oneshot_ask_llm_button() -> None:
    """Unity 'Ask LLM' button resolves the wait with an automated lesson."""
    pipe, ch = _pipeline_with_unity_dialog(seed=8)
    try:
        _arm_novel_interrupt(pipe, cue="visual_novelty")
        assert pipe._father_waiting is True
        ch.inject_reply_for_tests(
            request_id=pipe._father_request_id,
            answer="",
            source="ask_llm",
        )
        pipe._father_polled_step = -1
        assert pipe._poll_father_advocate() is False
        assert pipe._father_waiting is False
        assert pipe._father_lesson is not None
        assert pipe._father_lesson.source == "llm"
        assert pipe._father_lesson.answer
    finally:
        pipe.close()


def test_oneshot_timeout_auto_llm_after_60s() -> None:
    """No human reply → after timeout_seconds, Father auto-answers via LLM."""
    pipe, ch = _pipeline_with_unity_dialog(seed=9, timeout_seconds=60.0, auto_llm_away_seconds=-1.0)
    try:
        _arm_novel_interrupt(pipe, cue="bitter")
        assert pipe._father_waiting is True
        pipe._father_wait_started = time.monotonic() - 61.0
        pipe._father_polled_step = -1
        assert pipe._poll_father_advocate() is False
        assert pipe._father_waiting is False
        assert pipe._father_lesson is not None
        assert pipe._father_lesson.source == "timeout_llm"
        assert "toxin" in pipe._father_lesson.answer.lower() or pipe._father_lesson.answer
    finally:
        pipe.close()


def test_oneshot_fake_run_with_llm_mode_completes() -> None:
    """Headless one-shot: mode=llm auto-answers; short live run still finishes."""
    cfg = LiveMindConfig(
        backend="in_memory",
        llm_mock=True,
        console_log=False,
        async_mind=False,
        mind_interval=4,
        max_steps=12,
        fly_id="fly_father_fake",
        father_advocate=FatherAdvocateConfig(enabled=True, mode="llm", mock=True),
        stimulus_interrupt=StimulusInterruptConfig(enabled=True, cooldown_steps=0),
    )
    from flybrain.memory.place_graph import PlaceGraphConfig

    cfg.place_graph = PlaceGraphConfig(enabled=False, backend="in_memory")
    pipe = LiveMindPipeline(FakeFlyEnv(FakeFlyConfig(seed=11)), config=cfg)
    try:
        # Manual arm + auto Father (no Unity) during the run.
        obs, _ = pipe.env.reset(seed=11)
        pipe._step = 0
        _arm_novel_interrupt(pipe, cue="smell_novelty")
        # mode=llm path answers inside _begin_father_advocate when no channel —
        # here channel is absent, but we called _begin after forcing interrupt;
        # FakeFlyEnv has no father_dialog_channel → auto llm already applied.
        assert pipe._father_waiting is False
        assert pipe._father_lesson is not None

        result = pipe.run(max_steps=8)
        assert result["steps"] == 8
        assert pipe.father_lessons.latest_for_cue("smell_novelty") is not None
    finally:
        pipe.close()


def test_oneshot_freeze_countdown_paused_while_father_waits() -> None:
    """Interrupt freeze clock must not expire until Father replies."""
    pipe, ch = _pipeline_with_unity_dialog(seed=12)
    try:
        _arm_novel_interrupt(pipe, cue="wall")
        pipe._interrupt_remaining = 3
        assert pipe._father_waiting is True
        ch.inject_view_for_tests(request_id=pipe._father_request_id, viewing=True)

        for step in range(5):
            pipe._step = step
            pipe._father_polled_step = -1
            pipe._tick_interrupt_freeze()
            assert pipe._interrupt_pending is True
            assert pipe._interrupt_remaining == 3  # frozen while waiting

        ch.inject_reply_for_tests(
            request_id=pipe._father_request_id,
            answer="Turn toward the opening; do not push the wall.",
            source="human",
        )
        pipe._step = 5
        pipe._father_polled_step = -1
        pipe._tick_interrupt_freeze()
        assert pipe._father_waiting is False
        # After reply, countdown resumes (this tick decrements once).
        assert pipe._interrupt_remaining == 2
    finally:
        pipe.close()
