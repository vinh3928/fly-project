"""Async mind: background thoughts, motor never waits, stimulus → pause + reframe."""

from __future__ import annotations

import threading
from concurrent.futures import Executor, Future

import numpy as np
import pytest

from flybrain.pipeline.async_mind import (
    MIND_JOB_DEEP,
    MIND_JOB_THINK,
    AsyncMindScheduler,
    SerializedAccess,
)
from flybrain.pipeline.live_mind import LiveMindConfig, LiveMindPipeline, load_mind_config
from flybrain.thinking.stimulus_interrupt import StimulusInterruptDecision


class ManualExecutor(Executor):
    """Runs submitted jobs only when told, so tests control when a thought finishes."""

    def __init__(self) -> None:
        self.pending: list[tuple[Future, object]] = []

    def submit(self, fn, /, *args, **kwargs):  # type: ignore[override]
        future: Future = Future()
        self.pending.append((future, lambda: fn(*args, **kwargs)))
        return future

    def run_all(self) -> None:
        while self.pending:
            future, job = self.pending.pop(0)
            try:
                future.set_result(job())  # type: ignore[operator]
            except BaseException as exc:  # noqa: BLE001 - surfaced through the future
                future.set_exception(exc)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self.run_all()


# --- scheduler -------------------------------------------------------------------------


def test_scheduler_one_job_at_a_time_and_commits_result():
    ex = ManualExecutor()
    sched = AsyncMindScheduler(ex)
    assert sched.submit(MIND_JOB_THINK, lambda: "plan", step=3)
    assert sched.busy and sched.inflight_kind == MIND_JOB_THINK
    assert not sched.submit(MIND_JOB_THINK, lambda: "other", step=4)
    assert sched.poll() is None  # still thinking
    ex.run_all()
    result = sched.poll()
    assert (result.kind, result.value, result.submitted_step) == (MIND_JOB_THINK, "plan", 3)
    assert not sched.busy and sched.stats()["committed"] == 1


def test_scheduler_invalidate_discards_in_flight_result():
    ex = ManualExecutor()
    sched = AsyncMindScheduler(ex)
    sched.submit(MIND_JOB_THINK, lambda: "stale plan", step=0)
    assert sched.invalidate("stimulus:new")
    ex.run_all()
    assert sched.poll() is None
    assert sched.stats()["discarded"] == 1
    assert sched.stats()["last_discard_reason"] == "stimulus:new"
    assert not sched.invalidate("idle")  # nothing in flight
    sched.submit(MIND_JOB_DEEP, lambda: "reframed", step=1)
    ex.run_all()
    assert sched.poll().value == "reframed"


def test_scheduler_reraises_worker_errors_on_poll():
    ex = ManualExecutor()
    sched = AsyncMindScheduler(ex)

    def boom():
        raise RuntimeError("llm down")

    sched.submit(MIND_JOB_THINK, boom, step=0)
    ex.run_all()
    with pytest.raises(RuntimeError, match="llm down"):
        sched.poll()
    assert not sched.busy


def test_scheduler_default_executor_runs_on_background_thread():
    sched = AsyncMindScheduler()
    seen: dict = {}
    sched.submit(
        MIND_JOB_THINK, lambda: seen.setdefault("thread", threading.current_thread()), step=0
    )
    sched.wait(timeout=5)
    result = sched.poll()
    sched.shutdown()
    assert result is not None and result.value is not threading.main_thread()


def test_serialized_access_delegates_and_locks():
    class Store:
        def __init__(self):
            self.rows: list[int] = []

        def insert(self, row: int) -> int:
            self.rows.append(row)
            return len(self.rows)

    lock = threading.RLock()
    proxy = SerializedAccess(Store(), lock)
    assert proxy.insert(5) == 1
    assert proxy.rows == [5]
    proxy.extra = "x"
    assert proxy.target.extra == "x"

    held: list[bool] = []
    orig = proxy.target.insert
    proxy.target.insert = lambda row: (held.append(lock._is_owned()), orig(row))[1]  # type: ignore[attr-defined]
    proxy.insert(6)
    assert held == [True]


# --- pipeline ---------------------------------------------------------------------------


def _pipeline(ex: ManualExecutor, *, interrupt: bool = False) -> LiveMindPipeline:
    from flybrain.env import FakeFlyEnv
    from flybrain.memory.embedder import TOTAL_EMBEDDING_DIM
    from flybrain.memory.pgvector import InMemoryShortTermMemory
    from flybrain.motor import MotorLayerConfig
    from flybrain.thinking.father_advocate import FatherAdvocateConfig
    from flybrain.thinking.stimulus_interrupt import StimulusInterruptConfig

    pipe = LiveMindPipeline(
        FakeFlyEnv(),
        config=LiveMindConfig(
            backend="in_memory",
            llm_mock=True,
            enable_motor_layer=True,
            motor=MotorLayerConfig(act_state_enabled=True, mode="heuristic"),
            stimulus_interrupt=StimulusInterruptConfig(
                enabled=interrupt, deep_thought_steps=2, cooldown_steps=1000
            ),
            async_mind=True,
            async_min_steps=2,
            async_reframe_max_steps=50,
            use_vision_head=False,
            father_advocate=FatherAdvocateConfig(enabled=False),
        ),
        memory=InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM),
    )
    pipe.mind_scheduler = AsyncMindScheduler(ex)
    return pipe


def _start(pipe: LiveMindPipeline) -> np.ndarray:
    obs, _ = pipe.env.reset()
    pipe.process_observation(obs)  # first thought inline, as run() does
    pipe._step = 2
    pipe._last_mind_submit_step = 0
    return obs


def test_async_config_loads_from_yaml():
    cfg = LiveMindConfig.from_yaml(load_mind_config())
    assert cfg.async_mind is True
    assert cfg.async_min_steps >= 1 and cfg.async_reframe_max_steps > 0
    assert LiveMindConfig().async_mind is False  # tests / benchmark stay deterministic


@pytest.mark.parametrize(
    ("argv", "async_mind", "interval"),
    [([], True, 40), (["--sync-mind"], False, 40), (["--mind-interval", "4"], False, 4)],
)
def test_cli_mind_mode_flags(argv, async_mind, interval):
    from scripts.run_live_mind import _parse_args, apply_mind_mode_args

    cfg = LiveMindConfig.from_yaml(load_mind_config())
    apply_mind_mode_args(cfg, _parse_args(argv))
    assert (cfg.async_mind, cfg.mind_interval) == (async_mind, interval)


def test_async_pipeline_wraps_stm_for_cross_thread_access():
    pipe = _pipeline(ManualExecutor())
    assert isinstance(pipe.memory, SerializedAccess)
    pipe.close()


def test_motor_acts_every_step_while_thought_in_flight_then_commits():
    ex = ManualExecutor()
    pipe = _pipeline(ex)
    obs = _start(pipe)
    ticks = pipe._mind_ticks

    action, info = pipe.process_async_step(obs)
    assert action.shape == (5,)
    assert info["mind_tick"] is False
    assert info["async_mind"]["inflight"] == MIND_JOB_THINK

    pipe._step += 1
    action, info = pipe.process_async_step(obs)  # mind still busy: motor keeps going
    assert action.shape == (5,) and info["mind_tick"] is False
    assert pipe.mind_scheduler.submitted == 1

    ex.run_all()
    pipe._step += 1
    _, info = pipe.process_async_step(obs)
    assert info["mind_tick"] is True
    assert info["async_mind"]["lag_steps"] == 2
    assert pipe._mind_ticks == ticks + 1
    assert info["refined_thought"]
    ex.run_all()
    pipe.close()


def test_async_respects_min_steps_between_thoughts():
    ex = ManualExecutor()
    pipe = _pipeline(ex)
    obs = _start(pipe)
    pipe.process_async_step(obs)
    ex.run_all()
    pipe._step += 1  # 1 step after submit < async_min_steps=2
    pipe.process_async_step(obs)  # commits, but too soon to submit again
    assert not pipe.mind_scheduler.busy
    pipe._step += 1
    pipe.process_async_step(obs)
    assert pipe.mind_scheduler.inflight_kind == MIND_JOB_THINK
    ex.run_all()
    pipe.close()


def _arm_once(monkeypatch, pipe: LiveMindPipeline) -> None:
    import flybrain.pipeline.live_mind as live_mind

    calls = {"n": 0}

    def fake_eval(**_kw):
        calls["n"] += 1
        return StimulusInterruptDecision(
            candidate=True, should_arm=calls["n"] == 1, reason="novel_test", cue="vision"
        )

    monkeypatch.setattr(live_mind, "evaluate_stimulus_interrupt", fake_eval)


def test_new_stimulus_pauses_discards_stale_thought_and_reframes(monkeypatch):
    ex = ManualExecutor()
    pipe = _pipeline(ex, interrupt=True)
    _arm_once(monkeypatch, pipe)
    obs = _start(pipe)
    ticks = pipe._mind_ticks

    # Thought A in flight when the stimulus arrives on the motor tick: pause.
    pipe.stimulus_interrupt.enabled = False
    pipe.process_async_step(obs)
    assert pipe.mind_scheduler.inflight_kind == MIND_JOB_THINK
    pipe.stimulus_interrupt.enabled = True
    pipe._step += 1
    action, info = pipe.process_async_step(obs)
    assert pipe._interrupt_pending
    assert info["stimulus_interrupt"] is True
    assert pipe.motor.goal.tolist()[:2] == pytest.approx([0.0, 0.0])

    # Stale thought aborts at the LLM boundary and never reaches the motor.
    ex.run_all()
    assert pipe._mind_ticks == ticks
    pipe._step += 1
    _, info = pipe.process_async_step(obs)
    assert pipe.mind_scheduler.discarded == 1
    assert info["mind_tick"] is False
    assert pipe.mind_scheduler.inflight_kind == MIND_JOB_DEEP

    # deep_thought_steps elapsed, but no reframed thought yet → stay paused.
    for _ in range(5):
        pipe._tick_interrupt_freeze()
    assert pipe._interrupt_pending

    ex.run_all()
    pipe._step += 1
    _, info = pipe.process_async_step(obs)
    assert info["mind_tick"] is True
    assert pipe._deep_thoughts_done == 1
    assert info.get("deep_thought_gain") is not None  # heightened-senses deep thought
    pipe._tick_interrupt_freeze()
    assert not pipe._interrupt_pending  # reframed → resume
    ex.run_all()
    pipe.close()


def test_reframe_wait_is_capped(monkeypatch):
    ex = ManualExecutor()
    pipe = _pipeline(ex, interrupt=True)
    _arm_once(monkeypatch, pipe)
    obs = _start(pipe)
    pipe.process_async_step(obs)
    assert pipe._interrupt_pending
    # deep_thought_steps=2: tick 2 reaches zero, then waits async_reframe_max_steps more.
    for _ in range(1 + pipe.config.async_reframe_max_steps):
        pipe._tick_interrupt_freeze()
    assert pipe._interrupt_pending
    pipe._tick_interrupt_freeze()
    assert not pipe._interrupt_pending
    ex.run_all()
    pipe.close()


def test_deep_result_after_freeze_cleared_is_discarded(monkeypatch):
    ex = ManualExecutor()
    pipe = _pipeline(ex, interrupt=True)
    _arm_once(monkeypatch, pipe)
    obs = _start(pipe)
    pipe.process_async_step(obs)  # arms; submits deep (worker idle)
    assert pipe.mind_scheduler.inflight_kind == MIND_JOB_DEEP
    pipe._deep_thoughts_done = 1  # pretend an earlier reframe landed
    pipe._interrupt_remaining = 0
    pipe._tick_interrupt_freeze()
    assert not pipe._interrupt_pending
    ex.run_all()
    pipe._step += 1
    pipe.process_async_step(obs)
    assert pipe.mind_scheduler.discarded == 1
    ex.run_all()
    pipe.close()


def test_async_run_end_to_end_on_real_thread():
    from flybrain.env import FakeFlyEnv
    from flybrain.memory.embedder import TOTAL_EMBEDDING_DIM
    from flybrain.memory.pgvector import InMemoryShortTermMemory

    pipe = LiveMindPipeline(
        FakeFlyEnv(),
        config=LiveMindConfig(
            backend="in_memory",
            llm_mock=True,
            async_mind=True,
            async_min_steps=1,
            use_vision_head=False,
        ),
        memory=InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM),
    )
    out = pipe.run(max_steps=30)
    pipe.close()
    assert out["steps"] == 30
    assert out["motor_ticks"] >= 30
    assert out["async_mind"]["submitted"] >= 1
    assert out["mind_ticks"] >= 1
