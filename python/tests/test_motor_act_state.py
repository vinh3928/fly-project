"""Motor act-state controller: mind-pushed act_anything / still / idle."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from flybrain.env import FakeFlyEnv
from flybrain.memory.embedder import TOTAL_EMBEDDING_DIM
from flybrain.memory.pgvector import InMemoryShortTermMemory
from flybrain.motor import (
    MotorActController,
    MotorLayer,
    MotorLayerConfig,
    make_directive,
    mode_for_plan,
)
from flybrain.pipeline.live_mind import LiveMindConfig, LiveMindPipeline
from flybrain.thinking.stimulus_interrupt import STOP_GOAL, StimulusInterruptConfig


def test_mode_for_plan_mapping() -> None:
    assert mode_for_plan("explore") == "act_anything"
    assert mode_for_plan("forward") == "act_anything"
    assert mode_for_plan("stop") == "still"
    assert mode_for_plan("still") == "still"
    assert mode_for_plan(None) == "still"
    assert mode_for_plan("idle") == "idle"


def test_controller_push_replaces_goal_and_mode() -> None:
    motor = MotorLayer(MotorLayerConfig(mode="heuristic", smoothing=1.0, max_delta=1.0))
    ctrl = MotorActController(motor)
    assert ctrl.mode == "still"

    explore = make_directive(
        mode="act_anything",
        plan_name="explore",
        goal=np.asarray([0.8, 0.1, 0.5, 0.7, 0.0], dtype=np.float32),
        reason="mind_think",
        step=1,
    )
    ctrl.push(explore)
    assert ctrl.mode == "act_anything"
    assert ctrl.current.plan_name == "explore"
    assert float(motor.goal[0]) >= 0.4  # locomotion floor

    action = ctrl.apply_tick(None)
    assert action.shape == (5,)
    assert abs(float(action[0])) > 0.05

    ctrl.push(make_directive(mode="still", plan_name="stop", reason="mind_think", step=2))
    assert ctrl.mode == "still"
    stopped = ctrl.apply_tick(None)
    assert abs(float(stopped[0])) <= 0.08


def test_idle_and_still_emit_stop_goal() -> None:
    motor = MotorLayer(MotorLayerConfig(mode="heuristic", smoothing=1.0, max_delta=1.0))
    ctrl = MotorActController(motor)
    ctrl.push(
        make_directive(
            mode="act_anything",
            plan_name="forward",
            goal=np.asarray([1.0, 0.0, 0.5, 0.7, 0.0], dtype=np.float32),
            step=0,
        )
    )
    ctrl.push(make_directive(mode="idle", plan_name="idle", reason="si", step=1))
    assert ctrl.mode == "idle"
    np.testing.assert_allclose(ctrl.current.goal, STOP_GOAL, atol=1e-5)
    action = ctrl.apply_tick(None)
    assert abs(float(action[0])) <= 0.08


def test_motor_only_does_not_call_embedder() -> None:
    env = FakeFlyEnv()
    mem = InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM)
    pipe = LiveMindPipeline(
        env,
        config=LiveMindConfig(
            backend="in_memory",
            llm_mock=True,
            enable_motor_layer=True,
            mind_interval=40,
            motor=MotorLayerConfig(act_state_enabled=True, mode="heuristic"),
            stimulus_interrupt=StimulusInterruptConfig(enabled=False),
        ),
        memory=mem,
    )
    obs, _ = env.reset()
    pipe._push_motor_directive(
        mode="act_anything",
        plan_name="explore",
        goal=np.asarray([0.7, 0.0, 0.5, 0.6, 0.0], dtype=np.float32),
        reason="test",
    )
    pipe._last_mind_info = {
        "intent": {"future": {"chosen_name": "explore"}},
        "affect": {},
        "act_mode": "act_anything",
    }
    pipe._step = 1
    pipe.embedder.embed = MagicMock(side_effect=AssertionError("embed should not run"))
    pipe.embedder.embed_parts = MagicMock(side_effect=AssertionError("embed_parts should not run"))
    action, info = pipe.process_motor_only(obs)
    assert info.get("mind_tick") is False
    assert info.get("act_mode") == "act_anything"
    assert info.get("architecture") == "B"
    assert action.shape == (5,)
    pipe.embedder.embed.assert_not_called()
    pipe.embedder.embed_parts.assert_not_called()
    pipe.close()


def test_si_arm_pushes_idle_and_hard_stop() -> None:
    env = FakeFlyEnv()
    mem = InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM)
    pipe = LiveMindPipeline(
        env,
        config=LiveMindConfig(
            backend="in_memory",
            llm_mock=True,
            enable_motor_layer=True,
            mind_interval=40,
            motor=MotorLayerConfig(act_state_enabled=True),
            stimulus_interrupt=StimulusInterruptConfig(enabled=True, cooldown_steps=0),
        ),
        memory=mem,
    )
    obs, _ = env.reset()
    pipe._push_motor_directive(
        mode="act_anything",
        plan_name="explore",
        goal=np.asarray([0.9, 0.0, 0.5, 0.7, 0.0], dtype=np.float32),
        reason="setup",
    )
    pipe._last_mind_info = {"intent": {"future": {"chosen_name": "explore"}}, "affect": {}}
    pipe._step = 1
    pipe._interrupt_pending = True
    pipe._interrupt_remaining = 5
    pipe._interrupt_reason = "test"
    # Pending freeze should already be idle; hard-stop path stamps + stops.
    if pipe.act_controller is not None:
        pipe.act_controller.push(
            make_directive(mode="idle", plan_name="idle", reason="test", step=1)
        )
    action, info = pipe.process_motor_only(obs)
    assert info.get("stimulus_interrupt_pending") is True
    assert info.get("act_mode") == "idle"
    assert abs(float(action[0])) <= 0.08
    pipe.close()


def test_mind_think_pushes_act_mode() -> None:
    env = FakeFlyEnv()
    mem = InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM)
    pipe = LiveMindPipeline(
        env,
        config=LiveMindConfig(
            backend="in_memory",
            llm_mock=True,
            enable_motor_layer=True,
            mind_interval=40,
            motor=MotorLayerConfig(act_state_enabled=True, mode="heuristic"),
            stimulus_interrupt=StimulusInterruptConfig(enabled=False),
            use_vision_head=False,
        ),
        memory=mem,
    )
    obs, _ = env.reset()
    _action, info = pipe.process_observation(obs)
    assert info.get("mind_tick") is True
    assert info.get("act_mode") in ("act_anything", "still")
    assert pipe.act_controller is not None
    assert pipe.act_controller.current.reason == "mind_think"
    # Subsequent motor tick keeps the pushed mode without rethinking.
    action2, info2 = pipe.process_motor_only(obs)
    assert info2.get("mind_tick") is False
    assert info2.get("act_mode") == info.get("act_mode")
    assert action2.shape == (5,)
    pipe.close()


def test_config_loads_act_state_enabled() -> None:
    cfg = LiveMindConfig.from_yaml(
        {
            "fly_id": "fly_x",
            "motor": {"enabled": True, "act_state_enabled": True, "mind_interval": 40},
            "live": {},
            "database": {"backend": "in_memory"},
            "llm": {"mock": True},
            "thinking": {},
        }
    )
    assert cfg.motor is not None
    assert cfg.motor.act_state_enabled is True
