"""Blocked detection: pushing forward without moving → back up and turn; keep wall hazard raw."""

from __future__ import annotations

import numpy as np
import pytest

from flybrain.curiosity.boredom import (
    BoredomConfig,
    BoredomTracker,
    apply_blocked_escape_goal,
)
from flybrain.memory.place_graph import InMemoryPlaceGraph, PlaceGraphConfig
from flybrain.phase4 import PATCH_VEC_DIM
from flybrain.sensors.mapped_feedback import MappedSensorFeedback
from flybrain.signals.builder import (
    BODY_LINEAR_VELOCITY_SCALE,
    BODY_STATE_OFFSET,
    body_speed,
)

FORWARD = np.asarray([0.75, 0.17, 0.5, 0.5, 0.0], dtype=np.float32)
WALL = 0.8


def _tracker(horizon: int = 5, hold: int = 4) -> BoredomTracker:
    return BoredomTracker(BoredomConfig(blocked_horizon_steps=horizon, escape_hold_steps=hold))


def _push(
    t: BoredomTracker,
    n: int,
    *,
    speed: float | None = 0.0,
    action=FORWARD,
    wall: float = WALL,
):
    state = t.state()
    for _ in range(n):
        state = t.observe_physical(
            position=None, action=action, speed=speed, wall_hazard=wall
        )
    return state


def test_pushing_forward_without_moving_triggers_escape_after_horizon():
    t = _tracker(horizon=5)
    state = _push(t, 4)
    assert state.blocked_streak == 4
    assert state.pushing_blocked and not state.escaping
    state = _push(t, 1)
    assert state.escaping
    assert state.blocked_streak == 0
    assert state.physical_stuck == pytest.approx(0.0)


def test_moving_body_resets_blocked_streak():
    t = _tracker(horizon=5)
    _push(t, 4)
    state = _push(t, 1, speed=0.05)
    assert state.blocked_streak == 0
    assert not state.pushing_blocked


def test_slow_forward_or_turning_in_place_is_not_blocked():
    t = _tracker(horizon=3)
    turning = np.asarray([0.1, 0.8, 0.5, 0.5, 0.0], dtype=np.float32)
    assert not _push(t, 10, action=turning).pushing_blocked


def test_unknown_motion_never_blocks():
    t = _tracker(horizon=3)
    assert not _push(t, 10, speed=None).pushing_blocked


def test_open_space_still_body_does_not_arm_escape():
    """Motion-only jam with wall≈0 must not trigger backup (false unstick)."""
    t = _tracker(horizon=3)
    assert not _push(t, 10, wall=0.0).pushing_blocked
    assert not _push(t, 10, wall=0.1).pushing_blocked


def test_position_without_change_counts_as_blocked():
    t = _tracker(horizon=3)
    state = t.state()
    for _ in range(4):
        state = t.observe_physical(
            position=np.zeros(3), action=FORWARD, wall_hazard=WALL
        )
    assert state.escaping


def test_escape_holds_then_ends_and_next_jam_turns_the_other_way():
    t = _tracker(horizon=2, hold=3)
    _push(t, 2)
    first_sign = t.escape_sign
    # Escape steps ignore the (still) body; the motor is backing out.
    assert _push(t, 2).escaping
    assert not _push(t, 1).escaping
    _push(t, 2)
    assert t.state().escaping
    assert t.escape_sign == -first_sign


def test_blocked_escape_goal_backs_up_and_turns():
    t = _tracker(horizon=1)
    state = _push(t, 1)
    cfg = t.config
    goal = apply_blocked_escape_goal(FORWARD, state, config=cfg, escape_sign=-1.0)
    assert goal[0] == pytest.approx(-cfg.escape_backup)
    assert goal[1] == pytest.approx(-cfg.escape_turn)
    assert goal[2:] == pytest.approx(FORWARD[2:])


def test_blocked_escape_goal_untouched_when_not_escaping():
    goal = apply_blocked_escape_goal(FORWARD, BoredomTracker().state())
    np.testing.assert_allclose(goal, FORWARD)


def test_body_speed_reads_horizontal_measured_velocity():
    obs = np.zeros(BODY_STATE_OFFSET + 13, dtype=np.float32)
    obs[BODY_STATE_OFFSET] = 0.03 / BODY_LINEAR_VELOCITY_SCALE
    obs[BODY_STATE_OFFSET + 1] = 0.5  # vertical ignored
    obs[BODY_STATE_OFFSET + 2] = 0.04 / BODY_LINEAR_VELOCITY_SCALE
    assert body_speed(obs) == pytest.approx(0.05)
    assert body_speed(np.zeros(BODY_STATE_OFFSET, dtype=np.float32)) is None


def test_directive_allow_pivot_keeps_reverse_past_forward_floor():
    from flybrain.motor import MotorLayerConfig
    from flybrain.motor.act_state import MotorActController, make_directive
    from flybrain.motor.layer import MotorLayer

    ctrl = MotorActController(MotorLayer(MotorLayerConfig(mode="heuristic")))
    reverse = np.asarray([-0.5, 0.55, 0.5, 0.5, 0.0], dtype=np.float32)
    ctrl.push(make_directive(mode="act_anything", goal=reverse))
    assert float(ctrl.motor.goal[0]) > 0.0, "floor applies to ordinary goals"
    ctrl.push(make_directive(mode="act_anything", goal=reverse, allow_pivot=True))
    assert float(ctrl.motor.goal[0]) == pytest.approx(-0.5)


def _feedback() -> MappedSensorFeedback:
    cfg = PlaceGraphConfig(
        enabled=True,
        backend="in_memory",
        match_min=0.9,
        familiarity_bump=0.5,
        hazard_decay_strength=0.85,
        hazard_decay_floor=0.15,
        context_enabled=False,
    )
    return MappedSensorFeedback(
        store=InMemoryPlaceGraph(),
        config=cfg,
        sense_gate_ttl_seconds=0.0,
        smell_gate_ttl_seconds=0.0,
    )


def test_unattenuated_wall_stays_raw_on_familiar_place():
    fb = _feedback()
    view = np.zeros(PATCH_VEC_DIM, dtype=np.float32)
    view[0] = 1.0
    for step in range(4):
        faded = fb.sense(raw={"wall_hazard": 0.8}, view_vec=view, step=step, now_ms=float(step))
    assert faded.reported("wall_hazard") < 0.8

    held = fb.sense(
        raw={"wall_hazard": 0.8},
        view_vec=view,
        step=5,
        now_ms=5.0,
        unattenuated=("wall_hazard",),
    )
    assert held.reported("wall_hazard") == pytest.approx(0.8)
    assert held.readings["wall_hazard"].mapped is False


def _pipeline(horizon: int = 3, hold: int = 50):
    from flybrain.env import FakeFlyEnv
    from flybrain.memory.embedder import TOTAL_EMBEDDING_DIM
    from flybrain.memory.pgvector import InMemoryShortTermMemory
    from flybrain.motor import MotorLayerConfig
    from flybrain.pipeline.live_mind import LiveMindConfig, LiveMindPipeline
    from flybrain.thinking.stimulus_interrupt import StimulusInterruptConfig

    env = FakeFlyEnv()
    env.agent_speed = lambda: 0.0  # jammed: Unity reports no body motion
    pipe = LiveMindPipeline(
        env,
        config=LiveMindConfig(
            backend="in_memory",
            llm_mock=True,
            enable_motor_layer=True,
            motor=MotorLayerConfig(act_state_enabled=True, mode="heuristic"),
            stimulus_interrupt=StimulusInterruptConfig(enabled=False),
            boredom=BoredomConfig(blocked_horizon_steps=horizon, escape_hold_steps=hold),
        ),
        memory=InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM),
    )
    return pipe, env


def test_pipeline_reads_env_speed_and_mind_tick_backs_out():
    pipe, env = _pipeline(horizon=3)
    info: dict = {"vision_cues": {"wall_hazard_raw": WALL, "wall_hazard": WALL}}
    for _ in range(3):
        pipe._observe_physical_into(info, None, FORWARD)
    assert info["blocked_escape"] is True
    assert info["blocked_wall"] == pytest.approx(WALL)

    obs, _ = env.reset(seed=0)
    _action, mind_info = pipe.process_observation(obs)
    assert float(pipe.motor.goal[0]) < 0.0, mind_info.get("motor_goal")


def test_pipeline_open_space_does_not_arm_blocked_escape():
    pipe, _env = _pipeline(horizon=3)
    info: dict = {"vision_cues": {"wall_hazard_raw": 0.0, "wall_hazard": 0.0}}
    for _ in range(10):
        pipe._observe_physical_into(info, None, FORWARD)
    assert info["blocked_escape"] is False
    assert info["blocked_streak"] == 0


def test_pipeline_keeps_wall_raw_while_blocked():
    pipe, _env = _pipeline(horizon=50)
    calls: list[dict] = []

    class _Recorder:
        def sense(self, **kwargs):
            calls.append(kwargs)
            raise StopIteration

    pipe.mapped_sensors = _Recorder()
    pipe.place_graph_config.enabled = True
    pipe.place_graph_config.update_every = 1
    view_eye = np.zeros((1, 8, 8), dtype=np.float32)

    def place_map():
        with pytest.raises(StopIteration):
            pipe._apply_place_map(
                left_eye=view_eye,
                right_eye=view_eye,
                signal_payload={"wall_hazard": 0.8},
                info={},
            )

    place_map()
    assert calls[-1]["unattenuated"] == ()
    pipe._observe_physical_into(
        {"vision_cues": {"wall_hazard_raw": WALL}}, None, FORWARD
    )
    place_map()
    assert calls[-1]["unattenuated"] == ("wall_hazard",)
