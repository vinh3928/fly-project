"""Hungry sugar search: turn grows while smell stops improving, resets on rise."""

from __future__ import annotations

import pytest

from flybrain.pipeline.live_mind import LiveMindConfig, format_live_step_log
from flybrain.thinking.sugar_search import SugarSearchConfig, SugarSearchTracker
from flybrain.thinking.two_layer_future import (
    SUGAR_TURN_GAIN,
    cue_turn,
    sugar_approach_turn,
)


def _sig(smell: float, bearing: float = 0.0, *, hunger: float = 0.8) -> dict:
    return {"sugar_smell": smell, "sugar_bearing": bearing, "hunger": hunger, "energy": 0.2}


def _tracker(**kw) -> SugarSearchTracker:
    return SugarSearchTracker(SugarSearchConfig(**kw))


def test_off_when_not_hungry_or_no_smell():
    t = _tracker()
    assert t.observe(_sig(0.5, hunger=0.0) | {"energy": 1.0}).turn is None
    assert t.observe(_sig(0.0)).mode == "off"


def test_first_tick_uses_antenna_bearing():
    state = _tracker().observe(_sig(0.1, bearing=0.4))
    assert state.mode == "start"
    assert state.turn == pytest.approx(sugar_approach_turn(0.4))


def test_rising_smell_keeps_gentle_bearing_steering():
    t = _tracker()
    t.observe(_sig(0.10, bearing=0.2))
    state = t.observe(_sig(0.12, bearing=0.2))
    assert state.mode == "approach"
    assert state.turn == pytest.approx(SUGAR_TURN_GAIN * 0.2)


def test_flat_smell_turns_harder_each_tick_up_to_cap():
    t = _tracker(base_turn=0.25, turn_step=0.1, max_turn=0.5)
    t.observe(_sig(0.1, bearing=0.3))
    turns = [t.observe(_sig(0.1, bearing=0.3)).turn for _ in range(5)]
    assert turns[:3] == pytest.approx([0.25, 0.35, 0.45])
    assert turns[3:] == pytest.approx([0.5, 0.5])


def test_search_turns_even_with_zero_bearing():
    t = _tracker()
    t.observe(_sig(0.1, bearing=0.0))
    state = t.observe(_sig(0.1, bearing=0.0))
    assert state.mode == "search"
    assert abs(state.turn) > 0.2


def test_side_follows_bearing_then_sticks_when_bearing_vanishes():
    t = _tracker()
    t.observe(_sig(0.1, bearing=-0.4))
    assert t.observe(_sig(0.1, bearing=-0.4)).turn < 0
    assert t.observe(_sig(0.1, bearing=0.0)).turn < 0


def test_sharp_drop_reverses():
    t = _tracker(sharp_drop=0.05, reverse_turn=1.0)
    t.observe(_sig(0.2, bearing=0.3))
    state = t.observe(_sig(0.15, bearing=0.3))
    assert state.mode == "reverse"
    assert state.turn == pytest.approx(1.0)


def test_rise_resets_search_streak():
    t = _tracker(base_turn=0.25, turn_step=0.1)
    t.observe(_sig(0.1))
    for _ in range(4):
        t.observe(_sig(0.1))
    assert t.observe(_sig(0.2, bearing=0.1)).mode == "approach"
    again = t.observe(_sig(0.2))
    assert again.streak == 1
    assert abs(again.turn) == pytest.approx(0.25)


def test_cue_turn_prefers_search_turn_over_bearing():
    sig = _sig(0.1, bearing=0.0) | {"sugar_search_turn": 0.6}
    assert cue_turn(sig) == (pytest.approx(0.6), "sugar")
    assert cue_turn(_sig(0.1, bearing=0.5))[0] == pytest.approx(sugar_approach_turn(0.5))


def test_config_loads_from_yaml_section():
    cfg = LiveMindConfig.from_yaml({"sugar_search": {"max_turn": 0.6, "bogus": 1}})
    assert cfg.sugar_search.max_turn == pytest.approx(0.6)


def test_steer_off_until_search_active():
    assert _tracker().steer(0.5) is None


def _live(bearing: float, smell: float = 0.1, **extra) -> dict:
    return {"sugar_bearing": bearing, "sugar_smell": smell, **extra}


def test_steer_pivots_in_place_for_sharp_bearing():
    t = _tracker(steer_gain=1.5, pivot_turn=0.8, cruise_forward=0.6, pivot_max_ticks=12)
    t.observe(_sig(0.1, bearing=0.7))
    out = t.steer(_live(0.7))
    assert out.turn == pytest.approx(1.0)
    assert out.forward == pytest.approx(0.0)
    assert out.mode == "orient"
    assert out.proboscis == pytest.approx(0.0)


def test_steer_align_then_go_after_pivot_budget():
    """Saturated bearing: pivot briefly, then crawl while still turning."""
    t = _tracker(
        steer_gain=1.5,
        pivot_turn=0.8,
        cruise_forward=0.6,
        pivot_max_ticks=3,
        align_go_forward=0.35,
    )
    t.observe(_sig(0.1, bearing=0.7))
    pivots = [t.steer(_live(0.7)) for _ in range(3)]
    assert all(o.forward == pytest.approx(0.0) for o in pivots)
    going = t.steer(_live(0.7))
    assert going.turn == pytest.approx(1.0)
    assert going.forward == pytest.approx(0.35)
    # Aligning resets the pivot streak so a later sharp turn can pivot again.
    aligned = t.steer(_live(0.0))
    assert aligned.forward == pytest.approx(0.6)
    assert t.steer(_live(0.7)).forward == pytest.approx(0.0)


def test_steer_walks_straight_when_aligned():
    t = _tracker(cruise_forward=0.6)
    t.observe(_sig(0.1, bearing=0.0))
    out = t.steer(_live(0.0))
    assert out.turn == pytest.approx(0.0)
    assert out.forward == pytest.approx(0.6)


def test_steer_follows_live_bearing_each_step():
    t = _tracker(steer_gain=1.5, pivot_turn=0.8, cruise_forward=0.6)
    t.observe(_sig(0.1, bearing=0.6))
    turns = [t.steer(_live(b)).turn for b in (0.6, 0.3, 0.1, -0.1)]
    assert turns == pytest.approx([0.9, 0.45, 0.15, -0.15])
    forwards = [t.steer(_live(b)).forward for b in (0.6, 0.1)]
    assert forwards[0] < forwards[1]


def test_steer_uses_search_turn_until_antennae_pick_side():
    t = _tracker(reverse_turn=1.0, sharp_drop=0.05)
    t.observe(_sig(0.2, bearing=0.3))
    t.observe(_sig(0.1, bearing=0.0))  # sugar behind: reverse, bearing ~0
    out = t.steer(_live(0.0))
    assert out.turn == pytest.approx(1.0)
    assert out.forward == pytest.approx(0.0)
    assert t.steer(_live(-0.4)).turn < 0  # antennae now see it on the left
    assert t.steer(_live(0.0)).turn == pytest.approx(0.0)  # aligned after locking


def test_steer_near_source_extends_proboscis_and_slows():
    t = _tracker(cruise_forward=0.6, probe_smell=0.6, arrive_slow=0.6, probe_extension=1.0)
    t.observe(_sig(0.85, bearing=0.0))
    far = t.steer(_live(0.0, smell=0.5))
    near = t.steer(_live(0.0, smell=0.85))
    assert far.mode == "orient" and far.proboscis == pytest.approx(0.0)
    assert near.mode == "probe" and near.proboscis == pytest.approx(1.0)
    assert near.forward < far.forward
    assert near.forward > 0.0


def test_steer_stops_to_feed_when_proboscis_tastes_sugar():
    t = _tracker()
    t.observe(_sig(0.85, bearing=0.0))
    out = t.steer(_live(0.4, smell=0.85, proboscis_contact=1.0, sugar=0.9))
    assert out.mode == "feed"
    assert (out.forward, out.turn, out.proboscis) == pytest.approx((0.0, 0.0, 1.0))


def test_floor_contact_without_sugar_keeps_probing():
    t = _tracker()
    t.observe(_sig(0.85, bearing=0.0))
    assert t.steer(_live(0.0, smell=0.85, proboscis_contact=1.0, sugar=0.0)).mode == "probe"


ON_FOOD = {"proboscis_contact": 1.0, "sugar": 0.9}


def test_feed_bout_holds_motor_still_for_feed_ticks_even_if_contact_flickers():
    t = _tracker(feed_ticks=40)
    t.observe(_sig(0.85))
    outs = [t.steer(_live(0.6, smell=0.85, **ON_FOOD))]
    outs += [t.steer(_live(0.6, smell=0.85)) for _ in range(39)]  # contact lost
    assert all(o.mode == "feed" and o.forward == 0.0 and o.turn == 0.0 for o in outs)
    assert [o.ticks_left for o in (outs[0], outs[-1])] == [39, 0]
    assert t.steer(_live(0.6, smell=0.85)).mode != "feed"  # bout over, resumes steering


def test_feed_bout_repeats_while_proboscis_on_sugar_and_hungry():
    t = _tracker(feed_ticks=5)
    t.observe(_sig(0.85))
    modes = [t.steer(_live(0.0, smell=0.85, **ON_FOOD)).mode for _ in range(12)]
    assert modes == ["feed"] * 12


def test_sated_mid_bout_ends_feeding():
    t = _tracker(feed_ticks=40)
    t.observe(_sig(0.85))
    t.steer(_live(0.0, smell=0.85, **ON_FOOD))
    assert t.feed_ticks_left == 39
    t.observe(_sig(0.85, hunger=0.0) | {"energy": 1.0})
    assert t.feed_ticks_left == 0
    assert t.steer(_live(0.0, smell=0.85, **ON_FOOD)) is None


def _energy_sig(energy: float) -> dict:
    return _sig(0.85, hunger=1.0 - energy) | {"energy": energy}


def test_is_hungry_below_seek_threshold_and_holds_until_sated_while_feeding():
    from flybrain.thinking.composer import (
        ENERGY_SATED,
        ENERGY_SEEK_MAX,
        FEEDING_UNTIL_SATED_KEY,
        HUNGER_SEEK_MIN,
        is_hungry,
        is_sated,
    )

    assert (HUNGER_SEEK_MIN, ENERGY_SEEK_MAX, ENERGY_SATED) == (0.45, 0.55, 0.9)
    assert is_hungry({"energy": 0.29, "hunger": 0.71})
    assert is_hungry({"energy": 0.55, "hunger": 0.45})  # hunger 0.45 → seek
    assert not is_hungry({"energy": 0.6, "hunger": 0.4})
    assert not is_hungry({"energy": 0.8, "hunger": 0.2})
    feeding = {"energy": 0.85, "hunger": 0.15, FEEDING_UNTIL_SATED_KEY: True}
    assert is_hungry(feeding) and not is_sated(feeding)
    sated = feeding | {"energy": 0.9, "hunger": 0.1}
    assert not is_hungry(sated) and is_sated(sated)


def test_feeding_continues_past_seek_threshold_until_energy_09():
    t = _tracker(feed_ticks=3)
    t.observe(_energy_sig(0.25))
    assert t.steer(_live(0.0, smell=0.85, **ON_FOOD)).mode == "feed"
    assert t.feeding_until_sated
    for energy in (0.35, 0.6, 0.8, 0.89):
        t.observe(_energy_sig(energy))
        assert t.active, energy
        assert t.steer(_live(0.0, smell=0.85, **ON_FOOD)).mode == "feed"
    t.observe(_energy_sig(0.9))
    assert not t.active and not t.feeding_until_sated
    assert t.steer(_live(0.0, smell=0.85, **ON_FOOD)) is None


def test_no_search_above_seek_threshold_without_feeding_first():
    t = _tracker()
    for energy in (0.6, 0.8):
        t.observe(_energy_sig(energy))
        assert not t.active and not t.feeding_until_sated


def test_search_starts_when_energy_below_seek_max():
    t = _tracker()
    state = t.observe(_energy_sig(0.5))
    assert state.turn is not None
    assert t.active


def test_pipeline_flags_feeding_session_in_signals_for_composer():
    from flybrain.thinking.composer import FEEDING_UNTIL_SATED_KEY, is_hungry

    pipe = _pipeline()
    payloads: list[dict] = []
    real_observe = pipe.sugar_search.observe

    def observe_mid_session(signals):
        out = real_observe(signals)
        pipe.sugar_search._feeding_session = True
        payloads.append(signals)
        return out

    pipe.sugar_search.observe = observe_mid_session  # type: ignore[method-assign]
    obs, _ = pipe.env.reset()
    pipe.process_observation(obs)
    assert payloads, "mind tick ran"
    assert payloads[-1].get(FEEDING_UNTIL_SATED_KEY) is True
    assert is_hungry(payloads[-1] | {"energy": 0.8, "hunger": 0.2})


def test_feet_only_bout_without_proboscis_taste_cools_down():
    t = _tracker(feed_ticks=3)
    t.observe(_sig(0.85))
    feet_only = _live(0.0, smell=0.85, sugar=0.9)  # feet on sugar, proboscis off it
    assert [t.steer(feet_only).mode for _ in range(3)] == ["feed"] * 3
    # No proboscis contact during that bout → reposition instead of standing forever.
    assert [t.steer(feet_only).mode for _ in range(3)] == ["probe"] * 3
    assert t.steer(feet_only).mode == "feed"
    # Proboscis on sugar starts a bout immediately, even during cooldown.
    t2 = _tracker(feed_ticks=3)
    t2.observe(_sig(0.85))
    for _ in range(3):
        t2.steer(feet_only)
    assert t2.steer(_live(0.0, smell=0.85, **ON_FOOD)).mode == "feed"


def test_feeding_now_requires_hungry_search():
    t = _tracker()
    on_food = _live(0.0, smell=0.85, proboscis_contact=1.0, sugar=0.9)
    assert not t.feeding_now(on_food)
    t.observe(_sig(0.85))
    assert t.feeding_now(on_food)
    t.observe(_sig(0.85, hunger=0.0) | {"energy": 1.0})  # sated → stop feeding
    assert not t.feeding_now(on_food)
    assert t.steer(on_food) is None


def _pipeline():
    import numpy as np

    from flybrain.env import FakeFlyEnv
    from flybrain.memory.embedder import TOTAL_EMBEDDING_DIM
    from flybrain.memory.pgvector import InMemoryShortTermMemory
    from flybrain.motor import MotorLayerConfig
    from flybrain.pipeline.live_mind import LiveMindPipeline
    from flybrain.thinking.stimulus_interrupt import StimulusInterruptConfig

    env = FakeFlyEnv()
    pipe = LiveMindPipeline(
        env,
        config=LiveMindConfig(
            backend="in_memory",
            llm_mock=True,
            enable_motor_layer=True,
            motor=MotorLayerConfig(act_state_enabled=True, mode="heuristic"),
            stimulus_interrupt=StimulusInterruptConfig(enabled=False),
        ),
        memory=InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM),
    )
    pipe._push_motor_directive(
        mode="act_anything",
        plan_name="feed",
        goal=np.asarray([0.7, 0.0, 0.5, 0.6, 0.0], dtype=np.float32),
        reason="test",
    )
    return pipe


class _Sensory:
    def __init__(self, bearing: float, **extra) -> None:
        self.summary = {"sugar_bearing": bearing, "sugar_smell": 0.1, **extra}


def test_pipeline_orient_pivots_without_forward_floor_then_restores():
    pipe = _pipeline()
    pipe.sugar_search.observe(_sig(0.1, bearing=1.0))
    info: dict = {}
    pipe._apply_sugar_orient(_Sensory(1.0), info)
    assert info["sugar_orient"][0] == pytest.approx(0.0)
    assert float(pipe.motor.goal[0]) == pytest.approx(0.0)
    assert float(pipe.motor.goal[1]) == pytest.approx(1.0)

    pipe.sugar_search.observe(_sig(0.1, hunger=0.0) | {"energy": 1.0})
    info = {"sugar_orient": [0.0, 1.0]}
    pipe._apply_sugar_orient(_Sensory(0.7), info)
    assert "sugar_orient" not in info
    assert float(pipe.motor.goal[0]) == pytest.approx(0.7)
    pipe.close()


def test_pipeline_feeding_sets_proboscis_and_blocks_interrupt_arming(monkeypatch):
    import flybrain.pipeline.live_mind as live_mind

    def _no_eval(**_kw):
        raise AssertionError("feeding must not evaluate a stimulus interrupt")

    monkeypatch.setattr(live_mind, "evaluate_stimulus_interrupt", _no_eval)
    pipe = _pipeline()
    pipe.stimulus_interrupt.enabled = True
    pipe.sugar_search.observe(_sig(0.85))
    sensory = _Sensory(0.0, sugar_smell=0.85, proboscis_contact=1.0, sugar=0.9)
    info: dict = {}
    pipe._apply_sugar_orient(sensory, info)
    assert info["sugar_orient_mode"] == "feed(39)"
    assert pipe.motor.goal.tolist()[:2] == pytest.approx([0.0, 0.0])
    assert float(pipe.motor.goal[4]) == pytest.approx(1.0)

    pipe._arm_stimulus_interrupt(
        observation=None,
        left_eye=None,
        right_eye=None,
        sensory=sensory,
        affect={},
        visual_nov=None,
        info=info,
    )
    assert not pipe._interrupt_pending
    pipe.close()


def test_pipeline_feed_beats_active_route_home():
    """Route must not force p0.0 while standing on sugar (live_mind 12180)."""
    from flybrain.thinking.route_home import RouteHomeState

    pipe = _pipeline()
    pipe.sugar_search.observe(_sig(0.85))
    pipe.route_home._state = RouteHomeState(
        mode="route",
        turn=-0.05,
        forward=0.65,
        home_id="elsewhere",
        hops_left=2,
        reason="path",
        food_kind="sugar_puddle",
    )
    sensory = _Sensory(0.0, sugar_smell=0.85, proboscis_contact=1.0, sugar=0.9)
    info: dict = {}
    pipe._apply_sugar_orient(sensory, info)
    assert info["sugar_orient_mode"].startswith("feed(")
    assert info["sugar_orient"][2] == pytest.approx(1.0)
    assert "route_home_orient" not in info
    assert float(pipe.motor.goal[4]) == pytest.approx(1.0)
    pipe.close()


def test_pipeline_orient_skipped_when_motor_held_still():
    pipe = _pipeline()
    pipe._push_motor_directive(mode="still", plan_name="stop", goal=None, reason="test")
    pipe.sugar_search.observe(_sig(0.1, bearing=0.7))
    info: dict = {}
    pipe._apply_sugar_orient(_Sensory(0.7), info)
    assert "sugar_orient" not in info
    pipe.close()


def test_motor_set_goal_allow_pivot_skips_floor():
    import numpy as np

    from flybrain.motor import MotorLayer, orient_goal

    motor = MotorLayer()
    goal = orient_goal(np.asarray([0.5, 0.0, 0.4, 0.6, 0.2], dtype=np.float32), 0.0, -1.0)
    motor.set_goal(goal, allow_pivot=True)
    assert motor.goal.tolist() == pytest.approx([0.0, -1.0, 0.4, 0.6, 0.2])
    motor.set_goal(goal)
    assert float(motor.goal[0]) >= 0.4


def test_log_shows_unity_reported_proboscis_state():
    line = format_live_step_log(
        3,
        [0.0, 0.0, 0.5, 0.5, 1.0],
        {
            "sugar_smell": 0.8,
            "sugar_bearing": 0.0,
            "proboscis_extension": 0.0,
            "proboscis_contact": 0.0,
            "taste_sugar": 0.0,
        },
    )
    assert "prob(ext=0.00,touch=0,taste=0.00)" in line


def test_sniff_log_shows_search_mode():
    line = format_live_step_log(
        3,
        [0.5, 0.45, 0.5, 0.5, 0.0],
        {
            "sugar_smell": 0.05,
            "sugar_bearing": 0.0,
            "sugar_search": {"mode": "search", "turn": 0.45, "change": 0.0, "streak": 3},
        },
    )
    assert "search=+0.45" in line
