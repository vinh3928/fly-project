"""Phase 9 architecture B: mind goals + motor layer."""

from __future__ import annotations

import numpy as np

from flybrain.env_locomotion import FakeLocomotionConfig, FakeLocomotionEnv
from flybrain.motor import (
    HeuristicMotorTracker,
    MotorLayer,
    MotorLayerConfig,
    compute_locomotion_reward,
    goal_from_intent,
)
from flybrain.phase9 import MAX_TRACKING_ERROR, MIN_FORWARD_DISPLACEMENT
from flybrain.pipeline.live_mind import LiveMindConfig, LiveMindPipeline, make_live_env
from flybrain.training.train_motor_layer import train_motor_policy


def test_heuristic_tracker_approaches_goal() -> None:
    tracker = HeuristicMotorTracker(smoothing=0.5, max_delta=0.5)
    goal = np.asarray([1.0, -0.5, 0.5, 0.8, 0.0], dtype=np.float32)
    tracker.reset(np.zeros(5, dtype=np.float32))
    err0 = float(np.mean(np.abs(tracker.command - goal)))
    for _ in range(20):
        tracker.act(goal)
    err1 = float(np.mean(np.abs(tracker.command - goal)))
    assert err1 < err0
    assert err1 < 0.2


def test_locomotion_reward_prefers_tracking() -> None:
    goal = np.asarray([1.0, 0.0, 0.5, 0.5, 0.0], dtype=np.float32)
    good = compute_locomotion_reward(goal=goal, action=goal, forward_progress=0.05)
    bad = compute_locomotion_reward(
        goal=goal,
        action=np.asarray([-1.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32),
        forward_progress=0.05,
    )
    assert good > bad


def test_fake_locomotion_tracks_forward_goal() -> None:
    env = FakeLocomotionEnv(FakeLocomotionConfig(seed=0, max_steps=120))
    goal = np.asarray([1.0, 0.0, 0.5, 0.7, 0.0], dtype=np.float32)
    obs, info = env.reset(seed=0, options={"goal": goal})
    tracker = HeuristicMotorTracker(smoothing=0.45, max_delta=0.5)
    tracker.reset(goal)
    for _ in range(100):
        action = tracker.act(goal, observation=obs)
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            break
    assert info["forward_displacement"] >= MIN_FORWARD_DISPLACEMENT
    assert info["mean_tracking_error"] <= MAX_TRACKING_ERROR


def test_live_mind_architecture_b_mind_interval() -> None:
    env = make_live_env(fake_locomotion=True, seed=42)
    cfg = LiveMindConfig(
        backend="in_memory",
        max_steps=20,
        llm_mock=True,
        seed=42,
        use_vision_head=False,
        enable_motor_layer=True,
        mind_interval=4,
        motor=MotorLayerConfig(mode="heuristic", smoothing=0.4),
        fly_id="fly_p9",
    )
    pipeline = LiveMindPipeline(env, config=cfg)
    try:
        result = pipeline.run(max_steps=20)
    finally:
        pipeline.close()
    assert result["architecture"] == "B"
    assert result["motor_ticks"] >= result["mind_ticks"]
    assert result["mind_ticks"] == 5  # steps 0,4,8,12,16
    assert result["last"].get("stm_id") is not None


def test_goal_from_intent() -> None:
    g = goal_from_intent(
        {"forward": 0.8, "turn": -0.2, "height": 0.5, "frequency": 0.6, "proboscis": 0.1}
    )
    assert g.shape == (5,)
    assert abs(float(g[0]) - 0.8) < 1e-5


def test_goal_from_intent_prefers_chosen_action() -> None:
    """Architecture B: motor goals follow the L1/L2 plan, not a timid composer blend."""
    timid = np.asarray([0.06, -0.1, 0.5, 0.5, 0.0], dtype=np.float32)
    plan = [0.45, -0.85, 0.5, 0.65, 0.0]
    g = goal_from_intent(
        {
            "forward": 0.06,
            "turn": -0.1,
            "action": timid.tolist(),
            "future": {"chosen_name": "turn_left", "chosen_action": plan},
        },
        action=timid,
    )
    assert abs(float(g[0]) - 0.45) < 1e-5
    assert abs(float(g[1]) - (-0.85)) < 1e-5


def test_motor_locomotion_floor_enforces_walk() -> None:
    from flybrain.motor import ensure_locomotion_floor

    spun = ensure_locomotion_floor(np.asarray([0.05, -0.8, 0.5, 0.5, 0.0], dtype=np.float32))
    assert float(spun[0]) >= 0.4
    stopped = ensure_locomotion_floor(np.asarray([0.0, 0.0, 0.5, 0.3, 0.0], dtype=np.float32))
    assert abs(float(stopped[0])) < 1e-5
    # Near-still explore/inspect leftovers must walk — not hold a mind-interval pause.
    near_still = ensure_locomotion_floor(
        np.asarray([-0.03, -0.00, 0.5, 0.5, 0.0], dtype=np.float32)
    )
    assert float(near_still[0]) >= 0.4
    near_still_turn = ensure_locomotion_floor(
        np.asarray([0.02, -0.05, 0.5, 0.5, 0.0], dtype=np.float32)
    )
    assert float(near_still_turn[0]) >= 0.4


def test_motor_layer_floors_near_still_act_anything_goal() -> None:
    layer = MotorLayer(MotorLayerConfig(mode="heuristic", min_forward=0.4))
    layer.set_goal(np.asarray([-0.03, 0.0, 0.5, 0.5, 0.0], dtype=np.float32))
    assert float(layer.goal[0]) >= 0.4
    # Exact stop (still/idle) still allowed.
    layer.set_goal(np.asarray([0.0, 0.0, 0.5, 0.5, 0.0], dtype=np.float32))
    assert abs(float(layer.goal[0])) < 1e-5
    assert abs(float(layer.goal[1])) < 1e-5


def test_calm_layer1_prefers_locomotion() -> None:
    from flybrain.thinking.two_layer_future import layer1_quick_scan
    from flybrain.world_model import LifeExperience

    best, scored, _ = layer1_quick_scan(
        signals={"energy": 0.99, "sugar": 0.0, "bitter": 0.0, "wall_hazard": 0.0},
        affect={"survival_risk": 0.0, "exploration": 0.15, "novelty": 0.15, "altruism": 0.0},
        experience=LifeExperience(),
    )
    assert best.name in ("forward", "explore", "turn_left", "turn_right")
    assert float(best.action[0]) >= 0.4
    assert scored[0].name != "stop"


def test_train_motor_policy_smoke(tmp_path) -> None:
    ckpt = tmp_path / "motor.pt"
    metrics = train_motor_policy(
        steps=30, batch_episodes=2, episode_len=16, checkpoint_path=ckpt, seed=0
    )
    assert ckpt.exists()
    assert metrics["loss"] >= 0.0
    layer = MotorLayer(MotorLayerConfig(mode="learned", checkpoint_path=str(ckpt)))
    action = layer.act(np.zeros(197, dtype=np.float32))
    assert action.shape == (5,)
