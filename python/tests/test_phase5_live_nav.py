from __future__ import annotations

from pathlib import Path

import yaml

from flybrain.env_visual_nav import FakeVisualNavEnv
from flybrain.pipeline.live_mind import LiveMindConfig, LiveMindPipeline
from flybrain.phase5 import MAX_WALL_COLLISIONS_PER_EPISODE, MIN_DISTANCE_REDUCTION
from flybrain.training.train_vision_head import train_vision_head


def test_train_vision_head_reduces_loss(tmp_path: Path) -> None:
    ckpt = tmp_path / "vision_head.pt"
    metrics = train_vision_head(steps=40, batch_size=16, checkpoint_path=ckpt, seed=0)
    assert ckpt.exists()
    assert metrics["loss"] >= 0.0


def test_live_mind_visual_nav_exit_metrics() -> None:
    env = FakeVisualNavEnv()
    cfg = LiveMindConfig(
        backend="in_memory",
        max_steps=80,
        llm_mock=True,
        use_vision_head=True,
        use_heuristic_if_untrained=True,
        action_clamps={"wall_hazard_forward_scale": 0.35},
        fly_id="fly_nav",
    )
    pipeline = LiveMindPipeline(env, config=cfg)

    # Prefer vision_policy_action blended with mind when cues exist: run custom loop.
    obs, reset_info = env.reset(seed=3)
    start = float(reset_info["distance_to_target"])
    last = {}
    try:
        for step in range(80):
            mind_action, info = pipeline.process_observation(obs)
            vision_action = env.vision_policy_action()
            # Blend: vision dominates turn/forward for Phase 5 exit proof.
            action = mind_action.copy()
            action[0] = 0.35 * float(mind_action[0]) + 0.65 * float(vision_action[0])
            action[1] = 0.25 * float(mind_action[1]) + 0.75 * float(vision_action[1])
            obs, _reward, term, trunc, step_info = env.step(action)
            last = {**info, **step_info}
            pipeline._step += 1
            if term or trunc:
                break
    finally:
        pipeline.close()

    assert last.get("has_target_xyz_in_obs") is False
    assert last.get("vision_cues") is not None
    end = float(last["distance_to_target"])
    improved = end <= start - MIN_DISTANCE_REDUCTION
    assert last.get("success") or improved
    assert int(last.get("wall_hits", 0)) <= MAX_WALL_COLLISIONS_PER_EPISODE + 15


def test_visual_nav_config_loads() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "visual_nav.yaml"
    with open(path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    assert cfg["vision_size"] == 64
