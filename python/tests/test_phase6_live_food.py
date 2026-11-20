from __future__ import annotations

from flybrain.env_food_search import FakeFoodSearchEnv
from flybrain.phase6 import MIN_SUGAR_DISTANCE_REDUCTION
from flybrain.pipeline.live_mind import LiveMindConfig, LiveMindPipeline


def test_live_mind_food_search_smoke() -> None:
    env = FakeFoodSearchEnv()
    pipeline = LiveMindPipeline(
        env,
        config=LiveMindConfig(
            backend="in_memory",
            max_steps=60,
            llm_mock=True,
            use_vision_head=False,
            fly_id="fly_food",
        ),
    )
    try:
        result = pipeline.run(max_steps=60)
    finally:
        pipeline.close()

    last = result["last"]
    assert last.get("stm_id") is not None
    assert last.get("has_food_xyz_in_obs") is False
    improved = result["end_distance"] <= result["start_distance"] - MIN_SUGAR_DISTANCE_REDUCTION
    fed = result.get("sugar_consumed", 0.0) > 0.0
    assert improved or fed or last.get("energy", 0) > 0.2
