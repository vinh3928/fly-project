from __future__ import annotations

from flybrain.pipeline.live_mind import LiveMindConfig, LiveMindPipeline, load_mind_config, make_live_env


def test_load_mind_config() -> None:
    cfg = load_mind_config()
    assert "database" in cfg
    assert cfg["embedding"]["total_dim"] == 272


def test_live_mind_fake_smoke() -> None:
    env = make_live_env(fake=True)
    pipeline = LiveMindPipeline(
        env,
        config=LiveMindConfig(backend="in_memory", max_steps=3, llm_mock=True, fly_id="fly_test"),
    )
    try:
        result = pipeline.run(max_steps=3)
    finally:
        pipeline.close()

    assert result["steps"] == 3
    last = result["last"]
    assert last["stm_id"] is not None
    assert last["thought_id"] is not None
    assert last["devil_critique"]
    assert len(last["intent"]["action"]) == 5
    assert "joy" in last["affect"]
