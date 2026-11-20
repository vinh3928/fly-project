from __future__ import annotations

import numpy as np

from flybrain.pipeline.live_mind import (
    LiveMindConfig,
    LiveMindPipeline,
    load_mind_config,
    make_live_env,
)
from flybrain.thinking.interest import InterestConfig
from flybrain.thinking.judgement import DevilProposal, JudgementConfig


def test_load_mind_config() -> None:
    cfg = load_mind_config()
    assert "database" in cfg
    assert cfg["embedding"]["total_dim"] == 272
    assert "personality" in cfg
    assert "judgement" in cfg


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
    assert "judgement" in last
    assert "personality" in last
    assert "devil_proposal" in last.get("intent", {}).get("future", {}) or "devil_proposal" in last


def test_apply_judgement_accept_blends_l1_goal() -> None:
    env = make_live_env(fake=True)
    pipeline = LiveMindPipeline(
        env,
        config=LiveMindConfig(
            backend="in_memory",
            max_steps=1,
            llm_mock=True,
            fly_id="fly_judge",
            judgement=JudgementConfig(enabled=True, base_blend_weight=0.5, max_blend_weight=0.8),
        ),
    )
    try:
        pipeline.personality.set_traits(
            {
                "risk_appetite": 0.85,
                "curiosity_to_test": 0.9,
                "playfulness": 0.7,
                "wisdom": 0.1,
                "intelligence": 0.4,
            },
            source="test",
        )
        l1 = np.asarray([0.75, 0.20, 0.5, 0.5, 0.0], dtype=np.float32)
        proposal = DevilProposal(
            risk="wall",
            forward_scale=0.4,
            turn_delta=-0.5,
            confidence=0.85,
            reason="wall ahead",
            ran=True,
            source="test",
        )
        pipeline.thinking._last_devil_proposal = proposal
        info: dict = {}
        out = pipeline._apply_judgement(
            l1,
            info=info,
            signals={"wall_hazard": 0.3},
            affect={"survival_risk": 0.1},
        )
        assert info["judgement"]["decision"] == "accept"
        assert float(out[0]) < float(l1[0])
        assert float(out[1]) < float(l1[1])
    finally:
        pipeline.close()


def test_apply_judgement_reject_leaves_l1_unchanged() -> None:
    env = make_live_env(fake=True)
    pipeline = LiveMindPipeline(
        env,
        config=LiveMindConfig(
            backend="in_memory",
            max_steps=1,
            llm_mock=True,
            fly_id="fly_judge",
            judgement=JudgementConfig(enabled=True),
        ),
    )
    try:
        pipeline.personality.set_traits(
            {
                "risk_appetite": 0.05,
                "curiosity_to_test": 0.1,
                "playfulness": 0.1,
                "wisdom": 0.5,
                "intelligence": 0.5,
            },
            source="test",
        )
        l1 = np.asarray([0.75, 0.20, 0.5, 0.5, 0.0], dtype=np.float32)
        proposal = DevilProposal(
            risk="wall",
            forward_scale=0.4,
            turn_delta=0.3,
            confidence=0.8,
            reason="wall ahead",
            ran=True,
            source="test",
        )
        pipeline.thinking._last_devil_proposal = proposal
        info: dict = {}
        out = pipeline._apply_judgement(
            l1,
            info=info,
            signals={"wall_hazard": 0.8},
            affect={"survival_risk": 0.2},
        )
        assert info["judgement"]["decision"] == "reject"
        assert np.allclose(out, l1)
    finally:
        pipeline.close()


def test_apply_judgement_acknowledge_enqueues_interest() -> None:
    env = make_live_env(fake=True)
    pipeline = LiveMindPipeline(
        env,
        config=LiveMindConfig(
            backend="in_memory",
            max_steps=5,
            llm_mock=True,
            fly_id="fly_ack",
            judgement=JudgementConfig(enabled=True),
            interest=InterestConfig(enabled=True),
            enable_retrospective=True,
            retrospective={
                "survival_risk_max": 0.45,
                "bitter_max": 0.35,
                "wall_hazard_max": 0.45,
                "investigate_interest": True,
            },
        ),
    )
    try:
        pipeline.personality.set_traits(
            {
                "risk_appetite": 0.45,
                "curiosity_to_test": 0.75,
                "playfulness": 0.5,
                "wisdom": 0.55,
                "intelligence": 0.5,
            },
            source="test",
        )
        l1 = np.asarray([0.75, 0.20, 0.5, 0.5, 0.0], dtype=np.float32)
        proposal = DevilProposal(
            risk="wall",
            forward_scale=0.4,
            turn_delta=-0.5,
            confidence=0.8,
            reason="wall close ahead",
            ran=True,
            source="test",
        )
        pipeline.thinking._last_devil_proposal = proposal
        info: dict = {"stm_id": 7}
        out = pipeline._apply_judgement(
            l1,
            info=info,
            signals={"wall_hazard": 0.81, "bitter": 0.0, "visual_novelty": 0.0},
            affect={"survival_risk": 0.2},
        )
        assert info["judgement"]["decision"] == "acknowledge"
        assert np.allclose(out, l1)
        assert info.get("interest_point") is not None
        # Soft investigate gate allows wall revisit while survival is calm.
        assert info.get("investigation") is not None or len(pipeline.interests.pending()) >= 1
    finally:
        pipeline.close()
