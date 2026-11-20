"""Deferred interest points + acknowledge → investigate."""

from __future__ import annotations

import numpy as np

from flybrain.memory.pgvector import InMemoryShortTermMemory
from flybrain.thinking.interest import (
    InterestConfig,
    InterestPointStore,
    apply_investigation_goal,
)
from flybrain.thinking.judgement import DevilProposal, JudgementConfig, JudgementModule
from flybrain.thinking.personality import PersonalityTraits
from flybrain.thinking.retrospective import (
    OutcomeMemory,
    RetrospectiveConfig,
    can_investigate_interest,
    is_low_load,
    run_retrospective,
)
from flybrain.world_model.outcomes import LifeExperience


def test_acknowledge_merges_similar_pending() -> None:
    store = InterestPointStore(InterestConfig(enabled=True))
    a = store.acknowledge(
        stm_id=1, step=0, tick=0, kind="wall", bearing=0.2, novelty=0.5, reason="wall A"
    )
    b = store.acknowledge(
        stm_id=2, step=10, tick=10, kind="wall", bearing=0.25, novelty=0.6, reason="wall B"
    )
    assert a is not None and b is not None
    assert a.id == b.id
    assert b.hits == 2
    assert len(store.pending()) == 1


def test_frame_similarity_identical_and_different() -> None:
    from flybrain.thinking.interest import frame_similarity

    a = np.zeros((16, 16), dtype=np.float32)
    b = np.ones((16, 16), dtype=np.float32)
    assert frame_similarity(a, a) == 1.0
    assert frame_similarity(a, b) == 0.0
    assert 0.4 < frame_similarity(a, a * 0.0 + 0.2) < 0.9


def test_acknowledge_stores_eye_snapshots() -> None:
    store = InterestPointStore(InterestConfig(enabled=True, eye_store_size=8))
    left = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    right = np.linspace(1, 0, 64, dtype=np.float32).reshape(8, 8)
    point = store.acknowledge(
        stm_id=1,
        step=0,
        tick=0,
        kind="wall",
        bearing=0.1,
        novelty=0.5,
        reason="wall",
        left_eye=left,
        right_eye=right,
        stimuli_signature={"wall_hazard": 0.8, "energy": 0.9},
    )
    assert point is not None
    assert point.left_eye is not None and point.left_eye.shape == (8, 8)
    assert point.right_eye is not None
    assert point.to_dict()["has_eyes"] is True
    score, matched = store.max_eye_similarity(
        left, right, kind="wall", stimuli_signature={"wall_hazard": 0.82, "energy": 0.88}
    )
    assert matched is not None and matched.id == point.id
    assert score > 0.9


def test_eye_match_blocks_reacknowledge() -> None:
    """Same eyes as a stored acknowledge → acknowledge score forced to 0."""
    judge = JudgementModule(JudgementConfig(enabled=True, acknowledge_eye_match_max=0.4))
    traits = PersonalityTraits(
        risk_appetite=0.45,
        curiosity_to_test=0.75,
        playfulness=0.5,
        wisdom=0.55,
        intelligence=0.5,
    )
    prop = DevilProposal(
        risk="wall",
        forward_scale=0.4,
        turn_delta=-0.5,
        confidence=0.8,
        reason="wall close ahead",
        ran=True,
        source="llm",
    )
    signals = {"wall_hazard": 0.81, "survival_risk": 0.25, "visual_novelty": 0.0}
    open_ack = judge.evaluate(prop, traits=traits, signals=signals, eye_match_score=0.0)
    assert open_ack.decision == "acknowledge"
    blocked = judge.evaluate(prop, traits=traits, signals=signals, eye_match_score=0.85)
    assert blocked.decision != "acknowledge"
    assert blocked.scores["acknowledge"] == 0.0
    assert blocked.scores["eye_match"] == 0.85
    assert blocked.reason == "acknowledge_eye_match" or blocked.decision in ("reject", "ignore")


def test_pop_and_apply_investigation_goal() -> None:
    store = InterestPointStore(
        InterestConfig(enabled=True, investigate_ttl_ticks=5, investigate_forward_scale=0.2)
    )
    store.acknowledge(
        stm_id=3, step=0, tick=0, kind="novelty", bearing=-0.8, novelty=0.7, reason="flash"
    )
    dirs = store.pop_for_investigation(tick=1)
    assert len(dirs) == 1
    d = dirs[0]
    assert d.point.status == "investigating"
    assert d.turn < 0
    goal = apply_investigation_goal(np.asarray([0.9, 0.0, 0.5, 0.5, 0.0], dtype=np.float32), d)
    assert goal[0] < 0.3
    assert goal[1] < 0
    # TTL does not burn on started tick.
    assert store.advance_tick(1) is not None
    assert store.active is not None and store.active.remaining == 5
    for t in range(2, 7):
        store.advance_tick(t)
    assert store.active is None
    assert d.point.status == "done"


def test_curious_loaded_acknowledges_wall() -> None:
    """High curiosity + high wall load → acknowledge (save for later) over reject."""
    judge = JudgementModule(JudgementConfig(enabled=True))
    traits = PersonalityTraits(
        risk_appetite=0.45,
        curiosity_to_test=0.75,
        playfulness=0.5,
        wisdom=0.55,
        intelligence=0.5,
    )
    prop = DevilProposal(
        risk="wall",
        forward_scale=0.4,
        turn_delta=-0.5,
        confidence=0.8,
        reason="wall close ahead",
        ran=True,
        source="llm",
    )
    result = judge.evaluate(
        prop,
        traits=traits,
        signals={"wall_hazard": 0.81, "survival_risk": 0.25, "visual_novelty": 0.0},
    )
    assert result.decision == "acknowledge"
    assert result.blend_weight == 0.0
    assert result.scores["acknowledge"] >= result.scores["reject"]


def test_risk_averse_low_curiosity_still_rejects() -> None:
    judge = JudgementModule(JudgementConfig(enabled=True))
    traits = PersonalityTraits(
        risk_appetite=0.05, curiosity_to_test=0.15, playfulness=0.1, wisdom=0.2, intelligence=0.5
    )
    prop = DevilProposal(
        risk="wall", forward_scale=0.4, turn_delta=0.3, confidence=0.8, reason="wall", ran=True
    )
    result = judge.evaluate(prop, traits=traits, signals={"wall_hazard": 0.8, "survival_risk": 0.2})
    assert result.decision == "reject"


def test_can_investigate_interest_with_high_wall() -> None:
    cfg = RetrospectiveConfig()
    assert not is_low_load(
        signals={"wall_hazard": 0.8, "bitter": 0.0},
        affect={"survival_risk": 0.2},
        config=cfg,
    )
    assert can_investigate_interest(
        signals={"wall_hazard": 0.8, "bitter": 0.0},
        affect={"survival_risk": 0.2},
        config=cfg,
    )


def test_retrospective_investigates_pending_interest() -> None:
    memory = InMemoryShortTermMemory(dim=8)
    # Seed two STM rows so labeling path could also run if low-load; wall keeps labeling off.
    for step in (0, 1):
        memory.insert_embedding(
            embedding=np.zeros(8, dtype=np.float32),
            signals={"wall_hazard": 0.8, "bitter": 0.0, "energy": 0.9},
            affect={"survival_risk": 0.2},
            episode_id="e",
            step=step,
            sim_time=float(step),
            fly_id="fly_001",
        )
    store = InterestPointStore(InterestConfig(enabled=True))
    store.acknowledge(
        stm_id=1,
        step=1,
        tick=1,
        kind="wall",
        bearing=0.4,
        novelty=0.5,
        reason="parked wall",
    )
    result = run_retrospective(
        memory=memory,
        experience=LifeExperience(),
        outcome_memory=OutcomeMemory(),
        fly_id="fly_001",
        signals={"wall_hazard": 0.8, "bitter": 0.0},
        affect={"survival_risk": 0.2},
        config=RetrospectiveConfig(investigate_interest=True),
        interest_store=store,
        tick=2,
    )
    assert result.ran
    assert result.reason == "ok_investigate"
    assert len(result.investigations) == 1
    assert result.thought is not None
    assert "Investigate interest#" in (result.thought.refined_thought or "")
    assert store.active is not None
