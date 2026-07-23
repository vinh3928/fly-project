"""Phase 7: two-layer future, outcomes, retrospective, world-model training."""

from __future__ import annotations

import numpy as np
import torch

from flybrain.memory import InMemoryShortTermMemory
from flybrain.phase4 import HIGH_LEVEL_ACTION_DIM, TOTAL_EMBEDDING_DIM
from flybrain.thinking import (
    OutcomeMemory,
    ThinkingConfig,
    ThinkingService,
    is_low_load,
    predict_present_future,
    run_retrospective,
)
from flybrain.thinking.two_layer_future import layer1_quick_scan, layer2_reflect_outcomes
from flybrain.training.train_world_model import train_world_model
from flybrain.world_model import (
    LifeExperience,
    NextStateWorldModel,
    label_transition,
    spiritual_growth_score,
)


def test_life_experience_beginner_vs_advanced() -> None:
    young = LifeExperience(lifetime_steps=0, labeled_outcomes=0, age_horizon=100, data_horizon=50)
    old = LifeExperience(
        lifetime_steps=500,
        labeled_outcomes=200,
        age_horizon=100,
        data_horizon=50,
        max_altruism=0.35,
        max_spiritual=0.35,
    )
    w_young = young.priority_weights(survival_risk=0.1)
    w_old = old.priority_weights(survival_risk=0.1)
    assert w_young.survival > 0.9
    assert w_young.altruism < 0.05
    assert w_young.spiritual < 0.05
    assert old.maturity() > young.maturity()
    assert w_old.altruism + w_old.spiritual > w_young.altruism + w_young.spiritual
    # Under threat, survival stays elevated even when mature
    w_threat = old.priority_weights(survival_risk=0.95)
    assert w_threat.survival >= old.min_survival


def test_spiritual_growth_and_labeling() -> None:
    prev_a = {
        "comfort": 0.2,
        "trust": 0.2,
        "altruism": 0.1,
        "exploration": 0.2,
        "fear": 0.5,
        "pain": 0.4,
        "survival_risk": 0.6,
        "novelty": 0.2,
    }
    next_a = {
        "comfort": 0.6,
        "trust": 0.5,
        "altruism": 0.4,
        "exploration": 0.5,
        "fear": 0.1,
        "pain": 0.05,
        "survival_risk": 0.2,
        "novelty": 0.7,
    }
    g = spiritual_growth_score(affect=next_a, prev_affect=prev_a, learning_progress=0.3)
    assert 0.0 <= g <= 1.0
    good = label_transition(
        prev_signals={"energy": 0.3, "sugar": 0.1, "bitter": 0.0},
        prev_affect=prev_a,
        next_signals={"energy": 0.7, "sugar": 0.5, "bitter": 0.0, "sugar_consumed": 1.0},
        next_affect=next_a,
    )
    assert good.kind == "desirable"
    bad = label_transition(
        prev_signals={"energy": 0.7, "sugar": 0.0, "bitter": 0.0},
        prev_affect={"survival_risk": 0.2, "comfort": 0.5, "fear": 0.1, "pain": 0.0},
        next_signals={"energy": 0.4, "sugar": 0.0, "bitter": 0.9, "damage": 0.6},
        next_affect={"survival_risk": 0.8, "comfort": 0.1, "fear": 0.7, "pain": 0.6},
    )
    assert bad.kind == "undesirable"


def test_layer1_prefers_improving_action_young_survival() -> None:
    exp = LifeExperience(lifetime_steps=0, labeled_outcomes=0)
    best, scored, weights = layer1_quick_scan(
        signals={"energy": 0.25, "sugar": 0.7, "bitter": 0.0, "wall_hazard": 0.0},
        affect={"survival_risk": 0.7, "altruism": 0.1, "exploration": 0.2, "novelty": 0.1},
        experience=exp,
    )
    assert weights.survival > 0.85
    assert best.action.shape[0] == HIGH_LEVEL_ACTION_DIM
    # Feeding / approaching sugar should rank well when hungry/high survival
    names = [s.name for s in scored[:3]]
    assert "feed" in names or "forward" in names or best.improves


def test_layer1_advanced_weights_move_ranking() -> None:
    young = LifeExperience(lifetime_steps=0, labeled_outcomes=0)
    advanced = LifeExperience(
        lifetime_steps=2000,
        labeled_outcomes=500,
        age_horizon=100,
        data_horizon=50,
        max_altruism=0.4,
        max_spiritual=0.4,
    )
    signals = {"energy": 0.8, "sugar": 0.1, "bitter": 0.0, "wall_hazard": 0.0}
    affect = {
        "survival_risk": 0.15,
        "altruism": 0.5,
        "exploration": 0.7,
        "novelty": 0.7,
        "comfort": 0.6,
        "trust": 0.5,
        "fear": 0.05,
        "pain": 0.0,
    }
    _, scored_y, w_y = layer1_quick_scan(signals=signals, affect=affect, experience=young)
    _, scored_a, w_a = layer1_quick_scan(signals=signals, affect=affect, experience=advanced)
    assert w_a.spiritual + w_a.altruism > w_y.spiritual + w_y.altruism
    # Values differ under advanced weights for at least one candidate
    assert any(
        abs(a.value - b.value) > 1e-4 for a, b in zip(scored_y, scored_a, strict=False)
    ) or w_a.spiritual > 0.05


def test_layer2_vetoes_undesirable_history() -> None:
    exp = LifeExperience(lifetime_steps=100, labeled_outcomes=50, age_horizon=50, data_horizon=20)
    best, scored, _ = layer1_quick_scan(
        signals={"energy": 0.6, "sugar": 0.2, "bitter": 0.5, "wall_hazard": 0.0},
        affect={"survival_risk": 0.4, "exploration": 0.3, "altruism": 0.2},
        experience=exp,
    )
    past = [
        {"action_name": best.name, "kind": "undesirable", "score": -0.5},
        {"action_name": "stop", "kind": "desirable", "score": 0.4},
        {"action_name": "turn_left", "kind": "desirable", "score": 0.3},
    ]
    chosen, overridden, reason = layer2_reflect_outcomes(
        layer1=best,
        scored=scored,
        past_outcomes=past,
        signals={"energy": 0.6, "sugar": 0.2, "bitter": 0.5},
        affect={"survival_risk": 0.4},
    )
    assert overridden
    assert chosen.name != best.name
    assert "veto" in reason or "prefer" in reason


def test_low_load_gate() -> None:
    assert is_low_load(
        signals={"bitter": 0.0, "wall_hazard": 0.0},
        affect={"survival_risk": 0.2},
    )
    assert not is_low_load(
        signals={"bitter": 0.0, "wall_hazard": 0.0},
        affect={"survival_risk": 0.9},
    )


def test_retrospective_labels_and_skips_high_load() -> None:
    mem = InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM)
    exp = LifeExperience(age_horizon=10, data_horizon=5)
    outcomes = OutcomeMemory()
    fly = "fly_r"
    for i in range(5):
        mem.insert_embedding(
            embedding=np.random.randn(TOTAL_EMBEDDING_DIM).astype(np.float32),
            signals={"energy": 0.4 + 0.05 * i, "sugar": 0.2, "bitter": 0.0},
            affect={"survival_risk": 0.2, "comfort": 0.3 + 0.05 * i, "fear": 0.1},
            fly_id=fly,
            step=i,
        )
    blocked = run_retrospective(
        memory=mem,
        experience=exp,
        outcome_memory=outcomes,
        fly_id=fly,
        signals={"bitter": 0.0, "wall_hazard": 0.0},
        affect={"survival_risk": 0.9},
    )
    assert not blocked.ran
    assert blocked.reason == "high_load"

    labeled_before = exp.labeled_outcomes
    ok = run_retrospective(
        memory=mem,
        experience=exp,
        outcome_memory=outcomes,
        fly_id=fly,
        signals={"bitter": 0.0, "wall_hazard": 0.0},
        affect={"survival_risk": 0.2},
        recent_actions=[{"stm_id": 1, "step": 0, "action_name": "forward", "action": [0.5, 0, 0.5, 0.5, 0]}],
    )
    assert ok.ran
    assert exp.labeled_outcomes > labeled_before
    assert outcomes.all()
    mem.close()


def test_predict_present_future_structured() -> None:
    pred = predict_present_future(
        signals={"energy": 0.3, "sugar": 0.6, "bitter": 0.0},
        affect={"survival_risk": 0.55, "exploration": 0.4, "altruism": 0.1},
        experience=LifeExperience(lifetime_steps=10, labeled_outcomes=2),
    )
    assert "Present" in pred.present
    assert "Future" in pred.future_hypothesis
    assert pred.chosen_action.shape[0] == HIGH_LEVEL_ACTION_DIM
    assert pred.priority_weights["survival"] > 0


def test_thinking_service_two_layer_and_retrospective() -> None:
    mem = InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM)
    service = ThinkingService(
        mem,
        ThinkingConfig(
            llm_mock=True,
            fly_id="fly_001",
            enable_retrospective=True,
            priorities={"age_horizon": 20, "data_horizon": 5},
            retrospective={"survival_risk_max": 0.5, "batch_size": 4},
        ),
    )
    emb = np.random.randn(TOTAL_EMBEDDING_DIM).astype(np.float32)
    stm, thought, thought_rec = service.think_on_embedding(
        embedding=emb,
        signals={"sugar": 0.5, "bitter": 0.05, "energy": 0.6, "wall_hazard": 0.0},
        affect={
            "joy": 0.5,
            "anger": 0.1,
            "survival_risk": 0.2,
            "exploration": 0.5,
            "altruism": 0.2,
            "fear": 0.1,
            "comfort": 0.4,
            "novelty": 0.3,
        },
        step=1,
    )
    assert stm.id >= 1
    assert thought.action.shape == (HIGH_LEVEL_ACTION_DIM,)
    assert "future" in thought.intent
    assert thought_rec.stm_id == stm.id
    # Second step builds STM for retrospective pairing
    service.think_on_embedding(
        embedding=np.random.randn(TOTAL_EMBEDDING_DIM).astype(np.float32),
        signals={"sugar": 0.55, "bitter": 0.0, "energy": 0.65, "wall_hazard": 0.0},
        affect={
            "joy": 0.55,
            "survival_risk": 0.15,
            "exploration": 0.5,
            "altruism": 0.25,
            "comfort": 0.5,
            "fear": 0.05,
            "novelty": 0.35,
        },
        step=2,
    )
    assert service.experience.lifetime_steps >= 2
    mem.close()


def test_world_model_rollout_and_train_improves() -> None:
    model = NextStateWorldModel(latent_dim=16, action_dim=5, hidden_dim=32)
    z = torch.randn(2, 16)
    actions = torch.randn(2, 3, 5)
    out = model.rollout(z, actions, horizon=3)
    assert out.shape == (2, 16)
    metrics = train_world_model(
        steps=40,
        batch_size=32,
        latent_dim=32,
        action_dim=5,
        hidden_dim=64,
        seed=1,
    )
    assert metrics["eval_loss"] <= metrics["baseline_loss"] + 1e-5
