"""Trait-weighted judgement of devil proposals."""

from __future__ import annotations

import numpy as np

from flybrain.thinking.judgement import (
    DevilProposal,
    JudgementConfig,
    JudgementModule,
    apply_proposal,
    blend_goals,
)
from flybrain.thinking.personality import PersonalityTraits


def _wall_slow() -> DevilProposal:
    return DevilProposal(
        risk="wall",
        forward_scale=0.4,
        turn_delta=0.3,
        confidence=0.8,
        reason="wall ahead",
        ran=True,
        source="heuristic",
    )


def test_curious_confident_accepts() -> None:
    judge = JudgementModule(JudgementConfig(enabled=True))
    traits = PersonalityTraits(
        risk_appetite=0.7, curiosity_to_test=0.9, playfulness=0.7, wisdom=0.2, intelligence=0.4
    )
    result = judge.evaluate(
        _wall_slow(), traits=traits, signals={"wall_hazard": 0.3, "survival_risk": 0.1}
    )
    assert result.decision == "accept"
    assert result.blend_weight > 0.0


def test_risk_averse_rejects_wall() -> None:
    judge = JudgementModule(JudgementConfig(enabled=True))
    traits = PersonalityTraits(
        risk_appetite=0.05, curiosity_to_test=0.2, playfulness=0.1, wisdom=0.2, intelligence=0.5
    )
    result = judge.evaluate(
        _wall_slow(), traits=traits, signals={"wall_hazard": 0.8, "survival_risk": 0.2}
    )
    assert result.decision == "reject"


def test_weak_proposal_ignored_by_wisdom() -> None:
    judge = JudgementModule(JudgementConfig(enabled=True, ignore_weak_confidence=0.25))
    traits = PersonalityTraits(wisdom=0.9, curiosity_to_test=0.1, risk_appetite=0.5)
    weak = DevilProposal(
        risk="none", forward_scale=1.0, turn_delta=0.0, confidence=0.05, reason="", ran=True
    )
    result = judge.evaluate(weak, traits=traits, signals={"wall_hazard": 0.2})
    assert result.decision == "ignore"
    assert result.blend_weight == 0.0


def test_reconsider_budget_forces_ignore() -> None:
    judge = JudgementModule(JudgementConfig(enabled=True, reconsider_budget=1))
    # Inconsistent: high forward into wall with risk=none → intelligence wants reconsider.
    traits = PersonalityTraits(intelligence=0.95, curiosity_to_test=0.8, wisdom=0.1, risk_appetite=0.5)
    bad = DevilProposal(
        risk="none", forward_scale=0.95, turn_delta=0.0, confidence=0.3, reason="go", ran=True
    )
    first = judge.evaluate(bad, traits=traits, signals={"wall_hazard": 0.8}, reconsider_used=0)
    assert first.decision == "reconsider"
    second = judge.evaluate(bad, traits=traits, signals={"wall_hazard": 0.8}, reconsider_used=1)
    assert second.decision == "ignore"


def test_reject_memory_biases_similar_proposals() -> None:
    judge = JudgementModule(JudgementConfig(enabled=True, reject_memory_ticks=100))
    judge.advance_tick(10)
    traits = PersonalityTraits(
        risk_appetite=0.55, curiosity_to_test=0.5, playfulness=0.5, wisdom=0.2, intelligence=0.4
    )
    prop = _wall_slow()
    judge.record_reject(prop)
    result = judge.evaluate(prop, traits=traits, signals={"wall_hazard": 0.4, "survival_risk": 0.1})
    assert result.scores["reject"] > 0.3


def test_same_proposal_flips_after_trait_adjust() -> None:
    """Live traits: judgement rules stable, personality drift changes the decision."""
    judge = JudgementModule(JudgementConfig(enabled=True))
    prop = _wall_slow()
    seeking = PersonalityTraits(
        risk_appetite=0.85, curiosity_to_test=0.8, playfulness=0.6, wisdom=0.2, intelligence=0.4
    )
    averse = PersonalityTraits(
        risk_appetite=0.05, curiosity_to_test=0.15, playfulness=0.1, wisdom=0.5, intelligence=0.5
    )
    signals = {"wall_hazard": 0.5, "survival_risk": 0.15}
    assert judge.evaluate(prop, traits=seeking, signals=signals).decision == "accept"
    assert judge.evaluate(prop, traits=averse, signals=signals).decision == "reject"


def test_apply_and_blend_goals() -> None:
    l1 = np.asarray([0.75, 0.2, 0.5, 0.5, 0.0], dtype=np.float32)
    prop = DevilProposal(risk="wall", forward_scale=0.4, turn_delta=-0.5, confidence=0.8, ran=True)
    proposed = apply_proposal(l1, prop)
    assert float(proposed[0]) < float(l1[0])
    assert float(proposed[1]) < float(l1[1])
    blended = blend_goals(l1, proposed, 0.5)
    assert float(l1[0]) > float(blended[0]) > float(proposed[0])
