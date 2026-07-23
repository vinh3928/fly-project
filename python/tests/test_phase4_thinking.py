from __future__ import annotations

import numpy as np

from flybrain.memory import InMemoryShortTermMemory
from flybrain.phase4 import HIGH_LEVEL_ACTION_DIM, TOTAL_EMBEDDING_DIM
from flybrain.thinking import (
    HeuristicDevilClient,
    ThinkingConfig,
    ThinkingService,
    compose_thought,
    critique_draft,
    present_future_stub,
)


def test_present_future_stub_survival() -> None:
    out = present_future_stub(
        signals={"energy": 0.1, "bitter": 0.0, "sugar": 0.0},
        affect={"survival_risk": 0.9, "exploration": 0.1},
    )
    assert "Present" in out["present"]
    assert "Future" in out["future_hypothesis"]
    assert "energy" in out["future_hypothesis"].lower() or "reduce" in out["future_hypothesis"].lower()


def test_composer_clamps_and_survival() -> None:
    result = compose_thought(
        signals={"sugar": 0.8, "bitter": 0.0},
        affect={"survival_risk": 0.9, "exploration": 0.8, "joy": 0.7, "anger": 0.0, "altruism": 0.1},
        reflection="none",
        present_future={"present": "p", "future_hypothesis": "f"},
        devil_critique="Counter: forward drive without upright stability risks falling.",
        action_clamps={"survival_forward_scale": 0.25},
    )
    assert result.action.shape == (HIGH_LEVEL_ACTION_DIM,)
    assert -1.0 <= result.action[0] <= 1.0
    assert result.devil_critique
    assert "fall" in result.refined_thought.lower() or "critique" in result.refined_thought.lower()


def test_devils_advocate_heuristic() -> None:
    text = critique_draft(
        HeuristicDevilClient(),
        draft="Plan forward explore sugar joy",
        affect={"survival_risk": 0.2},
        signals={"sugar": 0.5},
    )
    assert "Counter" in text


def test_thinking_service_end_to_end() -> None:
    mem = InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM)
    service = ThinkingService(mem, ThinkingConfig(llm_mock=True, fly_id="fly_001"))
    emb = np.random.randn(TOTAL_EMBEDDING_DIM).astype(np.float32)
    stm, thought, thought_rec = service.think_on_embedding(
        embedding=emb,
        signals={"sugar": 0.6, "bitter": 0.1, "energy": 0.5},
        affect={
            "joy": 0.6,
            "anger": 0.1,
            "survival_risk": 0.2,
            "exploration": 0.5,
            "altruism": 0.1,
            "fear": 0.1,
        },
        step=3,
    )
    assert stm.id >= 1
    assert thought_rec.stm_id == stm.id
    assert thought.action.shape == (HIGH_LEVEL_ACTION_DIM,)
    assert thought.devil_critique
    assert "intent" in thought_rec.intent or "forward" in thought_rec.intent
    mem.close()
