"""Always-on thinking service over short-term memory embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from flybrain.memory.pgvector import ShortTermMemory, StmRecord, ThoughtRecord
from flybrain.thinking.composer import ThoughtResult, compose_thought
from flybrain.thinking.devils_advocate import LlmClient, critique_draft, make_llm_client
from flybrain.thinking.future_stub import present_future_stub
from flybrain.thinking.reflection import build_reflection


@dataclass
class ThinkingConfig:
    fly_id: str = "fly_001"
    top_k_similar: int = 5
    recent_k: int = 8
    action_clamps: dict[str, float] | None = None
    llm_mock: bool = True
    llm_kwargs: dict[str, Any] | None = None


class ThinkingService:
    """Consume an STM embedding and produce a refined thought + motor intent."""

    def __init__(
        self,
        memory: ShortTermMemory,
        config: ThinkingConfig | None = None,
        llm: LlmClient | None = None,
    ) -> None:
        self.memory = memory
        self.config = config or ThinkingConfig()
        self.llm = llm or make_llm_client(
            mock=self.config.llm_mock, **(self.config.llm_kwargs or {})
        )

    def think_on_record(self, stm: StmRecord) -> tuple[ThoughtResult, ThoughtRecord]:
        reflection, _, _ = build_reflection(
            self.memory,
            embedding=stm.embedding,
            fly_id=stm.fly_id,
            top_k=self.config.top_k_similar,
            recent_k=self.config.recent_k,
        )
        present_future = present_future_stub(signals=stm.signals, affect=stm.affect)
        draft = (
            f"Considering step={stm.step}: sugar={stm.signals.get('sugar')} "
            f"bitter={stm.signals.get('bitter')} survival={stm.affect.get('survival_risk')}"
        )
        critique = critique_draft(
            self.llm, draft=draft, affect=stm.affect, signals=stm.signals
        )
        result = compose_thought(
            signals=stm.signals,
            affect=stm.affect,
            reflection=reflection,
            present_future=present_future,
            devil_critique=critique,
            action_clamps=self.config.action_clamps,
        )
        thought = self.memory.insert_thought(
            stm_id=stm.id,
            reflection=result.reflection,
            devil_critique=result.devil_critique,
            refined_thought=result.refined_thought,
            intent=result.intent,
        )
        return result, thought

    def think_on_embedding(
        self,
        *,
        embedding: np.ndarray,
        signals: dict[str, Any],
        affect: dict[str, Any],
        episode_id: str = "live",
        step: int = 0,
        sim_time: float = 0.0,
        fly_id: str | None = None,
    ) -> tuple[StmRecord, ThoughtResult, ThoughtRecord]:
        fly = fly_id or self.config.fly_id
        stm = self.memory.insert_embedding(
            embedding=embedding,
            signals=signals,
            affect=affect,
            episode_id=episode_id,
            step=step,
            sim_time=sim_time,
            fly_id=fly,
        )
        result, thought = self.think_on_record(stm)
        return stm, result, thought
