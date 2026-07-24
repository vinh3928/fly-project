"""Always-on thinking service over short-term memory embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from flybrain.memory.pgvector import ShortTermMemory, StmRecord, ThoughtRecord
from flybrain.thinking.composer import ThoughtResult, compose_thought
from flybrain.thinking.devils_advocate import LlmClient, critique_draft, make_llm_client, propose_critique
from flybrain.thinking.reflection import build_reflection
from flybrain.thinking.retrospective import (
    OutcomeMemory,
    RetrospectiveConfig,
    RetrospectiveResult,
    run_retrospective,
)
from flybrain.thinking.tendency import TendencyConfig, TendencyDrive
from flybrain.thinking.two_layer_future import predict_present_future
from flybrain.world_model.outcomes import LifeExperience
from flybrain.world_model.predictor import NextStateWorldModel


@dataclass
class ThinkingConfig:
    fly_id: str = "fly_001"
    top_k_similar: int = 5
    recent_k: int = 8
    action_clamps: dict[str, float] | None = None
    stimulus_thresholds: dict[str, float] | None = None
    llm_mock: bool = True
    llm_kwargs: dict[str, Any] | None = None
    priorities: dict[str, Any] | None = None
    retrospective: dict[str, Any] | None = None
    enable_retrospective: bool = True
    tendency: dict[str, Any] | None = None


class ThinkingService:
    """Consume an STM embedding and produce a refined thought + motor intent."""

    def __init__(
        self,
        memory: ShortTermMemory,
        config: ThinkingConfig | None = None,
        llm: LlmClient | None = None,
        *,
        experience: LifeExperience | None = None,
        outcome_memory: OutcomeMemory | None = None,
        world_model: NextStateWorldModel | None = None,
        tendency: TendencyDrive | None = None,
    ) -> None:
        self.memory = memory
        self.config = config or ThinkingConfig()
        self.llm = llm or make_llm_client(
            mock=self.config.llm_mock, **(self.config.llm_kwargs or {})
        )
        self.experience = experience or LifeExperience.from_config(self.config.priorities)
        self.outcome_memory = outcome_memory or OutcomeMemory()
        self.world_model = world_model
        self.tendency = tendency or TendencyDrive(
            TendencyConfig.from_dict(self.config.tendency)
        )
        self.retrospective_config = RetrospectiveConfig(
            **{
                k: v
                for k, v in (self.config.retrospective or {}).items()
                if k in RetrospectiveConfig.__dataclass_fields__
            }
        )
        self._recent_actions: list[dict[str, Any]] = []
        self._last_retrospective: RetrospectiveResult | None = None
        self._last_devil_proposal = None
        self.interest_store = None  # set by LiveMindPipeline when available

    def think_on_record(self, stm: StmRecord) -> tuple[ThoughtResult, ThoughtRecord]:
        reflection, _, _ = build_reflection(
            self.memory,
            embedding=stm.embedding,
            fly_id=stm.fly_id,
            top_k=self.config.top_k_similar,
            recent_k=self.config.recent_k,
        )
        future = predict_present_future(
            signals=stm.signals,
            affect=stm.affect,
            experience=self.experience,
            past_outcomes=self.outcome_memory.all(),
            world_model=self.world_model,
            embedding=stm.embedding,
            stimulus_thresholds=self.config.stimulus_thresholds,
            tendency=self.tendency,
        )
        from flybrain.thinking.tendency import TendencyDrive
        from flybrain.thinking.two_layer_future import _stimulus_peaks

        peaks = _stimulus_peaks(stm.signals, stm.affect, self.config.stimulus_thresholds)
        safe_mode = TendencyDrive.is_safe_mode(stm.signals, peaks)
        tend_state = self.tendency.state(
            maturity=self.experience.maturity(),
            safe_mode=safe_mode,
        )
        present_future = future.as_present_future()
        draft = (
            f"Considering step={stm.step}: sugar={stm.signals.get('sugar')} "
            f"bitter={stm.signals.get('bitter')} survival={stm.affect.get('survival_risk')} "
            f"L1={future.chosen_name} tendency={tend_state.dominant}"
        )
        proposal = propose_critique(
            self.llm, draft=draft, affect=stm.affect, signals=stm.signals
        )
        critique = (
            f"Counter: {proposal.reason} (risk={proposal.risk}, "
            f"fwd×{proposal.forward_scale:.2f}, turnΔ={proposal.turn_delta:+.2f})."
            if proposal.reason
            else critique_draft(self.llm, draft=draft, affect=stm.affect, signals=stm.signals)
        )
        result = compose_thought(
            signals=stm.signals,
            affect=stm.affect,
            reflection=reflection,
            present_future=present_future,
            devil_critique=critique,
            action_clamps=self.config.action_clamps,
            future_action_bias=future.chosen_action,
            future_confidence=future.confidence,
            future_meta={
                "chosen_name": future.chosen_name,
                "chosen_action": np.asarray(future.chosen_action, dtype=np.float32).tolist(),
                "layer2_override": future.layer2_override,
                "layer2_reason": future.layer2_reason,
                "priority_weights": future.priority_weights,
                "layer1_values": future.layer1_values,
                "maturity": self.experience.maturity(),
                "tendency": tend_state.as_dict(),
                "devil_proposal": proposal.to_dict(),
            },
        )
        self._last_devil_proposal = proposal
        thought = self.memory.insert_thought(
            stm_id=stm.id,
            reflection=result.reflection,
            devil_critique=result.devil_critique,
            refined_thought=result.refined_thought,
            intent=result.intent,
        )
        self._recent_actions.append(
            {
                "stm_id": stm.id,
                "step": stm.step,
                "action_name": future.chosen_name,
                "action": result.action.tolist(),
            }
        )
        if len(self._recent_actions) > 64:
            self._recent_actions = self._recent_actions[-64:]
        self.experience.tick_step()
        return result, thought

    def maybe_retrospective(
        self,
        *,
        signals: dict[str, Any],
        affect: dict[str, Any],
        fly_id: str | None = None,
        force: bool = False,
        tick: int = 0,
    ) -> RetrospectiveResult:
        if not self.config.enable_retrospective and not force:
            return RetrospectiveResult(ran=False, reason="disabled")
        result = run_retrospective(
            memory=self.memory,
            experience=self.experience,
            outcome_memory=self.outcome_memory,
            fly_id=fly_id or self.config.fly_id,
            signals=signals,
            affect=affect,
            recent_actions=self._recent_actions,
            world_model=self.world_model,
            config=self.retrospective_config,
            force=force,
            tendency=self.tendency,
            interest_store=self.interest_store,
            tick=tick,
        )
        self._last_retrospective = result
        return result

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
        left_eye_vec: np.ndarray | None = None,
        right_eye_vec: np.ndarray | None = None,
        sensors_vec: np.ndarray | None = None,
        status_vec: np.ndarray | None = None,
        affect_vec: np.ndarray | None = None,
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
            left_eye_vec=left_eye_vec,
            right_eye_vec=right_eye_vec,
            sensors_vec=sensors_vec,
            status_vec=status_vec,
            affect_vec=affect_vec,
        )
        result, thought = self.think_on_record(stm)
        # Low-load retrospective after the hot-path decision.
        self.maybe_retrospective(signals=signals, affect=affect, fly_id=fly, tick=step)
        return stm, result, thought
