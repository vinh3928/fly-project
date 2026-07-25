"""Thinking service: reflection, two-layer future, devil's advocate, composer."""

from .composer import ThoughtResult, compose_thought
from .devils_advocate import (
    HeuristicDevilClient,
    critique_draft,
    make_llm_client,
    parse_proposal,
    propose_critique,
)
from .future_stub import present_future_stub
from .interest import (
    InterestConfig,
    InterestPoint,
    InterestPointStore,
    InvestigationDirective,
    apply_investigation_goal,
    eye_pair_similarity,
    frame_similarity,
)
from .judgement import (
    DevilProposal,
    JudgementConfig,
    JudgementModule,
    JudgementResult,
    apply_proposal,
    blend_goals,
)
from .personality import PersonalityState, PersonalityTraits
from .reflection import build_reflection
from .retrospective import (
    OutcomeMemory,
    RetrospectiveConfig,
    can_investigate_interest,
    is_low_load,
    run_retrospective,
)
from .service import ThinkingConfig, ThinkingService
from .stimulus_interrupt import (
    PatchGridHistoryCache,
    RelativeHistoryBaseline,
    StimulusInterruptConfig,
    StimulusInterruptDecision,
    STOP_GOAL,
    evaluate_stimulus_interrupt,
)
from .two_layer_future import FuturePrediction, predict_present_future
from .visual_thinker import (
    VisualInsight,
    VisualThinker,
    VisualThinkerConfig,
    make_visual_client,
)

__all__ = [
    "DevilProposal",
    "FuturePrediction",
    "HeuristicDevilClient",
    "InterestConfig",
    "InterestPoint",
    "InterestPointStore",
    "InvestigationDirective",
    "JudgementConfig",
    "JudgementModule",
    "JudgementResult",
    "OutcomeMemory",
    "PersonalityState",
    "PersonalityTraits",
    "RetrospectiveConfig",
    "PatchGridHistoryCache",
    "RelativeHistoryBaseline",
    "STOP_GOAL",
    "StimulusInterruptConfig",
    "StimulusInterruptDecision",
    "ThinkingConfig",
    "ThinkingService",
    "ThoughtResult",
    "VisualInsight",
    "VisualThinker",
    "VisualThinkerConfig",
    "apply_investigation_goal",
    "apply_proposal",
    "blend_goals",
    "build_reflection",
    "can_investigate_interest",
    "compose_thought",
    "critique_draft",
    "evaluate_stimulus_interrupt",
    "eye_pair_similarity",
    "frame_similarity",
    "is_low_load",
    "make_llm_client",
    "make_visual_client",
    "parse_proposal",
    "predict_present_future",
    "present_future_stub",
    "propose_critique",
    "run_retrospective",
]
