"""Thinking service: reflection, two-layer future, devil's advocate, composer."""

from .composer import ThoughtResult, compose_thought
from .devils_advocate import HeuristicDevilClient, critique_draft, make_llm_client
from .future_stub import present_future_stub
from .reflection import build_reflection
from .retrospective import OutcomeMemory, RetrospectiveConfig, is_low_load, run_retrospective
from .service import ThinkingConfig, ThinkingService
from .two_layer_future import FuturePrediction, predict_present_future

__all__ = [
    "FuturePrediction",
    "HeuristicDevilClient",
    "OutcomeMemory",
    "RetrospectiveConfig",
    "ThinkingConfig",
    "ThinkingService",
    "ThoughtResult",
    "build_reflection",
    "compose_thought",
    "critique_draft",
    "is_low_load",
    "make_llm_client",
    "predict_present_future",
    "present_future_stub",
    "run_retrospective",
]
