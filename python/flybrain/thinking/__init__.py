"""Thinking service: reflection, future stub, devil's advocate, composer."""

from .composer import ThoughtResult, compose_thought
from .devils_advocate import HeuristicDevilClient, critique_draft, make_llm_client
from .future_stub import present_future_stub
from .reflection import build_reflection
from .service import ThinkingConfig, ThinkingService

__all__ = [
    "HeuristicDevilClient",
    "ThinkingConfig",
    "ThinkingService",
    "ThoughtResult",
    "build_reflection",
    "compose_thought",
    "critique_draft",
    "make_llm_client",
    "present_future_stub",
]
