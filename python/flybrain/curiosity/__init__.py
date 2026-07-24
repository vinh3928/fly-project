"""Intrinsic motivation / curiosity rewards."""

from flybrain.curiosity.boredom import (
    BoredomConfig,
    BoredomState,
    BoredomTracker,
    apply_boredom_escape_goal,
    apply_boredom_to_candidate_value,
    thought_fingerprint,
)
from flybrain.curiosity.drive import CuriosityConfig, CuriosityDrive, CuriosityStepResult, curiosity_gate
from flybrain.curiosity.module import CuriosityModule
from flybrain.curiosity.novelty import NoveltyConfig, NoveltyObservation, NoveltyTracker

__all__ = [
    "BoredomConfig",
    "BoredomState",
    "BoredomTracker",
    "CuriosityConfig",
    "CuriosityDrive",
    "CuriosityModule",
    "CuriosityStepResult",
    "NoveltyConfig",
    "NoveltyObservation",
    "NoveltyTracker",
    "apply_boredom_escape_goal",
    "apply_boredom_to_candidate_value",
    "curiosity_gate",
    "thought_fingerprint",
]
