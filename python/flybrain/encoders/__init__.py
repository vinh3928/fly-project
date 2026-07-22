"""Observation encoders for multimodal fly sensors."""

from .body import BodyEncoder
from .chemical import ChemicalEncoder
from .fusion import MultimodalEncoder
from .internal import InternalStateEncoder
from .vision import VisionEncoder

__all__ = [
    "VisionEncoder",
    "BodyEncoder",
    "ChemicalEncoder",
    "InternalStateEncoder",
    "MultimodalEncoder",
]
