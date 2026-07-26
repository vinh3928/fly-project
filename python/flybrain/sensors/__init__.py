"""Sensor-side helpers (mapped place feedback, modal L1 gates, etc.)."""

from .mapped_feedback import (
    CHANNEL_ITEMS,
    DEFAULT_FEEDBACK_CHANNELS,
    MappedSensorFeedback,
    MappedSensorReading,
    MappedSensorReport,
    attenuate_mapped_reading,
    item_familiarity,
)
from .modal_gate import (
    ModalChangeGate,
    ModalGateConfig,
    ModalSample,
    modal_delta,
    pad_modal_vec,
)
from .place_senses import (
    SENSE_CHANNELS,
    infer_place_label,
    merge_place_senses,
    sense_rows_for_neo4j,
    senses_vector,
)

__all__ = [
    "CHANNEL_ITEMS",
    "DEFAULT_FEEDBACK_CHANNELS",
    "MappedSensorFeedback",
    "MappedSensorReading",
    "MappedSensorReport",
    "ModalChangeGate",
    "ModalGateConfig",
    "ModalSample",
    "SENSE_CHANNELS",
    "attenuate_mapped_reading",
    "infer_place_label",
    "item_familiarity",
    "merge_place_senses",
    "modal_delta",
    "pad_modal_vec",
    "sense_rows_for_neo4j",
    "senses_vector",
]
