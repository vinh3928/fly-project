"""Vision-keyed place details: smell/sensors correlated onto Place nodes."""

from __future__ import annotations

import numpy as np

from flybrain.memory.place_graph import InMemoryPlaceGraph, PlaceGraphConfig
from flybrain.phase4 import SMELL_VEC_DIM
from flybrain.sensors.mapped_feedback import MappedSensorFeedback
from flybrain.sensors.place_senses import (
    infer_place_label,
    merge_place_senses,
    sense_rows_for_neo4j,
    senses_vector,
)


def test_infer_place_label_puddle_from_bitter() -> None:
    assert infer_place_label({"bitter": 0.49, "wall_hazard": 0.1}) == "puddle"
    assert infer_place_label({"sugar": 0.3}) == "puddle"
    assert infer_place_label({"odor": 0.2}) == "odor_plume"
    assert infer_place_label({"wall_hazard": 0.8}) == "wall"
    assert infer_place_label({}) == "place"


def test_merge_place_senses_stores_bitter_detail() -> None:
    props = merge_place_senses(
        {},
        {"bitter": 0.49, "odor": 0.1, "wall_hazard": 0.2},
        step=12,
        smell_vec=np.ones(SMELL_VEC_DIM, dtype=np.float32) * 0.2,
    )
    assert props["label"] == "puddle"
    assert abs(float(props["senses"]["bitter"]["value"]) - 0.49) < 1e-6
    assert props["senses"]["bitter"]["last_seen_step"] == 12
    assert len(props["smell_vec"]) == SMELL_VEC_DIM
    rows = sense_rows_for_neo4j("place-1", props["senses"])
    kinds = {r["kind"] for r in rows}
    assert "bitter" in kinds


def test_vision_place_gets_puddle_label_and_bitter_sense() -> None:
    """Vision keys the node; bitter reading becomes a detail on that Place."""
    g = InMemoryPlaceGraph()
    cfg = PlaceGraphConfig(
        enabled=True,
        backend="in_memory",
        match_min=0.9,
        familiarity_bump=0.25,
        hazard_decay_strength=0.9,
        hazard_decay_floor=0.1,
        context_enabled=False,
    )
    fb = MappedSensorFeedback(
        store=g,
        config=cfg,
        fly_id="fly_a",
        sense_gate_ttl_seconds=0.0,
        smell_gate_ttl_seconds=0.0,
    )
    view = np.zeros(90, dtype=np.float32)
    view[0] = 1.0
    smell = np.zeros(SMELL_VEC_DIM, dtype=np.float32)
    smell[0] = 0.4

    first = fb.sense(
        raw={"wall_hazard": 0.2, "bitter": 0.49, "odor": 0.2},
        view_vec=view,
        smell_vec=smell,
        step=0,
        now_ms=0.0,
    )
    assert first.is_new is True
    assert first.label == "puddle"
    assert first.place is not None
    senses = first.place.props.get("senses") or {}
    assert abs(float(senses["bitter"]["value"]) - 0.49) < 1e-6
    assert first.place.props.get("label") == "puddle"

    # Unchanged senses → L1 hit, details remain on the same vision Place.
    second = fb.sense(
        raw={"wall_hazard": 0.2, "bitter": 0.49, "odor": 0.2},
        view_vec=view,
        smell_vec=smell,
        step=1,
        now_ms=1.0,
    )
    assert second.is_new is False
    assert second.place is not None
    assert second.place.id == first.place.id
    assert second.senses_l1_hit is True
    assert second.smell_l1_hit is True
    assert abs(float(second.place.props["senses"]["bitter"]["value"]) - 0.49) < 1e-6


def test_senses_vector_order_stable() -> None:
    v = senses_vector({"bitter": 0.49, "wall_hazard": 0.1})
    assert v.shape[0] == 6
    assert abs(float(v[4]) - 0.49) < 1e-6  # bitter index in SENSE_CHANNELS
