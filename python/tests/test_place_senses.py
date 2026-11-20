"""Vision-keyed place details: smell/sensors correlated onto Place nodes."""

from __future__ import annotations

import numpy as np

from flybrain.memory.place_graph import InMemoryPlaceGraph, PlaceGraphConfig
from flybrain.phase4 import PATCH_VEC_DIM, SMELL_VEC_DIM
from flybrain.sensors.mapped_feedback import MappedSensorFeedback
from flybrain.sensors.place_senses import (
    OUTSIDE_OPENING_MIN,
    SENSE_CHANNELS,
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
    view = np.zeros(PATCH_VEC_DIM, dtype=np.float32)
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
    assert v.shape[0] == len(SENSE_CHANNELS)
    assert abs(float(v[4]) - 0.49) < 1e-6  # bitter index in SENSE_CHANNELS


def test_mostly_open_view_is_labelled_outside() -> None:
    assert infer_place_label({"opening_strength": OUTSIDE_OPENING_MIN, "odor": 0.3}) == "outside"
    assert (
        infer_place_label({"opening_strength": OUTSIDE_OPENING_MIN - 0.01, "odor": 0.3})
        == "odor_plume"
    )
    assert infer_place_label({"opening_strength": 0.9, "sugar": 0.3}) == "puddle"


def test_outside_label_wins_over_plume_but_not_the_reverse() -> None:
    outside = merge_place_senses({"label": "odor_plume"}, {"opening_strength": 0.9})
    assert outside["label"] == "outside"
    assert abs(float(outside["senses"]["opening_strength"]["value"]) - 0.9) < 1e-6
    still_outside = merge_place_senses(outside, {"odor": 0.4})
    assert still_outside["label"] == "outside"


def test_vision_place_outside_the_box_is_labelled_outside() -> None:
    fb = MappedSensorFeedback(
        store=InMemoryPlaceGraph(),
        config=PlaceGraphConfig(enabled=True, backend="in_memory", context_enabled=False),
        fly_id="fly_a",
        sense_gate_ttl_seconds=0.0,
        smell_gate_ttl_seconds=0.0,
    )
    view = np.zeros(PATCH_VEC_DIM, dtype=np.float32)
    view[3] = 1.0
    report = fb.sense(
        raw={"wall_hazard": 0.0, "opening_strength": 0.95, "odor": 0.2}, view_vec=view, step=0
    )
    assert report.label == "outside"
    assert report.place is not None and report.place.props["label"] == "outside"
    assert "opening_strength" not in report.readings  # stored as a detail, never attenuated
