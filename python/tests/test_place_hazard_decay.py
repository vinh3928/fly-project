"""Mapped sensor feedback: Neo4j-known items report at decayed rates."""

from __future__ import annotations

import numpy as np

from flybrain.env_visual_nav import FakeVisualNavConfig, FakeVisualNavEnv
from flybrain.memory.place_graph import InMemoryPlaceGraph, PlaceGraphConfig
from flybrain.pipeline.live_mind import LiveMindConfig, LiveMindPipeline, format_live_step_log
from flybrain.sensors.mapped_feedback import (
    MappedSensorFeedback,
    attenuate_mapped_reading,
)
from flybrain.vision.novelty import stereo_patch_embedding


def test_format_live_step_log_includes_place() -> None:
    line = format_live_step_log(
        3,
        [0.1, 0.0, 0.5, 0.5, 0.0],
        {
            "mind_tick": True,
            "motor_goal": [0.1, 0.0],
            "vision_cues": {"wall_hazard": 0.4},
            "place": {
                "place_id": 2,
                "is_new": False,
                "similarity": 0.97,
                "mean_familiarity": 0.36,
                "wall_hazard_raw": 0.8,
            },
        },
    )
    assert "wall=0.40" in line
    assert "place=2(sim=0.97,fam=0.36,raw=0.80)" in line


def test_attenuate_mapped_reading_math() -> None:
    raw = 0.8
    reported, scale = attenuate_mapped_reading(
        raw, mapped=False, familiarity=1.0, strength=0.9, floor=0.1
    )
    assert reported == raw and scale == 1.0
    reported, scale = attenuate_mapped_reading(
        raw, mapped=True, familiarity=0.5, strength=0.9, floor=0.1
    )
    assert abs(reported - raw * max(0.1, 1.0 - 0.9 * 0.5)) < 1e-6
    assert scale < 1.0


def test_mapped_sensor_feedback_decays_wall_and_odor() -> None:
    g = InMemoryPlaceGraph()
    cfg = PlaceGraphConfig(
        enabled=True,
        backend="in_memory",
        match_min=0.9,
        familiarity_bump=0.25,
        hazard_decay_strength=0.9,
        hazard_decay_floor=0.1,
    )
    fb = MappedSensorFeedback(store=g, config=cfg, fly_id="fly_a")
    view = np.zeros(90, dtype=np.float32)
    view[0] = 1.0

    first = fb.sense(
        raw={"wall_hazard": 0.8, "odor": 0.6},
        view_vec=view,
        step=0,
    )
    assert first.is_new is True
    assert first.reported("wall_hazard") == 0.8
    assert first.reported("odor") == 0.6

    second = fb.sense(
        raw={"wall_hazard": 0.8, "odor": 0.6},
        view_vec=view,
        step=1,
    )
    assert second.is_new is False
    assert second.reported("wall_hazard") < 0.8
    assert second.raw("wall_hazard") == 0.8
    # Odor item bumped on revisit while present → also decays.
    assert second.readings["odor"].mapped is True
    assert second.reported("odor") < 0.6
    assert "odor" in (second.place.props.get("items") or {})


def test_live_mind_sensor_feedback_decays_wall_hazard() -> None:
    env = FakeVisualNavEnv(FakeVisualNavConfig(seed=7))
    obs, _ = env.reset()
    wall = np.zeros((1, 64, 64), dtype=np.float32)
    wall[:, :, :20] = 1.0
    env._left_eye = wall.copy()
    env._right_eye = wall.copy()

    graph = InMemoryPlaceGraph()
    pipe = LiveMindPipeline(
        env,
        config=LiveMindConfig(
            backend="in_memory",
            max_steps=4,
            llm_mock=True,
            fly_id="fly_place",
            mind_interval=1,
            place_graph=PlaceGraphConfig(
                enabled=True,
                backend="in_memory",
                match_min=0.9,
                familiarity_bump=0.25,
                hazard_decay_strength=0.9,
                hazard_decay_floor=0.1,
            ),
        ),
    )
    pipe.place_graph = graph
    assert pipe.mapped_sensors is not None
    pipe.mapped_sensors.store = graph

    _, info1 = pipe.process_observation(obs, left_eye=env.left_eye, right_eye=env.right_eye)
    place1 = info1.get("place") or {}
    assert place1.get("is_new") is True
    raw1 = float(place1.get("wall_hazard_raw", 0.0))
    assert raw1 > 0.4
    assert float(place1.get("wall_hazard", raw1)) == raw1
    assert "mapped_sensors" in info1
    assert info1["mapped_sensors"]["wall_hazard"]["raw"] == raw1

    pipe._step = 1
    _, info2 = pipe.process_observation(obs, left_eye=env.left_eye, right_eye=env.right_eye)
    place2 = info2.get("place") or {}
    assert place2.get("is_new") is False
    assert float(place2.get("wall_hazard", 1.0)) < raw1
    cues = info2.get("vision_cues") or {}
    assert float(cues.get("wall_hazard", 1.0)) < raw1
    assert float(cues.get("wall_hazard_raw", 0.0)) == raw1

    view = stereo_patch_embedding(env.left_eye, env.right_eye)
    hits = graph.nearest(view, fly_id="fly_place", top_k=1)
    assert hits and hits[0][1] >= 0.9
