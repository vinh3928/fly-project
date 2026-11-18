"""Route-home path SideChannel pack/parse + place pose resolution."""

from __future__ import annotations

import numpy as np

from flybrain.debug.route_path_channel import (
    FLAG_CURRENT,
    FLAG_HOME,
    FLAG_ROUTE,
    GRAPH_PROTOCOL_VERSION,
    LABEL_ODOR_PLUME,
    LABEL_PUDDLE,
    MODE_OFF,
    MODE_ROUTE,
    MODE_SCAN,
    PROTOCOL_VERSION,
    ROUTE_PATH_CHANNEL_ID,
    RoutePathDebugChannel,
    build_graph_snapshot,
    mode_from_route,
    pack_graph_message,
    pack_path_message,
    parse_graph_message,
    parse_path_message,
    parse_pose_message,
)
from flybrain.memory.place_graph import InMemoryPlaceGraph, observe_place


def test_channel_id_is_stable() -> None:
    assert str(ROUTE_PATH_CHANNEL_ID) == "c8f0a2b4-5d6e-4a1b-9c3d-7e8f90123456"


def test_pack_parse_roundtrip() -> None:
    pts = [(0.0, 0.02, 0.0), (0.1, 0.02, 0.2)]
    packed = pack_path_message(pts, mode=MODE_ROUTE, current_index=1)
    assert packed[0] == PROTOCOL_VERSION
    assert packed[1] == MODE_ROUTE
    assert packed[2] == 1.0
    assert packed[3] == 2.0
    parsed = parse_path_message(packed)
    assert parsed["mode"] == MODE_ROUTE
    assert parsed["current_index"] == 1
    assert parsed["points"] == pts


def test_pack_empty_clears() -> None:
    packed = pack_path_message([], mode=MODE_OFF)
    parsed = parse_path_message(packed)
    assert parsed["mode"] == MODE_OFF
    assert parsed["points"] == []
    assert parsed["current_index"] == -1


def test_parse_pose_message() -> None:
    from flybrain.debug.route_path_channel import parse_pose_yaw, world_bearing_turn

    pose = parse_pose_message([1.5, 0.02, -0.3, 9.0])
    assert pose is not None
    np.testing.assert_allclose(pose, [1.5, 0.02, -0.3])
    assert parse_pose_yaw([1.5, 0.02, -0.3, 90.0]) == 90.0
    assert parse_pose_yaw([1.5, 0.02, -0.3]) is None
    assert parse_pose_message([1.0]) is None

    # Yaw 0 → +Z; target to +X (right) → positive turn.
    turn_right = world_bearing_turn([0.0, 0.0, 0.0], 0.0, [1.0, 0.0, 0.0])
    assert turn_right > 0.5
    # Target ahead → near-zero turn.
    turn_fwd = world_bearing_turn([0.0, 0.0, 0.0], 0.0, [0.0, 0.0, 1.0])
    assert abs(turn_fwd) < 0.05
    # Target to -X (left) → negative turn.
    turn_left = world_bearing_turn([0.0, 0.0, 0.0], 0.0, [-1.0, 0.0, 0.0])
    assert turn_left < -0.5


def test_mode_from_route() -> None:
    assert mode_from_route(None) == MODE_OFF
    assert mode_from_route({"mode": "off"}) == MODE_OFF
    assert mode_from_route({"mode": "route"}) == MODE_ROUTE
    assert mode_from_route({"mode": "handoff"}) == MODE_ROUTE
    assert mode_from_route({"mode": "scan"}) == MODE_SCAN


def test_build_path_points_resolves_place_poses() -> None:
    ch = RoutePathDebugChannel()
    ch.last_pose = np.asarray([0.0, 0.02, 0.0], dtype=np.float32)
    ch.record_place_pose("a", np.asarray([0.0, 0.02, 0.0], dtype=np.float32))
    ch.record_place_pose("b", np.asarray([0.1, 0.02, 0.1], dtype=np.float32))
    ch.record_place_pose("c", np.asarray([0.2, 0.02, 0.2], dtype=np.float32))

    route = {"mode": "route", "path": ["a", "b", "c"]}
    points, cur = ch.build_path_points(route, current_place_id="b")
    assert len(points) >= 3
    assert cur >= 0
    # Fly tip + places (a may collapse with fly).
    np.testing.assert_allclose(points[-1], (0.2, 0.02, 0.2), atol=1e-6)


def test_send_route_queues_packed_floats() -> None:
    ch = RoutePathDebugChannel()
    ch.record_place_pose("home", np.asarray([0.05, 0.02, 0.05], dtype=np.float32))
    packed = ch.send_route(
        {"mode": "scan", "path": ["home"]},
        current_place_id="home",
    )
    parsed = parse_path_message(packed)
    assert parsed["mode"] == MODE_SCAN
    assert len(parsed["points"]) == 1
    np.testing.assert_allclose(parsed["points"][0], (0.05, 0.02, 0.05), atol=1e-6)


def test_clear_resets_poses() -> None:
    ch = RoutePathDebugChannel()
    ch.record_place_pose("a", np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    ch.clear()
    assert ch.pose_for("a") is None
    assert ch.last_pose is None


def test_pack_parse_graph_roundtrip() -> None:
    nodes = [
        {
            "label": LABEL_PUDDLE,
            "flags": FLAG_HOME | FLAG_CURRENT,
            "x": 0.1,
            "y": 0.02,
            "z": 0.2,
            "smell": 0.7,
        },
        {
            "label": LABEL_ODOR_PLUME,
            "flags": FLAG_ROUTE,
            "x": 0.0,
            "y": 0.02,
            "z": 0.0,
            "smell": 0.3,
        },
    ]
    packed = pack_graph_message(nodes, [(0, 1)])
    assert packed[0] == GRAPH_PROTOCOL_VERSION
    assert packed[1] == 2.0
    assert packed[2] == 1.0
    parsed = parse_graph_message(packed)
    assert parsed["version"] == GRAPH_PROTOCOL_VERSION
    assert len(parsed["nodes"]) == 2
    assert parsed["nodes"][0]["label"] == LABEL_PUDDLE
    assert parsed["nodes"][0]["flags"] == FLAG_HOME | FLAG_CURRENT
    np.testing.assert_allclose(
        [parsed["nodes"][0]["x"], parsed["nodes"][0]["y"], parsed["nodes"][0]["z"]],
        [0.1, 0.02, 0.2],
    )
    assert parsed["edges"] == [(0, 1)]
    # parse_path_message routes version>=2 to graph parser
    via_path = parse_path_message(packed)
    assert via_path["nodes"][1]["label"] == LABEL_ODOR_PLUME


def test_build_graph_snapshot_flags_and_world_xyz() -> None:
    g = InMemoryPlaceGraph(view_dim=8)
    a = observe_place(
        g,
        np.zeros(8, dtype=np.float32),
        fly_id="fly_a",
        step=0,
        props={
            "label": "odor_plume",
            "world_xyz": [0.0, 0.02, 0.0],
            "senses": {"sugar_smell": {"value": 0.4}},
        },
    ).place
    b = observe_place(
        g,
        np.ones(8, dtype=np.float32) * 0.5,
        fly_id="fly_a",
        step=1,
        from_place_id=a.id,
        props={
            "label": "puddle",
            "world_xyz": [0.2, 0.02, 0.1],
            "senses": {"sugar_smell": {"value": 0.9}},
        },
    ).place
    g.upsert_adjacent(a.id, b.id)

    poses = {
        str(a.id): np.asarray([0.0, 0.02, 0.0], dtype=np.float32),
        str(b.id): np.asarray([0.2, 0.02, 0.1], dtype=np.float32),
    }
    nodes, edges = build_graph_snapshot(
        g,
        place_poses=poses,
        current_place_id=a.id,
        home_id=b.id,
        route_path=[a.id, b.id],
        fly_id="fly_a",
    )
    assert len(nodes) == 2
    by_id = {n["id"]: n for n in nodes}
    assert by_id[str(a.id)]["flags"] & FLAG_CURRENT
    assert by_id[str(a.id)]["flags"] & FLAG_ROUTE
    assert by_id[str(b.id)]["flags"] & FLAG_HOME
    assert by_id[str(b.id)]["label"] == LABEL_PUDDLE
    assert by_id[str(b.id)]["smell"] >= 0.9
    assert len(edges) >= 1


def test_send_place_graph_queues_version2() -> None:
    ch = RoutePathDebugChannel()
    g = InMemoryPlaceGraph(view_dim=4)
    node = observe_place(
        g,
        np.zeros(4, dtype=np.float32),
        fly_id="fly_a",
        step=0,
        props={"label": "place", "world_xyz": [1.0, 0.02, 2.0]},
    ).place
    ch.record_place_pose(node.id, np.asarray([1.0, 0.02, 2.0], dtype=np.float32))
    packed = ch.send_place_graph(
        g, current_place_id=node.id, home_id=node.id, fly_id="fly_a", force=True
    )
    assert packed is not None
    parsed = parse_graph_message(packed)
    assert parsed["version"] == GRAPH_PROTOCOL_VERSION
    assert len(parsed["nodes"]) == 1
    assert parsed["nodes"][0]["flags"] & FLAG_CURRENT
    assert parsed["nodes"][0]["flags"] & FLAG_HOME


def test_send_place_graph_throttles_unchanged_snapshots() -> None:
    ch = RoutePathDebugChannel()
    g = InMemoryPlaceGraph(view_dim=4)
    node = observe_place(
        g,
        np.zeros(4, dtype=np.float32),
        fly_id="fly_a",
        step=0,
        props={"label": "place", "world_xyz": [1.0, 0.02, 2.0]},
    ).place
    ch.record_place_pose(node.id, np.asarray([1.0, 0.02, 2.0], dtype=np.float32))
    first = ch.send_place_graph(
        g, current_place_id=node.id, home_id=node.id, step=1, force=True
    )
    assert first is not None
    skipped = ch.send_place_graph(
        g, current_place_id=node.id, home_id=node.id, step=2
    )
    assert skipped is None
    forced_tick = ch.send_place_graph(
        g, current_place_id=node.id, home_id=node.id, step=8
    )
    assert forced_tick is not None
