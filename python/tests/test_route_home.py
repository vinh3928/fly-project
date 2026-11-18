"""Path home via ADJACENT place memory (prefer sugar puddle, else chamber anchor)."""

from __future__ import annotations

import numpy as np

from flybrain.memory.place_graph import InMemoryPlaceGraph, observe_place
from flybrain.phase4 import PATCH_VEC_DIM
from flybrain.thinking.route_home import RouteHomeConfig, RouteHomeState, RouteHomeTracker
from flybrain.thinking.two_layer_future import cue_turn


def _view(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=PATCH_VEC_DIM).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def test_shortest_path_and_neighbors_undirected() -> None:
    g = InMemoryPlaceGraph()
    a = g.create_place(_view(1), fly_id="fly_a")
    b = g.create_place(_view(2), fly_id="fly_a")
    c = g.create_place(_view(3), fly_id="fly_a")
    g.upsert_adjacent(a.id, b.id, props={"turn": 0.4})
    g.upsert_adjacent(b.id, c.id, props={"turn": -0.2})
    assert g.shortest_path(a.id, c.id) == [a.id, b.id, c.id]
    assert g.shortest_path(c.id, a.id) == [c.id, b.id, a.id]  # undirected
    others = set()
    for e in g.neighbors(b.id):
        others.add(str(e.src_id if str(e.dst_id) == str(b.id) else e.dst_id))
    assert others == {str(a.id), str(c.id)}
    assert g.find_places(fly_id="fly_a", limit=10)


def test_observe_place_stores_edge_turn() -> None:
    g = InMemoryPlaceGraph()
    first = observe_place(g, _view(1), fly_id="fly_a", step=0)
    second = observe_place(
        g,
        _view(2),
        fly_id="fly_a",
        step=1,
        from_place_id=first.place.id,
        edge_props={"turn": 0.55, "opening_strength": 0.9},
    )
    assert second.edge_created
    edge = g.edges()[0]
    assert edge.props["turn"] == 0.55
    assert edge.props["opening_strength"] == 0.9


def test_route_prefers_sugar_puddle_over_chamber_anchor() -> None:
    g = InMemoryPlaceGraph()
    chamber = g.create_place(_view(1), fly_id="fly_a", props={"label": "place"})
    tunnel = g.create_place(_view(2), fly_id="fly_a", props={"label": "place"})
    outside = g.create_place(_view(3), fly_id="fly_a", props={"label": "outside"})
    puddle = g.create_place(
        _view(4),
        fly_id="fly_a",
        props={"label": "puddle", "senses": {"sugar": {"value": 0.8}}},
    )
    g.upsert_adjacent(puddle.id, chamber.id, props={"turn": 0.1})
    g.upsert_adjacent(chamber.id, tunnel.id, props={"turn": 0.3})
    g.upsert_adjacent(tunnel.id, outside.id, props={"turn": 0.5, "opening_strength": 0.9})

    tracker = RouteHomeTracker(config=RouteHomeConfig(), fly_id="fly_a")
    # Walk out so chamber becomes the anchor.
    tracker.observe(
        store=g,
        place=chamber,
        signals={"place_label": "place", "hunger": 0.0, "energy": 1.0},
    )
    tracker.observe(
        store=g,
        place=outside,
        signals={"place_label": "outside", "hunger": 0.0, "energy": 1.0},
    )
    assert tracker._chamber_anchor_id == chamber.id

    state = tracker.observe(
        store=g,
        place=outside,
        signals={
            "place_label": "outside",
            "hunger": 0.9,
            "energy": 0.1,
            "sugar_smell": 0.5,
            "opening_strength": 0.9,
            "opening_bearing": 0.0,
        },
    )
    assert state.mode == "route"
    assert state.home_id == puddle.id
    assert state.reason in ("graph_food", "sugar_puddle")
    assert state.food_kind == "sugar"
    assert state.path[0] == outside.id and state.path[-1] == puddle.id
    assert state.next_id == tunnel.id
    assert state.path_edges  # ADJACENT hops tracked for return steering
    # Reverse of outbound tunnel→outside turn 0.5 → -0.5
    assert state.turn == -0.5


def test_route_falls_back_to_chamber_anchor() -> None:
    g = InMemoryPlaceGraph()
    chamber = g.create_place(_view(1), fly_id="fly_a", props={"label": "place"})
    outside = g.create_place(_view(2), fly_id="fly_a", props={"label": "outside"})
    g.upsert_adjacent(chamber.id, outside.id, props={"turn": 0.4})

    tracker = RouteHomeTracker(fly_id="fly_a")
    tracker.observe(store=g, place=chamber, signals={"place_label": "place", "energy": 1.0})
    tracker.observe(store=g, place=outside, signals={"place_label": "outside", "energy": 1.0})
    state = tracker.observe(
        store=g,
        place=outside,
        signals={"place_label": "outside", "hunger": 0.9, "energy": 0.1, "sugar_smell": 0.2},
    )
    assert state.mode == "route"
    assert state.home_id == chamber.id
    assert state.turn == -0.4


def test_handoff_at_home_allows_chemotaxis() -> None:
    g = InMemoryPlaceGraph()
    home = g.create_place(
        _view(1), fly_id="fly_a", props={"label": "puddle", "senses": {"sugar": {"value": 0.9}}}
    )
    tracker = RouteHomeTracker(fly_id="fly_a")
    tracker._home_place_id = home.id
    state = tracker.observe(
        store=g,
        place=home,
        signals={"place_label": "puddle", "hunger": 0.9, "energy": 0.1, "sugar_smell": 0.8},
    )
    assert state.mode == "handoff"
    assert state.turn is None


def test_handoff_on_sugar_even_when_routing_elsewhere() -> None:
    """Contact sugar on a different puddle must stop route (log: p0.0 past food)."""
    g = InMemoryPlaceGraph()
    mid = g.create_place(_view(1), fly_id="fly_a", props={"label": "place"})
    other_home = g.create_place(
        _view(2), fly_id="fly_a", props={"label": "puddle", "senses": {"sugar": {"value": 0.9}}}
    )
    here = g.create_place(
        _view(3), fly_id="fly_a", props={"label": "puddle", "senses": {"sugar": {"value": 0.8}}}
    )
    g.upsert_adjacent(here.id, mid.id, props={"turn": 0.5})
    g.upsert_adjacent(mid.id, other_home.id, props={"turn": 0.4})

    tracker = RouteHomeTracker(fly_id="fly_a")
    tracker._home_place_id = other_home.id
    state = tracker.observe(
        store=g,
        place=here,
        signals={
            "place_label": "puddle",
            "hunger": 0.9,
            "energy": 0.01,
            "sugar_smell": 0.85,
            "sugar": 0.9,
            "proboscis_contact": 1.0,
        },
    )
    assert state.mode == "handoff"
    assert state.turn is None
    assert state.reason == "on_sugar"
    assert state.home_id == here.id
    assert state.hops_left == 0
    assert not state.active


def test_cue_turn_route_beats_sugar() -> None:
    turn, source = cue_turn(
        {
            "hunger": 0.9,
            "energy": 0.1,
            "sugar_smell": 0.8,
            "sugar_bearing": 1.0,
            "route_home_turn": -0.6,
        }
    )
    assert source == "route_home"
    assert turn == -0.6


def test_outside_stored_sugar_sense_does_not_become_food_home() -> None:
    g = InMemoryPlaceGraph()
    chamber = g.create_place(_view(1), fly_id="fly_a", props={"label": "place"})
    outside = g.create_place(
        _view(2),
        fly_id="fly_a",
        props={"label": "outside", "senses": {"sugar": {"value": 0.6}}},
    )
    g.upsert_adjacent(chamber.id, outside.id, props={"turn": 0.4})
    tracker = RouteHomeTracker(fly_id="fly_a")
    tracker.observe(
        store=g,
        place=outside,
        signals={"place_label": "outside", "energy": 1.0, "sugar_smell": 0.6},
    )
    assert tracker._home_place_id is None  # outdoor never permanent bookmark
    assert tracker._peak_smell_id == outside.id
    # Even if a prior bug bookmarked outside, permanent bookmark must clear.
    tracker._home_place_id = outside.id
    tracker._chamber_anchor_id = chamber.id
    state = tracker.observe(
        store=g,
        place=outside,
        signals={"place_label": "outside", "hunger": 0.9, "energy": 0.1, "sugar_smell": 0.2},
    )
    assert tracker._home_place_id is None
    # Already on the plume peak → skip self; fall back to chamber.
    assert state.mode == "route"
    assert state.home_id == chamber.id


def test_outdoor_odor_plume_trail_redirects_without_bookmark() -> None:
    """Outdoor plume smell is trail-only: reverse-path there, never bookmarked_home."""
    g = InMemoryPlaceGraph()
    chamber = g.create_place(
        _view(1),
        fly_id="fly_a",
        props={"label": "place", "senses": {"opening_strength": {"value": 0.1}}},
    )
    plume = g.create_place(
        _view(2),
        fly_id="fly_a",
        props={
            "label": "odor_plume",
            "senses": {"opening_strength": {"value": 0.53}, "odor": {"value": 0.4}},
        },
    )
    outside = g.create_place(_view(3), fly_id="fly_a", props={"label": "outside"})
    g.upsert_adjacent(chamber.id, outside.id, props={"turn": 0.4})
    g.upsert_adjacent(plume.id, outside.id, props={"turn": -0.55})
    tracker = RouteHomeTracker(fly_id="fly_a")
    tracker.observe(
        store=g,
        place=plume,
        signals={
            "place_label": "odor_plume",
            "energy": 1.0,
            "sugar_smell": 0.5,
            "opening_strength": 0.53,
        },
    )
    assert tracker._home_place_id is None
    assert tracker._peak_smell_id == plume.id
    # Stale wrong bookmark must be cleared; trail still redirects to the plume.
    tracker._home_place_id = plume.id
    state = tracker.observe(
        store=g,
        place=outside,
        signals={
            "place_label": "outside",
            "hunger": 0.9,
            "energy": 0.1,
            "sugar_smell": 0.0,
            "opening_strength": 0.95,
        },
    )
    assert tracker._home_place_id is None
    assert state.mode == "route"
    assert state.home_id == plume.id
    assert state.reason == "plume_trail"
    assert state.turn == 0.55  # reverse of outside→plume edge


def test_world_steer_turns_toward_next_place_xyz() -> None:
    """When places carry Unity world_xyz, route turn follows bearing not edge turn."""
    g = InMemoryPlaceGraph()
    home = g.create_place(
        _view(1),
        fly_id="fly_a",
        props={
            "label": "puddle",
            "senses": {"sugar": {"value": 0.9}, "opening_strength": {"value": 0.05}},
            "world_xyz": [0.0, 0.02, 1.0],
        },
    )
    mid = g.create_place(
        _view(2),
        fly_id="fly_a",
        props={
            "label": "outside",
            # To the right of fly at origin facing +Z.
            "world_xyz": [1.0, 0.02, 0.0],
        },
    )
    outside = g.create_place(
        _view(3),
        fly_id="fly_a",
        props={"label": "outside", "world_xyz": [0.0, 0.02, 0.0]},
    )
    # Edge turn deliberately wrong (hard left) — world steer must override.
    g.upsert_adjacent(home.id, mid.id, props={"turn": 0.2})
    g.upsert_adjacent(mid.id, outside.id, props={"turn": -0.9})

    tracker = RouteHomeTracker(
        fly_id="fly_a",
        config=RouteHomeConfig(world_steer=True, min_turn_degrees=0),
    )
    tracker._home_place_id = home.id
    state = tracker.observe(
        store=g,
        place=outside,
        signals={
            "place_label": "outside",
            "hunger": 0.9,
            "energy": 0.1,
            "sugar_smell": 0.0,
            "fly_xyz": [0.0, 0.02, 0.0],
            "fly_yaw_deg": 0.0,
        },
    )
    assert state.mode == "route"
    assert state.next_id == mid.id
    assert state.turn is not None and state.turn > 0.5  # right toward mid at +X


def test_checkpoint_advances_by_world_proximity_without_place_rematch() -> None:
    """Near next node's XYZ → consume checkpoint even if matcher minted a new id."""
    g = InMemoryPlaceGraph()
    home = g.create_place(
        _view(1),
        fly_id="fly_a",
        props={
            "label": "puddle",
            "senses": {"sugar": {"value": 0.9}, "opening_strength": {"value": 0.05}},
            "world_xyz": [0.0, 0.02, 2.0],
        },
    )
    mid = g.create_place(
        _view(2),
        fly_id="fly_a",
        props={"label": "outside", "world_xyz": [0.0, 0.02, 1.0]},
    )
    start = g.create_place(
        _view(3),
        fly_id="fly_a",
        props={"label": "outside", "world_xyz": [0.0, 0.02, 0.0]},
    )
    g.upsert_adjacent(home.id, mid.id, props={"turn": 0.1})
    g.upsert_adjacent(mid.id, start.id, props={"turn": 0.1})

    tracker = RouteHomeTracker(
        fly_id="fly_a",
        config=RouteHomeConfig(
            world_steer=True,
            world_arrive_m=0.1,
            min_turn_degrees=0,
        ),
    )
    tracker._home_place_id = home.id
    # Establish cached path start → mid → home.
    first = tracker.observe(
        store=g,
        place=start,
        signals={
            "place_label": "outside",
            "hunger": 0.9,
            "energy": 0.1,
            "sugar_smell": 0.0,
            "fly_xyz": [0.0, 0.02, 0.0],
            "fly_yaw_deg": 0.0,
        },
    )
    assert first.mode == "route"
    assert first.next_id == mid.id
    assert first.hops_left == 2

    # New place id (not on path) but physically at mid's XYZ → advance to home.
    stray = g.create_place(
        _view(99),
        fly_id="fly_a",
        props={"label": "outside", "world_xyz": [0.0, 0.02, 1.0]},
    )
    second = tracker.observe(
        store=g,
        place=stray,
        signals={
            "place_label": "outside",
            "hunger": 0.9,
            "energy": 0.1,
            "sugar_smell": 0.0,
            "fly_xyz": [0.0, 0.02, 1.0],
            "fly_yaw_deg": 0.0,
        },
    )
    assert second.mode == "route"
    assert second.reason in ("checkpoint_home", "graph_food", "bookmarked_home")
    assert second.next_id == home.id
    assert second.hops_left == 1
    # Must not replan into a longer outside chain.
    assert second.hops_left < first.hops_left


def test_plume_trail_beats_distant_sugar_puddle() -> None:
    """Nearby session plume peak wins over a far enclosed sugar_puddle memory."""
    g = InMemoryPlaceGraph()
    puddle = g.create_place(
        _view(1),
        fly_id="fly_a",
        props={
            "label": "puddle",
            "senses": {"sugar": {"value": 0.9}, "opening_strength": {"value": 0.05}},
        },
    )
    a = g.create_place(_view(2), fly_id="fly_a", props={"label": "outside"})
    b = g.create_place(_view(3), fly_id="fly_a", props={"label": "outside"})
    plume = g.create_place(_view(4), fly_id="fly_a", props={"label": "odor_plume"})
    # Long chain to puddle; short hop back to plume.
    g.upsert_adjacent(puddle.id, a.id, props={"turn": 0.1})
    g.upsert_adjacent(a.id, b.id, props={"turn": 0.1})
    g.upsert_adjacent(b.id, plume.id, props={"turn": 0.4})

    tracker = RouteHomeTracker(fly_id="fly_a")
    tracker._home_place_id = puddle.id  # stale enclosed food bookmark
    tracker.observe(
        store=g,
        place=plume,
        signals={"place_label": "odor_plume", "energy": 1.0, "sugar_smell": 0.55},
    )
    tracker.observe(
        store=g,
        place=b,
        signals={"place_label": "outside", "energy": 1.0, "sugar_smell": 0.1},
    )
    state = tracker.observe(
        store=g,
        place=b,
        signals={"place_label": "outside", "hunger": 0.9, "energy": 0.05, "sugar_smell": 0.0},
    )
    assert state.mode == "route"
    assert state.home_id == plume.id
    assert state.reason == "plume_trail"
    assert state.hops_left == 1
    assert state.next_id == plume.id


def test_route_in_progress_while_active_home_or_steering() -> None:
    tracker = RouteHomeTracker(fly_id="fly_a")
    assert not tracker.route_in_progress
    tracker._active_home_id = "home"
    assert tracker.route_in_progress
    tracker._clear_active_route()
    tracker._state = RouteHomeState(
        mode="route", turn=0.4, forward=0.55, home_id="home", hops_left=2
    )
    assert tracker.route_in_progress


def test_active_route_home_stays_locked_when_hunger_retriggers() -> None:
    """Low-energy starts a path; later hunger + new plume must not retarget."""
    g = InMemoryPlaceGraph()
    chamber = g.create_place(_view(1), fly_id="fly_a", props={"label": "place"})
    first_plume = g.create_place(_view(2), fly_id="fly_a", props={"label": "odor_plume"})
    second_plume = g.create_place(_view(3), fly_id="fly_a", props={"label": "odor_plume"})
    outside = g.create_place(_view(4), fly_id="fly_a", props={"label": "outside"})
    g.upsert_adjacent(chamber.id, outside.id, props={"turn": 0.3})
    g.upsert_adjacent(first_plume.id, outside.id, props={"turn": -0.4})
    g.upsert_adjacent(second_plume.id, outside.id, props={"turn": 0.6})

    tracker = RouteHomeTracker(fly_id="fly_a")
    # Leave a weaker plume on the trail, then go outside.
    tracker.observe(
        store=g,
        place=first_plume,
        signals={
            "place_label": "odor_plume",
            "energy": 1.0,
            "sugar_smell": 0.4,
        },
    )
    # Low energy (not yet high hunger) starts the return path to first plume.
    first = tracker.observe(
        store=g,
        place=outside,
        signals={
            "place_label": "outside",
            "hunger": 0.2,
            "energy": 0.5,
            "sugar_smell": 0.0,
        },
    )
    assert first.mode == "route"
    assert first.home_id == first_plume.id
    assert tracker._active_home_id == first_plume.id

    # Stronger plume appears while still hungry — must keep the active home.
    tracker.observe(
        store=g,
        place=second_plume,
        signals={
            "place_label": "odor_plume",
            "hunger": 0.9,
            "energy": 0.2,
            "sugar_smell": 0.9,
        },
    )
    again = tracker.observe(
        store=g,
        place=outside,
        signals={
            "place_label": "outside",
            "hunger": 0.9,
            "energy": 0.2,
            "sugar_smell": 0.0,
        },
    )
    assert again.mode == "route"
    assert again.home_id == first_plume.id
    assert again.home_id != second_plume.id
    assert tracker._active_home_id == first_plume.id


def test_active_route_clears_when_sated_so_next_bout_can_retarget() -> None:
    g = InMemoryPlaceGraph()
    plume_a = g.create_place(_view(1), fly_id="fly_a", props={"label": "odor_plume"})
    plume_b = g.create_place(_view(2), fly_id="fly_a", props={"label": "odor_plume"})
    outside = g.create_place(_view(3), fly_id="fly_a", props={"label": "outside"})
    g.upsert_adjacent(plume_a.id, outside.id, props={"turn": 0.4})
    g.upsert_adjacent(plume_b.id, outside.id, props={"turn": -0.5})

    tracker = RouteHomeTracker(fly_id="fly_a")
    tracker.observe(
        store=g,
        place=plume_a,
        signals={"place_label": "odor_plume", "energy": 1.0, "sugar_smell": 0.5},
    )
    first = tracker.observe(
        store=g,
        place=outside,
        signals={"place_label": "outside", "hunger": 0.9, "energy": 0.1, "sugar_smell": 0.0},
    )
    assert first.home_id == plume_a.id
    # Sated → drop the bout lock.
    off = tracker.observe(
        store=g,
        place=outside,
        signals={"place_label": "outside", "hunger": 0.0, "energy": 1.0, "sugar_smell": 0.0},
    )
    assert off.mode == "off"
    assert tracker._active_home_id is None

    tracker.observe(
        store=g,
        place=plume_b,
        signals={"place_label": "odor_plume", "energy": 1.0, "sugar_smell": 0.7},
    )
    second = tracker.observe(
        store=g,
        place=outside,
        signals={"place_label": "outside", "hunger": 0.9, "energy": 0.1, "sugar_smell": 0.0},
    )
    assert second.home_id == plume_b.id


def test_resolve_refuses_longer_replan_while_path_active() -> None:
    """New place-id chain must not replace a short active checkpoint path."""
    g = InMemoryPlaceGraph()
    home = g.create_place(
        _view(1),
        fly_id="fly_a",
        props={
            "label": "puddle",
            "senses": {"sugar": {"value": 0.9}, "opening_strength": {"value": 0.05}},
            "world_xyz": [0.0, 0.02, 2.0],
        },
    )
    mid = g.create_place(
        _view(2),
        fly_id="fly_a",
        props={"label": "outside", "world_xyz": [0.0, 0.02, 1.0]},
    )
    start = g.create_place(
        _view(3),
        fly_id="fly_a",
        props={"label": "outside", "world_xyz": [0.0, 0.02, 0.0]},
    )
    g.upsert_adjacent(home.id, mid.id, props={"turn": 0.1})
    g.upsert_adjacent(mid.id, start.id, props={"turn": 0.1})

    tracker = RouteHomeTracker(
        fly_id="fly_a",
        config=RouteHomeConfig(world_steer=True, world_arrive_m=0.1, min_turn_degrees=0),
    )
    tracker._home_place_id = home.id
    first = tracker.observe(
        store=g,
        place=start,
        signals={
            "place_label": "outside",
            "hunger": 0.9,
            "energy": 0.1,
            "sugar_smell": 0.0,
            "fly_xyz": [0.0, 0.02, 0.0],
            "fly_yaw_deg": 0.0,
        },
    )
    assert first.hops_left == 2

    # Long detour graph from a fresh place id near start — replan would be longer.
    detour = [g.create_place(_view(10 + i), fly_id="fly_a", props={"label": "outside"}) for i in range(5)]
    stray = g.create_place(
        _view(99),
        fly_id="fly_a",
        props={"label": "outside", "world_xyz": [0.05, 0.02, 0.05]},
    )
    g.upsert_adjacent(stray.id, detour[0].id, props={"turn": 0.1})
    for a, b in zip(detour, detour[1:]):
        g.upsert_adjacent(a.id, b.id, props={"turn": 0.1})
    g.upsert_adjacent(detour[-1].id, home.id, props={"turn": 0.1})

    second = tracker.observe(
        store=g,
        place=stray,
        signals={
            "place_label": "outside",
            "hunger": 0.9,
            "energy": 0.1,
            "sugar_smell": 0.0,
            "fly_xyz": [0.05, 0.02, 0.05],
            "fly_yaw_deg": 0.0,
        },
    )
    assert second.mode == "route"
    assert second.home_id == home.id
    assert second.hops_left <= first.hops_left


def test_strong_sugar_smell_bookmarks_food_home_inside() -> None:
    g = InMemoryPlaceGraph()
    chamber = g.create_place(_view(1), fly_id="fly_a", props={"label": "place"})
    tunnel = g.create_place(_view(2), fly_id="fly_a", props={"label": "place"})
    outside = g.create_place(_view(3), fly_id="fly_a", props={"label": "outside"})
    g.upsert_adjacent(chamber.id, tunnel.id, props={"turn": 0.2})
    g.upsert_adjacent(tunnel.id, outside.id, props={"turn": 0.3})
    tracker = RouteHomeTracker(fly_id="fly_a")
    tracker.observe(
        store=g,
        place=chamber,
        signals={"place_label": "place", "energy": 1.0, "sugar_smell": 0.4},
    )
    assert tracker._home_place_id == chamber.id
    tracker.observe(
        store=g,
        place=outside,
        signals={"place_label": "outside", "energy": 1.0, "hunger": 0.0, "sugar_smell": 0.5},
    )
    # Outside smell must not replace the chamber food bookmark.
    assert tracker._home_place_id == chamber.id
    far = tracker.observe(
        store=g,
        place=outside,
        signals={"place_label": "outside", "hunger": 0.9, "energy": 0.1, "sugar_smell": 0.2},
    )
    assert far.mode == "route"
    assert far.home_id == chamber.id
    assert far.hops_left == 2
    assert far.next_id == tunnel.id


def test_starts_outside_uses_reachable_chamber_from_graph() -> None:
    g = InMemoryPlaceGraph()
    chamber = g.create_place(_view(1), fly_id="fly_a", props={"label": "place"})
    outside = g.create_place(_view(2), fly_id="fly_a", props={"label": "outside"})
    g.upsert_adjacent(chamber.id, outside.id, props={"turn": 0.4})
    tracker = RouteHomeTracker(fly_id="fly_a")
    # No prior chamber visit in this session — only the graph knows the interior.
    state = tracker.observe(
        store=g,
        place=outside,
        signals={"place_label": "outside", "hunger": 0.9, "energy": 0.1, "sugar_smell": 0.0},
    )
    assert tracker._chamber_anchor_id == chamber.id
    assert state.mode == "route"
    assert state.home_id == chamber.id
    assert state.turn == -0.4


def test_layer1_locks_feed_while_routing() -> None:
    from flybrain.thinking.two_layer_future import layer1_quick_scan
    from flybrain.world_model.outcomes import LifeExperience

    best, scored, _ = layer1_quick_scan(
        signals={
            "hunger": 0.9,
            "energy": 0.1,
            "odor": 0.8,
            "route_home_mode": "route",
            "route_home_turn": -0.5,
        },
        affect={"survival_risk": 0.4},
        experience=LifeExperience(),
    )
    assert best.name == "feed"
    assert float(best.action[1]) == -0.5


def test_gradual_walk_then_hungry_returns_through_hops() -> None:
    g = InMemoryPlaceGraph()
    inside = _view(1)
    outside = _view(2)
    prev = None
    places = []
    for step, t in enumerate(np.linspace(0.0, 1.0, 40)):
        view = (1.0 - t) * inside + t * outside
        obs = observe_place(
            g,
            view,
            fly_id="fly_a",
            step=step,
            from_place_id=prev,
            edge_props={"turn": 0.2 if step > 0 else 0.0},
            props={"label": "outside" if t > 0.7 else "place"},
        )
        prev = obs.place.id
        places.append(obs.place)

    # Tag the first place as a sugar puddle home.
    start = places[0]
    g.refresh_place(
        start.id,
        start.view_vec,
        props={"label": "puddle", "senses": {"sugar": {"value": 0.7}}},
    )
    end = places[-1]
    g.refresh_place(end.id, end.view_vec, props={"label": "outside"})

    tracker = RouteHomeTracker(fly_id="fly_a")
    for place in places:
        label = str((place.props or {}).get("label") or "place")
        tracker.observe(
            store=g,
            place=place,
            signals={"place_label": label, "energy": 1.0, "hunger": 0.0},
        )
    state = tracker.observe(
        store=g,
        place=end,
        signals={
            "place_label": "outside",
            "hunger": 0.95,
            "energy": 0.05,
            "sugar_smell": 0.4,
            "opening_strength": 0.9,
        },
    )
    assert state.mode == "route"
    assert state.home_id == start.id
    assert len(state.path) >= 2
    assert state.path[0] == end.id and state.path[-1] == start.id


def test_smell_trail_becomes_home_when_hungry() -> None:
    """Walk past an enclosed smell site, leave, then reverse-path to it when hungry.

    Smell is above trail_smell_min but below bookmark_sugar_smell so home comes
    from the session trail, not the bookmark path.
    """
    g = InMemoryPlaceGraph()
    food = g.create_place(_view(1), fly_id="fly_a", props={"label": "place"})
    mid = g.create_place(_view(2), fly_id="fly_a", props={"label": "place"})
    outside = g.create_place(_view(3), fly_id="fly_a", props={"label": "outside"})
    g.upsert_adjacent(food.id, mid.id, props={"turn": 0.2})
    g.upsert_adjacent(mid.id, outside.id, props={"turn": 0.5, "opening_strength": 0.9})

    tracker = RouteHomeTracker(fly_id="fly_a")
    tracker.observe(
        store=g,
        place=food,
        signals={
            "place_label": "place",
            "energy": 1.0,
            "sugar_smell": 0.15,
            "opening_strength": 0.1,
        },
    )
    assert tracker._peak_smell_id == food.id
    assert tracker._home_place_id is None  # below bookmark threshold
    tracker.observe(
        store=g,
        place=mid,
        signals={"place_label": "place", "energy": 1.0, "sugar_smell": 0.05, "opening_strength": 0.2},
    )
    tracker.observe(
        store=g,
        place=outside,
        signals={
            "place_label": "outside",
            "energy": 1.0,
            "sugar_smell": 0.0,
            "opening_strength": 0.95,
        },
    )
    state = tracker.observe(
        store=g,
        place=outside,
        signals={
            "place_label": "outside",
            "hunger": 0.9,
            "energy": 0.1,
            "sugar_smell": 0.05,
            "opening_strength": 0.95,
        },
    )
    assert state.mode == "route"
    assert state.home_id == food.id
    assert state.reason == "smell_trail"
    assert state.path[-1] == food.id
    assert state.next_id == mid.id


def test_rising_smell_blends_antenna_while_routing() -> None:
    g = InMemoryPlaceGraph()
    home = g.create_place(_view(1), fly_id="fly_a", props={"label": "place"})
    mid = g.create_place(_view(2), fly_id="fly_a", props={"label": "place"})
    outside = g.create_place(_view(3), fly_id="fly_a", props={"label": "outside"})
    g.upsert_adjacent(home.id, mid.id, props={"turn": 0.0})
    g.upsert_adjacent(mid.id, outside.id, props={"turn": 0.0})  # weak edge turns
    tracker = RouteHomeTracker(fly_id="fly_a")
    tracker.observe(
        store=g,
        place=home,
        signals={"place_label": "place", "energy": 1.0, "sugar_smell": 0.5, "opening_strength": 0.05},
    )
    tracker.observe(
        store=g,
        place=mid,
        signals={"place_label": "place", "energy": 1.0, "sugar_smell": 0.1, "opening_strength": 0.1},
    )
    # First hungry tick seeds last smell (still multi-hop so stay in route).
    tracker.observe(
        store=g,
        place=outside,
        signals={
            "place_label": "outside",
            "hunger": 0.9,
            "energy": 0.1,
            "sugar_smell": 0.10,
            "sugar_bearing": 0.0,
            "opening_strength": 0.2,
            "wall_hazard": 0.1,
        },
    )
    # Smell rises with clear right bearing → turn should lean right.
    state = tracker.observe(
        store=g,
        place=outside,
        signals={
            "place_label": "outside",
            "hunger": 0.9,
            "energy": 0.1,
            "sugar_smell": 0.25,
            "sugar_bearing": 0.9,
            "opening_strength": 0.2,
            "wall_hazard": 0.1,
        },
    )
    assert state.mode == "route"
    assert state.smell_delta > 0.0
    assert state.turn is not None and state.turn > 0.2


def test_wall_outside_steers_toward_opening_entrance() -> None:
    g = InMemoryPlaceGraph()
    chamber = g.create_place(_view(1), fly_id="fly_a", props={"label": "place"})
    outside = g.create_place(_view(2), fly_id="fly_a", props={"label": "outside"})
    # Outbound turn would reverse to -0.8; wall+opening should override toward opening.
    g.upsert_adjacent(chamber.id, outside.id, props={"turn": 0.8})
    tracker = RouteHomeTracker(fly_id="fly_a")
    tracker.observe(
        store=g,
        place=chamber,
        signals={"place_label": "place", "energy": 1.0, "sugar_smell": 0.4, "opening_strength": 0.1},
    )
    state = tracker.observe(
        store=g,
        place=outside,
        signals={
            "place_label": "outside",
            "hunger": 0.9,
            "energy": 0.1,
            "sugar_smell": 0.1,
            "wall_hazard": 0.7,
            "opening_strength": 0.6,
            "opening_bearing": -0.9,
        },
    )
    assert state.mode == "route"
    assert state.turn is not None
    assert state.turn < -0.4  # opening bearing, not reversed edge +0.8→-0.8 alone wait -0.8 is similar
    # Explicitly: must follow opening (-0.9 * 0.75 = -0.675), not ignore it.
    assert abs(state.turn - (-0.75 * 0.9)) < 0.05
    assert state.commit_degrees_left < 180.0  # first tick consumed some yaw budget


def test_wall_aware_path_prefers_opening_edge() -> None:
    """Two routes to home: wall-side vs opening/tunnel — pick the opening route."""
    from flybrain.thinking.route_home import shortest_path_wall_aware

    g = InMemoryPlaceGraph()
    start = g.create_place(_view(1), fly_id="fly_a", props={"label": "outside"})
    wall_hop = g.create_place(
        _view(2),
        fly_id="fly_a",
        props={"label": "outside", "senses": {"wall_hazard": {"value": 0.9}}},
    )
    open_hop = g.create_place(
        _view(3),
        fly_id="fly_a",
        props={"label": "outside", "senses": {"wall_hazard": {"value": 0.05}}},
    )
    home = g.create_place(_view(4), fly_id="fly_a", props={"label": "place"})
    g.upsert_adjacent(start.id, wall_hop.id, props={"turn": 0.1, "opening_strength": 0.0})
    g.upsert_adjacent(wall_hop.id, home.id, props={"turn": 0.1, "opening_strength": 0.0})
    g.upsert_adjacent(start.id, open_hop.id, props={"turn": -0.2, "opening_strength": 0.95})
    g.upsert_adjacent(open_hop.id, home.id, props={"turn": -0.2, "opening_strength": 0.8})

    def wall_of(pid):
        node = g.get(pid)
        senses = (node.props or {}).get("senses") or {} if node else {}
        row = senses.get("wall_hazard") or {}
        return float(row.get("value") or 0.0)

    path = shortest_path_wall_aware(
        g.edges(),
        start.id,
        home.id,
        wall_of=wall_of,
        wall_penalty=1.5,
        opening_bonus=0.75,
    )
    assert path is not None
    assert open_hop.id in path
    assert wall_hop.id not in path


def test_graph_food_prefers_first_experience_over_later_smell() -> None:
    """Hungry routing queries food senses and returns to the earliest food place."""
    from datetime import datetime, timedelta, timezone

    g = InMemoryPlaceGraph()
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    first = g.create_place(
        _view(1),
        fly_id="fly_a",
        props={
            "label": "puddle",
            "senses": {"sugar": {"value": 0.4, "last_seen_step": 10}},
        },
    )
    first.created_at = t0
    later = g.create_place(
        _view(2),
        fly_id="fly_a",
        props={
            "label": "place",
            "senses": {"sugar_smell": {"value": 0.9, "last_seen_step": 80}},
        },
    )
    later.created_at = t0 + timedelta(hours=1)
    outside = g.create_place(_view(3), fly_id="fly_a", props={"label": "outside"})
    g.upsert_adjacent(first.id, later.id, props={"turn": 0.1})
    g.upsert_adjacent(later.id, outside.id, props={"turn": 0.4, "opening_strength": 0.8})

    tracker = RouteHomeTracker(
        fly_id="fly_a",
        config=RouteHomeConfig(prefer_first_food=True, graph_food_enabled=True),
    )
    state = tracker.observe(
        store=g,
        place=outside,
        signals={"place_label": "outside", "hunger": 0.9, "energy": 0.1, "sugar_smell": 0.0},
    )
    assert state.mode == "route"
    assert state.home_id == first.id
    assert state.reason == "graph_food"
    assert state.food_kind == "sugar"
    assert [h["src_id"] for h in state.path_edges][0] == outside.id
    assert state.path[-1] == first.id


def test_find_food_places_orders_by_first_step() -> None:
    g = InMemoryPlaceGraph()
    late = g.create_place(
        _view(1),
        fly_id="fly_a",
        props={
            "label": "place",
            "senses": {"sugar_smell": {"value": 0.5, "last_seen_step": 50}},
        },
    )
    early = g.create_place(
        _view(2),
        fly_id="fly_a",
        props={"label": "puddle", "senses": {"sugar": {"value": 0.3, "last_seen_step": 5}}},
    )
    hits = g.find_food_places(fly_id="fly_a", min_value=0.05, limit=8)
    assert [h.place.id for h in hits] == [early.id, late.id]
    assert hits[0].kind == "sugar"


def test_hunger_ignores_generic_odor_places() -> None:
    """Hungry graph food search must not home on odor-only (non-sugar) places."""
    g = InMemoryPlaceGraph()
    odor_only = g.create_place(
        _view(1),
        fly_id="fly_a",
        props={
            "label": "place",
            "senses": {"odor": {"value": 0.9, "last_seen_step": 3}},
        },
    )
    chamber = g.create_place(_view(2), fly_id="fly_a", props={"label": "place"})
    outside = g.create_place(_view(3), fly_id="fly_a", props={"label": "outside"})
    g.upsert_adjacent(odor_only.id, outside.id, props={"turn": 0.3})
    g.upsert_adjacent(chamber.id, outside.id, props={"turn": 0.4})
    hits = g.find_food_places(fly_id="fly_a", min_value=0.05, limit=8)
    assert hits == []
    tracker = RouteHomeTracker(fly_id="fly_a")
    tracker._chamber_anchor_id = chamber.id
    state = tracker.observe(
        store=g,
        place=outside,
        signals={
            "place_label": "outside",
            "hunger": 0.9,
            "energy": 0.1,
            "odor": 0.9,
            "sugar_smell": 0.0,
        },
    )
    assert state.mode == "route"
    assert state.home_id == chamber.id
    assert state.reason != "graph_food"
    assert state.home_id != odor_only.id


def test_route_path_cache_reuses_without_requery() -> None:
    """Second observe on the same home reuses trimmed cached path hops."""
    g = InMemoryPlaceGraph()
    home = g.create_place(
        _view(1),
        fly_id="fly_a",
        props={"label": "puddle", "senses": {"sugar": {"value": 0.8}}},
    )
    mid = g.create_place(_view(2), fly_id="fly_a", props={"label": "place"})
    outside = g.create_place(_view(3), fly_id="fly_a", props={"label": "outside"})
    g.upsert_adjacent(home.id, mid.id, props={"turn": 0.2})
    g.upsert_adjacent(mid.id, outside.id, props={"turn": 0.5})

    tracker = RouteHomeTracker(fly_id="fly_a")
    first = tracker.observe(
        store=g,
        place=outside,
        signals={"place_label": "outside", "hunger": 0.9, "energy": 0.1, "sugar_smell": 0.0},
    )
    assert first.mode == "route"
    assert first.path == [outside.id, mid.id, home.id]
    assert tracker._path_cache is not None

    # Move to mid — should trim cache, not lose home.
    second = tracker.observe(
        store=g,
        place=mid,
        signals={"place_label": "place", "hunger": 0.9, "energy": 0.1, "sugar_smell": 0.1},
    )
    assert second.home_id == home.id
    assert second.path[0] == mid.id
    assert second.path[-1] == home.id


def test_commit_turn_holds_sign_until_hops_progress() -> None:
    """Opposite edge turns stay locked until ≥180° and hops drop (or max 360°)."""
    tracker = RouteHomeTracker(
        config=RouteHomeConfig(
            min_turn_degrees=180,
            max_turn_degrees=360,
            commit_turn=0.8,
            deg_per_unit_turn=90,
        ),
        fly_id="fly_a",
    )
    first = tracker._commit_turn(0.55, hops_left=45)
    assert first == 0.55
    # Place-flip feeds the opposite edge — keep + even after 180° if hops stuck.
    for _ in range(3):
        assert tracker._commit_turn(-0.55, hops_left=45) == 0.8
    assert tracker._commit_yaw >= 180.0
    assert tracker._commit_sign > 0.0
    # Hop progress after min yaw unlocks a reverse.
    unlocked = tracker._commit_turn(-0.55, hops_left=44)
    assert unlocked == -0.55


def test_commit_turn_releases_at_max_when_hops_stuck() -> None:
    tracker = RouteHomeTracker(
        config=RouteHomeConfig(
            min_turn_degrees=180,
            max_turn_degrees=360,
            commit_turn=0.8,
            commit_escape_ticks=2,
            commit_escape_turn=0.55,
            deg_per_unit_turn=90,
        ),
        fly_id="fly_a",
    )
    tracker._commit_turn(0.8, hops_left=45)  # 72°
    # Keep resisting flips until accumulated yaw crosses the 360° cap.
    for _ in range(4):
        assert tracker._commit_turn(-0.8, hops_left=45) == 0.8
    assert tracker._commit_yaw >= 360.0 - 1e-6
    # Max-yaw with hops stuck → soft escape (no immediate opposite pivot re-arm).
    freed = tracker._commit_turn(-0.8, hops_left=45)
    assert freed == -0.55
    assert tracker._commit_sign == 0.0
    assert tracker._commit_escaped_this_tick
    second = tracker._commit_turn(-0.8, hops_left=45)
    assert second == -0.55
    # Escape expired → opposite raw may arm a new commit.
    armed = tracker._commit_turn(-0.8, hops_left=45)
    assert armed == -0.8
    assert tracker._commit_sign < 0.0


def test_commit_keeps_forward_crawl_when_resisting_opposite() -> None:
    tracker = RouteHomeTracker(
        config=RouteHomeConfig(
            min_turn_degrees=180,
            max_turn_degrees=360,
            commit_turn=0.8,
            commit_forward_min=0.3,
            pivot_turn=0.8,
            cruise_forward=0.55,
            deg_per_unit_turn=90,
        ),
        fly_id="fly_a",
    )
    tracker._commit_turn(0.55, hops_left=30)
    turn = tracker._commit_turn(-0.55, hops_left=30)
    assert turn == 0.8
    assert tracker._commit_sign > 0.0
    forward = tracker._route_forward(turn, smell_delta=0.0)
    assert forward >= 0.3


def test_route_observe_crawls_forward_under_commit() -> None:
    g = InMemoryPlaceGraph()
    a = g.create_place(_view(1), fly_id="fly_a", props={"label": "outside"})
    b = g.create_place(_view(2), fly_id="fly_a", props={"label": "outside"})
    mid = g.create_place(_view(3), fly_id="fly_a", props={"label": "place"})
    home = g.create_place(
        _view(4), fly_id="fly_a", props={"label": "puddle", "senses": {"sugar": {"value": 0.9}}}
    )
    g.upsert_adjacent(a.id, b.id, props={"turn": 0.55})
    g.upsert_adjacent(b.id, mid.id, props={"turn": -0.55})
    g.upsert_adjacent(mid.id, home.id, props={"turn": 0.2})

    tracker = RouteHomeTracker(
        config=RouteHomeConfig(
            min_turn_degrees=180,
            max_turn_degrees=360,
            commit_turn=0.8,
            commit_forward_min=0.3,
            deg_per_unit_turn=90,
        ),
        fly_id="fly_a",
    )
    tracker._home_place_id = home.id
    first = tracker.observe(
        store=g,
        place=a,
        signals={"place_label": "outside", "hunger": 0.9, "energy": 0.1, "sugar_smell": 0.1},
    )
    assert first.mode == "route"
    assert first.turn is not None and first.turn > 0.0
    # Thrash: now on b where the next hop wants the opposite turn.
    second = tracker.observe(
        store=g,
        place=b,
        signals={"place_label": "outside", "hunger": 0.9, "energy": 0.1, "sugar_smell": 0.1},
    )
    assert second.mode == "route"
    assert second.turn is not None and second.turn > 0.0  # commit holds +
    assert second.forward >= 0.3


def test_no_home_starts_landmark_scan() -> None:
    g = InMemoryPlaceGraph()
    outside = g.create_place(_view(1), fly_id="fly_a", props={"label": "outside"})
    tracker = RouteHomeTracker(
        config=RouteHomeConfig(
            landmark_scan=True,
            scan_step_degrees=20,
            scan_span_degrees=360,
            scan_turn=0.12,
        ),
        fly_id="fly_a",
    )
    state = tracker.observe(
        store=g,
        place=outside,
        signals={"place_label": "outside", "hunger": 0.9, "energy": 0.1, "sugar_smell": 0.0},
    )
    assert state.mode == "scan"
    assert state.reason == "no_home"
    assert state.forward == 0.0
    assert state.turn == 0.12
    assert state.scan_yaw_degrees == 20.0
    second = tracker.observe(
        store=g,
        place=outside,
        signals={"place_label": "outside", "hunger": 0.9, "energy": 0.1, "sugar_smell": 0.0},
    )
    assert second.mode == "scan"
    assert second.turn == 0.12  # continuous one-way body spin
    assert second.scan_yaw_degrees == 40.0


def test_landmark_scan_wraps_full_circle() -> None:
    g = InMemoryPlaceGraph()
    outside = g.create_place(_view(1), fly_id="fly_a", props={"label": "outside"})
    tracker = RouteHomeTracker(
        config=RouteHomeConfig(scan_step_degrees=20, scan_span_degrees=360, scan_turn=0.12),
        fly_id="fly_a",
    )
    yaw = 0.0
    for _ in range(18):  # 18 × 20° = 360° → wrap to 0
        state = tracker.observe(
            store=g,
            place=outside,
            signals={"place_label": "outside", "hunger": 0.9, "energy": 0.1, "sugar_smell": 0.0},
        )
        assert state.mode == "scan"
        assert state.turn == 0.12
        yaw = float(state.scan_yaw_degrees)
    assert yaw == 0.0
    again = tracker.observe(
        store=g,
        place=outside,
        signals={"place_label": "outside", "hunger": 0.9, "energy": 0.1, "sugar_smell": 0.0},
    )
    assert again.scan_yaw_degrees == 20.0


def test_unreachable_home_scans() -> None:
    g = InMemoryPlaceGraph()
    home = g.create_place(
        _view(1),
        fly_id="fly_a",
        props={"label": "puddle", "senses": {"sugar": {"value": 0.9}}},
    )
    outside = g.create_place(_view(2), fly_id="fly_a", props={"label": "outside"})
    # No ADJACENT edge — home known via graph food but unreachable.
    tracker = RouteHomeTracker(fly_id="fly_a")
    # Seed home bookmark by visiting with smell (no need for graph food query).
    tracker._home_place_id = home.id
    state = tracker.observe(
        store=g,
        place=outside,
        signals={"place_label": "outside", "hunger": 0.9, "energy": 0.1, "sugar_smell": 0.0},
    )
    assert state.mode == "scan"
    assert state.reason in ("unreachable", "no_home")
    assert state.turn is not None and state.forward == 0.0


def test_missing_edge_props_scans() -> None:
    g = InMemoryPlaceGraph()
    home = g.create_place(
        _view(1),
        fly_id="fly_a",
        props={"label": "puddle", "senses": {"sugar": {"value": 0.8}}},
    )
    outside = g.create_place(_view(2), fly_id="fly_a", props={"label": "outside"})
    g.upsert_adjacent(home.id, outside.id, props={})  # edge exists but no turn/opening
    tracker = RouteHomeTracker(fly_id="fly_a")
    state = tracker.observe(
        store=g,
        place=outside,
        signals={
            "place_label": "outside",
            "hunger": 0.9,
            "energy": 0.1,
            "sugar_smell": 0.0,
            "opening_strength": 0.0,
        },
    )
    assert state.mode == "scan"
    assert state.reason == "missing_edge"


def test_scan_stops_on_opening_landmark() -> None:
    g = InMemoryPlaceGraph()
    outside = g.create_place(_view(1), fly_id="fly_a", props={"label": "outside"})
    tracker = RouteHomeTracker(
        config=RouteHomeConfig(scan_landmark_opening=0.35),
        fly_id="fly_a",
    )
    tracker.observe(
        store=g,
        place=outside,
        signals={
            "place_label": "outside",
            "hunger": 0.9,
            "energy": 0.1,
            "opening_strength": 0.1,
        },
    )
    found = tracker.observe(
        store=g,
        place=outside,
        signals={
            "place_label": "outside",
            "hunger": 0.9,
            "energy": 0.1,
            "opening_strength": 0.5,
        },
    )
    assert found.mode == "handoff"
    assert found.reason == "scan_landmark"
    assert found.turn is None
