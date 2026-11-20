"""Spatial place graph: location node = view vector of the chamber box."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from flybrain.memory.place_graph import (
    WALL_DIRS,
    InMemoryPlaceGraph,
    Neo4jPlaceGraph,
    PlaceNode,
    default_walls,
    encode_place_view,
    place_similarity,
)
from flybrain.phase4 import PATCH_VEC_DIM


def _view(seed: int, dim: int = PATCH_VEC_DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _patch_like(seed: int, dim: int = PATCH_VEC_DIM) -> np.ndarray:
    """Non-negative patch-style vector (same regime as stereo patch_vec)."""
    rng = np.random.default_rng(seed)
    return np.abs(rng.normal(0.3, 0.22, size=dim)).astype(np.float32)


def test_default_walls_has_four_cardinals() -> None:
    walls = default_walls()
    assert tuple(walls.keys()) == WALL_DIRS
    for d in WALL_DIRS:
        assert walls[d]["present"] is True
        assert walls[d]["familiarity"] == 0.0


def test_upsert_creates_place_with_wall_details() -> None:
    g = InMemoryPlaceGraph()
    node = g.upsert_place(_view(1), fly_id="fly_a", step=10)
    assert node.id == 1
    assert node.view_vec.shape == (PATCH_VEC_DIM,)
    assert set(node.walls) == set(WALL_DIRS)
    assert node.props["last_step"] == 10
    assert g.edges() == []


def test_similar_view_merges_same_place() -> None:
    g = InMemoryPlaceGraph()
    a = g.upsert_place(_view(7), fly_id="fly_a", match_min=0.99)
    # Identical vector → same node.
    b = g.upsert_place(_view(7), fly_id="fly_a", match_min=0.99)
    assert a.id == b.id
    assert len(g.list_places(fly_id="fly_a")) == 1


def test_distinct_views_create_separate_places() -> None:
    g = InMemoryPlaceGraph()
    a = g.upsert_place(_view(1), fly_id="fly_a", match_min=0.99)
    b = g.upsert_place(_view(2), fly_id="fly_a", match_min=0.99)
    assert a.id != b.id
    assert len(g.list_places(fly_id="fly_a")) == 2


def test_encode_place_view_is_centered_unit() -> None:
    raw = _patch_like(11)
    enc = encode_place_view(raw)
    assert enc.shape == (PATCH_VEC_DIM,)
    assert abs(float(enc.mean())) < 1e-5
    assert abs(float(np.linalg.norm(enc)) - 1.0) < 1e-5


def test_place_similarity_separates_nonneg_patch_views() -> None:
    """Raw cosine on nonneg patches is ~0.99; centered cosine must discriminate."""
    a = _patch_like(1)
    b = _patch_like(2)
    raw_dot = float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))
    assert raw_dot > 0.6  # raw cosine is uninformative in this regime
    assert place_similarity(a, b) < 0.5
    assert place_similarity(a, a) > 0.999


def test_nonneg_patch_views_create_separate_places() -> None:
    g = InMemoryPlaceGraph()
    a = g.upsert_place(_patch_like(1), fly_id="fly_a", match_min=0.97)
    b = g.upsert_place(_patch_like(2), fly_id="fly_a", match_min=0.97)
    assert a.id != b.id
    assert len(g.list_places(fly_id="fly_a")) == 2


def test_near_duplicate_nonneg_patch_merges() -> None:
    g = InMemoryPlaceGraph()
    base = _patch_like(7)
    near = np.clip(base + np.random.default_rng(0).normal(0, 0.01, size=base.shape), 0, None)
    a = g.upsert_place(base, fly_id="fly_a", match_min=0.97)
    b = g.upsert_place(near.astype(np.float32), fly_id="fly_a", match_min=0.97)
    assert a.id == b.id
    assert len(g.list_places(fly_id="fly_a")) == 1


def test_nearest_ranks_by_similarity() -> None:
    g = InMemoryPlaceGraph()
    g.upsert_place(_view(1), fly_id="fly_a")
    g.upsert_place(_view(99), fly_id="fly_a")
    hits = g.nearest(_view(1), fly_id="fly_a", top_k=2)
    assert len(hits) == 2
    assert hits[0][1] >= hits[1][1]
    assert hits[0][1] > 0.99


def test_update_wall_familiarity_on_place_node() -> None:
    g = InMemoryPlaceGraph()
    node = g.upsert_place(_view(3), fly_id="fly_a")
    updated = g.update_wall(node.id, "north", familiarity=0.4, last_seen_step=50, notes="seen")
    assert updated is not None
    assert updated.walls["north"]["familiarity"] == 0.4
    assert updated.walls["north"]["last_seen_step"] == 50
    assert updated.walls["east"]["familiarity"] == 0.0  # other walls untouched
    got = g.get(node.id)
    assert got is not None
    assert got.walls["north"]["notes"] == "seen"


def test_place_node_roundtrip_dict() -> None:
    node = PlaceNode(
        id=9,
        fly_id="fly_a",
        view_vec=encode_place_view(_view(5)),
        walls=default_walls(),
        props={"label": "box_center"},
    )
    node.walls["west"]["familiarity"] = 0.25
    restored = PlaceNode.from_dict(node.to_dict())
    assert restored.id == 9
    assert restored.walls["west"]["familiarity"] == 0.25
    assert restored.props["label"] == "box_center"
    assert np.allclose(restored.view_vec, node.view_vec, atol=1e-5)


def test_update_wall_rejects_bad_direction() -> None:
    g = InMemoryPlaceGraph()
    node = g.upsert_place(_view(4))
    try:
        g.update_wall(node.id, "up")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_decay_mapped_wall_hazard_only_when_mapped() -> None:
    from flybrain.memory.place_graph import decay_mapped_wall_hazard

    raw = 0.8
    assert decay_mapped_wall_hazard(raw, mapped=False, familiarity=1.0) == raw
    expected = raw * max(0.15, 1.0 - 0.85 * 0.5)
    decayed = decay_mapped_wall_hazard(raw, mapped=True, familiarity=0.5, strength=0.85, floor=0.15)
    assert decayed < raw
    assert abs(decayed - expected) < 1e-6


def test_observe_place_creates_then_bumps_familiarity_on_revisit() -> None:
    from flybrain.memory.place_graph import observe_place

    g = InMemoryPlaceGraph()
    view = _view(11)
    first = observe_place(g, view, fly_id="fly_a", step=1, familiarity_bump=0.2)
    assert first.is_new is True
    assert first.mean_familiarity == 0.0
    assert set(first.place.walls) == set(WALL_DIRS)

    second = observe_place(g, view, fly_id="fly_a", step=2, familiarity_bump=0.2)
    assert second.is_new is False
    assert second.place.id == first.place.id
    assert second.mean_familiarity == 0.2
    for d in WALL_DIRS:
        assert second.place.walls[d]["familiarity"] == 0.2
        assert second.place.walls[d]["last_seen_step"] == 2


def test_identical_view_from_different_references_creates_distinct_places() -> None:
    """Tree-room vs bare-room → same lookalike chamber = two Places + edges."""
    from flybrain.memory.place_graph import observe_place

    g = InMemoryPlaceGraph()
    tree = observe_place(g, _view(1), fly_id="fly_a", step=1, context_enabled=True)
    bare = observe_place(g, _view(2), fly_id="fly_a", step=2, context_enabled=True)
    assert tree.place.id != bare.place.id

    twin = _view(50)
    from_tree = observe_place(
        g,
        twin,
        fly_id="fly_a",
        step=3,
        from_place_id=tree.place.id,
        context_enabled=True,
    )
    assert from_tree.is_new is True
    assert from_tree.match_reason == "created"
    assert from_tree.edge_created is True
    assert g.has_adjacent(tree.place.id, from_tree.place.id)

    from_bare = observe_place(
        g,
        twin,
        fly_id="fly_a",
        step=4,
        from_place_id=bare.place.id,
        context_enabled=True,
    )
    assert from_bare.is_new is True
    assert from_bare.place.id != from_tree.place.id
    assert g.has_adjacent(bare.place.id, from_bare.place.id)
    assert len(g.list_places(fly_id="fly_a")) == 4


def test_adjacent_edge_reuses_place_on_same_path() -> None:
    from flybrain.memory.place_graph import observe_place

    g = InMemoryPlaceGraph()
    lobby = observe_place(g, _view(1), fly_id="fly_a", step=1)
    room = observe_place(
        g,
        _view(9),
        fly_id="fly_a",
        step=2,
        from_place_id=lobby.place.id,
        context_enabled=True,
    )
    assert room.edge_created is True

    again = observe_place(
        g,
        _view(9),
        fly_id="fly_a",
        step=3,
        from_place_id=lobby.place.id,
        context_enabled=True,
    )
    assert again.is_new is False
    assert again.match_reason == "adjacent"
    assert again.place.id == room.place.id


def test_same_place_micro_motion_keeps_node() -> None:
    from flybrain.memory.place_graph import observe_place

    g = InMemoryPlaceGraph()
    first = observe_place(g, _view(7), fly_id="fly_a", step=1)
    near = _view(7)  # identical
    stay = observe_place(
        g,
        near,
        fly_id="fly_a",
        step=2,
        from_place_id=first.place.id,
        context_enabled=True,
    )
    assert stay.is_new is False
    assert stay.match_reason == "same_place"
    assert stay.place.id == first.place.id
    assert stay.edge_created is False


def test_should_leave_same_place_on_novelty_and_landmarks() -> None:
    from flybrain.memory.place_graph import should_leave_same_place

    leave, why = should_leave_same_place(visual_novelty=0.4, novelty_split_min=0.25)
    assert leave is True and why == "novelty"
    leave, why = should_leave_same_place(new_stimulus=True)
    assert leave is True and why == "new_stimulus"
    leave, why = should_leave_same_place(opening=0.9, prev_opening=0.5, opening_delta_min=0.15)
    assert leave is True and why == "opening_delta"
    leave, why = should_leave_same_place(wall=0.8, prev_wall=0.1, wall_delta_min=0.15)
    assert leave is True and why == "wall_delta"
    leave, why = should_leave_same_place(visual_novelty=0.1, opening=0.5, prev_opening=0.5)
    assert leave is False and why == ""


def test_leave_same_place_creates_new_node_on_identical_view() -> None:
    from flybrain.memory.place_graph import observe_place

    g = InMemoryPlaceGraph()
    first = observe_place(g, _view(7), fly_id="fly_a", step=1)
    split = observe_place(
        g,
        _view(7),
        fly_id="fly_a",
        step=2,
        from_place_id=first.place.id,
        context_enabled=True,
        leave_same_place=True,
        split_reason="opening_delta",
    )
    assert split.is_new is True
    assert split.match_reason == "created"
    assert split.split_reason == "opening_delta"
    assert split.place.id != first.place.id
    assert split.edge_created is True
    assert g.has_adjacent(first.place.id, split.place.id)


def test_leave_same_place_can_take_adjacent_hop() -> None:
    from flybrain.memory.place_graph import observe_place

    g = InMemoryPlaceGraph()
    lobby = observe_place(g, _view(1), fly_id="fly_a", step=1)
    room = observe_place(
        g,
        _view(2),
        fly_id="fly_a",
        step=2,
        from_place_id=lobby.place.id,
        context_enabled=True,
    )
    # Landmark split while looking at the known adjacent room → hop there.
    hop = observe_place(
        g,
        _view(2),
        fly_id="fly_a",
        step=3,
        from_place_id=lobby.place.id,
        context_enabled=True,
        leave_same_place=True,
        split_reason="opening_delta",
    )
    assert hop.place.id == room.place.id
    assert hop.match_reason == "adjacent"
    assert hop.split_reason == "opening_delta"


def test_mapped_feedback_splits_on_opening_delta() -> None:
    from flybrain.memory.place_graph import PlaceGraphConfig
    from flybrain.sensors.mapped_feedback import MappedSensorFeedback

    g = InMemoryPlaceGraph()
    cfg = PlaceGraphConfig(
        enabled=True,
        backend="in_memory",
        match_min=0.97,
        split_on_novelty=False,
        split_on_landmark=True,
        opening_delta_min=0.15,
    )
    fb = MappedSensorFeedback(store=g, config=cfg, fly_id="fly_a")
    view = _view(7)
    first = fb.sense(
        raw={"opening_strength": 0.95, "wall_hazard": 0.0},
        view_vec=view,
        step=1,
    )
    assert first.match_reason in ("created", "view_only", "same_place")
    stuck = fb.sense(
        raw={"opening_strength": 0.95, "wall_hazard": 0.0},
        view_vec=view,
        step=2,
    )
    assert stuck.place is not None and first.place is not None
    assert stuck.place.id == first.place.id
    assert stuck.match_reason == "same_place"
    split = fb.sense(
        raw={"opening_strength": 0.50, "wall_hazard": 0.0},
        view_vec=view,
        step=3,
    )
    assert split.place is not None
    assert split.place.id != first.place.id
    assert split.split_reason == "opening_delta"
    assert split.is_new is True


def _walk(a: np.ndarray, b: np.ndarray, steps: int) -> list[np.ndarray]:
    return [encode_place_view((1.0 - t) * a + t * b) for t in np.linspace(0.0, 1.0, steps)]


def test_revisits_do_not_move_the_place_view() -> None:
    g = InMemoryPlaceGraph()
    node = g.upsert_place(_view(1), fly_id="fly_a", match_min=0.9)
    anchor = node.view_vec.copy()
    nudged = encode_place_view(_view(1) + 0.05 * _view(2))
    g.upsert_place(nudged, fly_id="fly_a", match_min=0.9)
    g.refresh_place(node.id, nudged)
    assert np.allclose(g.get(node.id).view_vec, anchor)


def test_gradual_walk_leaves_the_start_place() -> None:
    """Box → tunnel → outside in tiny steps must not stay one place (no view drift)."""
    from flybrain.memory.place_graph import observe_place

    g = InMemoryPlaceGraph()
    inside, outside = _view(1), _view(2)
    prev = None
    seen = []
    for step, view in enumerate(_walk(inside, outside, 200)):
        obs = observe_place(g, view, fly_id="fly_a", step=step, from_place_id=prev)
        prev = obs.place.id
        seen.append(prev)
    assert len(set(seen)) > 2
    assert place_similarity(g.get(seen[-1]).view_vec, outside) >= 0.97
    assert place_similarity(g.get(seen[0]).view_vec, inside) > 0.999


def test_find_places_filters_by_label() -> None:
    g = InMemoryPlaceGraph()
    g.create_place(_view(1), fly_id="fly_a", props={"label": "outside"})
    g.create_place(_view(2), fly_id="fly_a", props={"label": "puddle"})
    found = g.find_places(fly_id="fly_a", label="puddle")
    assert len(found) == 1
    assert found[0].props["label"] == "puddle"


def test_neo4j_shortest_path_uses_undirected_cypher(monkeypatch) -> None:
    driver = _FakeNeo4jDriver()
    driver.vector_rows = [{"place_ids": ["a", "b", "c"]}]
    # shortest_path uses _query, not vector index — override after construct.
    graph_database = SimpleNamespace(driver=lambda *args, **kwargs: driver)
    monkeypatch.setitem(sys.modules, "neo4j", SimpleNamespace(GraphDatabase=graph_database))
    graph = Neo4jPlaceGraph(password="test-password")
    captured: list[tuple[str, dict]] = []

    def _query(cypher, params=None):
        captured.append((str(cypher), dict(params or {})))
        if "shortestPath" in str(cypher):
            return [{"place_ids": ["a", "b", "c"]}]
        return []

    graph._query = _query  # type: ignore[method-assign]
    path = graph.shortest_path("a", "c", undirected=True, max_hops=8)
    assert path == ["a", "b", "c"]
    assert "shortestPath" in captured[-1][0]
    assert "-[:ADJACENT*1..8]-" in captured[-1][0]
    graph.close()


def test_walking_back_reuses_places_via_reverse_edges() -> None:
    from flybrain.memory.place_graph import observe_place

    g = InMemoryPlaceGraph()
    box = observe_place(g, _view(1), fly_id="fly_a", step=0)
    out = observe_place(g, _view(2), fly_id="fly_a", step=1, from_place_id=box.place.id)
    back = observe_place(g, _view(1), fly_id="fly_a", step=2, from_place_id=out.place.id)
    assert back.place.id == box.place.id
    assert back.match_reason == "adjacent"
    assert len(g.list_places(fly_id="fly_a")) == 2


def test_place_graph_config_from_dict() -> None:
    from flybrain.memory.place_graph import PlaceGraphConfig

    cfg = PlaceGraphConfig.from_dict(
        {
            "backend": "neo4j",
            "match_min": 0.9,
            "hazard_decay_floor": 0.2,
            "context_enabled": False,
            "context_top_k": 4,
            "split_on_novelty": False,
            "novelty_split_min": 0.4,
            "opening_delta_min": 0.2,
        }
    )
    assert cfg.backend == "neo4j"
    assert cfg.match_min == 0.9
    assert cfg.hazard_decay_floor == 0.2
    assert cfg.enabled is True
    assert cfg.context_enabled is False
    assert cfg.context_top_k == 4
    assert cfg.split_on_novelty is False
    assert cfg.novelty_split_min == 0.4
    assert cfg.opening_delta_min == 0.2


class _FakeNeo4jDriver:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.closed = False
        self.vector_rows: list[dict] = []

    def execute_query(self, cypher, *, parameters_, database_):
        self.calls.append((str(cypher), dict(parameters_)))
        if "db.index.vector.queryNodes" in str(cypher):
            return self.vector_rows, None, []
        return [], None, []

    def close(self) -> None:
        self.closed = True


def _neo_record() -> dict:
    return {
        "place_id": "place-001",
        "fly_id": "fly_a",
        "view_vec": _view(1).tolist(),
        "props_json": '{"label": "box"}',
        "created_at": "2026-07-26T09:30:00+00:00",
        "updated_at": "2026-07-26T09:31:00+00:00",
        "walls": [
            {
                "direction": direction,
                "present": True,
                "label": direction,
                "familiarity": 0.25 if direction == "north" else 0.0,
                "last_seen_step": 10 if direction == "north" else None,
                "notes": "",
                "extra_json": "{}",
            }
            for direction in WALL_DIRS
        ],
        "score": 0.97,
    }


def test_neo4j_creates_constraints_and_vector_index(monkeypatch) -> None:
    driver = _FakeNeo4jDriver()
    graph_database = SimpleNamespace(driver=lambda *args, **kwargs: driver)
    monkeypatch.setitem(sys.modules, "neo4j", SimpleNamespace(GraphDatabase=graph_database))

    graph = Neo4jPlaceGraph(password="test-password")
    cypher = "\n".join(query for query, _params in driver.calls)
    assert "CREATE CONSTRAINT place_id_unique" in cypher
    assert "CREATE CONSTRAINT wall_id_unique" in cypher
    assert "CREATE VECTOR INDEX place_view_vec" in cypher
    assert f"`vector.dimensions`: {PATCH_VEC_DIM}" in cypher
    assert "db.awaitIndex" in cypher
    graph.close()
    assert driver.closed is True


def test_neo4j_nearest_uses_vector_index_and_rebuilds_wall_details(monkeypatch) -> None:
    driver = _FakeNeo4jDriver()
    driver.vector_rows = [_neo_record()]
    graph_database = SimpleNamespace(driver=lambda *args, **kwargs: driver)
    monkeypatch.setitem(sys.modules, "neo4j", SimpleNamespace(GraphDatabase=graph_database))

    graph = Neo4jPlaceGraph(password="test-password")
    hits = graph.nearest(_view(1), fly_id="fly_a", top_k=1)
    assert len(hits) == 1
    node, score = hits[0]
    assert node.id == "place-001"
    assert score == 0.97
    assert node.walls["north"]["familiarity"] == 0.25
    query, params = driver.calls[-1]
    assert "db.index.vector.queryNodes" in query
    assert params["fly_id"] == "fly_a"
    assert len(params["view_vec"]) == PATCH_VEC_DIM


def test_neo4j_save_materializes_four_wall_nodes() -> None:
    graph = object.__new__(Neo4jPlaceGraph)
    graph.view_dim = PATCH_VEC_DIM
    captured: list[tuple[str, dict]] = []
    graph._query = lambda query, params=None: captured.append((query, params or {})) or []
    expected = PlaceNode(
        id="place-001",
        fly_id="fly_a",
        view_vec=_view(1),
        walls=default_walls(),
    )
    graph.get = lambda place_id: expected

    saved = graph._save_place(
        place_id="place-001",
        fly_id="fly_a",
        view_vec=_view(1),
        walls=default_walls(),
        props={"label": "box"},
    )
    assert saved is expected
    query, params = captured[0]
    assert "MERGE (w:Wall" in query
    assert "MERGE (p)-[:HAS_WALL" in query
    assert {wall["direction"] for wall in params["walls"]} == set(WALL_DIRS)


def test_neo4j_find_food_places_queries_has_sense(monkeypatch) -> None:
    driver = _FakeNeo4jDriver()
    graph_database = SimpleNamespace(driver=lambda *args, **kwargs: driver)
    monkeypatch.setitem(sys.modules, "neo4j", SimpleNamespace(GraphDatabase=graph_database))
    graph = Neo4jPlaceGraph(password="test-password")
    captured: list[tuple[str, dict]] = []

    def _query(cypher, params=None):
        captured.append((str(cypher), dict(params or {})))
        if "HAS_SENSE" in str(cypher):
            row = _neo_record()
            row["sense_kind"] = "sugar"
            row["sense_value"] = 0.7
            row["sense_step"] = 12
            row["props"] = row["walls"]
            return [row]
        return []

    graph._query = _query  # type: ignore[method-assign]
    hits = graph.find_food_places(fly_id="fly_a", min_value=0.05, limit=4)
    assert len(hits) == 1
    assert hits[0].kind == "sugar"
    assert hits[0].value == 0.7
    assert hits[0].first_step == 12
    assert "HAS_SENSE" in captured[-1][0]
    assert "sugar" in captured[-1][1]["kinds"] or "sugar" in str(captured[-1][1]["kinds"])
    graph.close()


def test_neo4j_shortest_path_with_edges_returns_props(monkeypatch) -> None:
    driver = _FakeNeo4jDriver()
    graph_database = SimpleNamespace(driver=lambda *args, **kwargs: driver)
    monkeypatch.setitem(sys.modules, "neo4j", SimpleNamespace(GraphDatabase=graph_database))
    graph = Neo4jPlaceGraph(password="test-password")

    def _query(cypher, params=None):
        if "shortestPath" in str(cypher):
            return [
                {
                    "place_ids": ["out", "mid", "food"],
                    "edge_rows": [
                        {
                            "src_id": "mid",
                            "dst_id": "out",
                            "props_json": '{"turn": 0.5, "opening_strength": 0.9}',
                        },
                        {
                            "src_id": "food",
                            "dst_id": "mid",
                            "props_json": '{"turn": 0.2}',
                        },
                    ],
                }
            ]
        return []

    graph._query = _query  # type: ignore[method-assign]
    detailed = graph.shortest_path_with_edges("out", "food", undirected=True, max_hops=8)
    assert detailed is not None
    ids, hops = detailed
    assert ids == ["out", "mid", "food"]
    assert len(hops) == 2
    assert hops[0].reversed is True  # mid→out edge followed out→mid
    assert hops[0].props["turn"] == 0.5
    graph.close()


def test_neo4j_find_places_filters_label_in_cypher(monkeypatch) -> None:
    driver = _FakeNeo4jDriver()
    graph_database = SimpleNamespace(driver=lambda *args, **kwargs: driver)
    monkeypatch.setitem(sys.modules, "neo4j", SimpleNamespace(GraphDatabase=graph_database))
    graph = Neo4jPlaceGraph(password="test-password")
    captured: list[tuple[str, dict]] = []

    def _query(cypher, params=None):
        captured.append((str(cypher), dict(params or {})))
        if "label_marker" in (params or {}):
            row = _neo_record()
            row["props_json"] = '{"label": "puddle"}'
            return [row]
        return []

    graph._query = _query  # type: ignore[method-assign]
    found = graph.find_places(fly_id="fly_a", label="puddle", limit=4)
    assert captured
    assert "CONTAINS $label_marker" in captured[-1][0]
    assert captured[-1][1]["label_marker"] == '"label": "puddle"'
    assert len(found) == 1
    graph.close()
