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
    raw_dot = float(
        np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8))
    )
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
    decayed = decay_mapped_wall_hazard(
        raw, mapped=True, familiarity=0.5, strength=0.85, floor=0.15
    )
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


def test_place_graph_config_from_dict() -> None:
    from flybrain.memory.place_graph import PlaceGraphConfig

    cfg = PlaceGraphConfig.from_dict(
        {
            "backend": "neo4j",
            "match_min": 0.9,
            "hazard_decay_floor": 0.2,
            "context_enabled": False,
            "context_top_k": 4,
        }
    )
    assert cfg.backend == "neo4j"
    assert cfg.match_min == 0.9
    assert cfg.hazard_decay_floor == 0.2
    assert cfg.enabled is True
    assert cfg.context_enabled is False
    assert cfg.context_top_k == 4


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
    assert "`vector.dimensions`: 90" in cypher
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
