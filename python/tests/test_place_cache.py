"""Two-layer place-graph cache (L1=10s stop-on-hit, L2=24h + Neo4j)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from flybrain.memory.place_cache import CachedPlaceGraph, PlaceCacheConfig
from flybrain.memory.place_graph import PlaceEdge, PlaceNode, _cosine_similarity, encode_place_view
from flybrain.phase4 import PATCH_VEC_DIM


def _view(seed: int, dim: int = PATCH_VEC_DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


@dataclass
class FakeNeo4jPlaceGraph:
    """Minimal Neo4j stand-in with ``_save_place`` for cache tests."""

    view_dim: int = PATCH_VEC_DIM
    nearest_calls: int = 0
    get_calls: int = 0
    save_calls: int = 0
    _nodes: dict[str, PlaceNode] = field(default_factory=dict)
    _edges: list[PlaceEdge] = field(default_factory=list)

    def _pad_vec(self, view_vec: np.ndarray) -> np.ndarray:
        return encode_place_view(view_vec, self.view_dim)

    def nearest(
        self,
        view_vec: np.ndarray,
        *,
        fly_id: str = "fly_001",
        top_k: int = 5,
    ) -> list[tuple[PlaceNode, float]]:
        self.nearest_calls += 1
        vec = self._pad_vec(view_vec)
        scored: list[tuple[PlaceNode, float]] = []
        for node in self._nodes.values():
            if node.fly_id != fly_id:
                continue
            scored.append((node, _cosine_similarity(vec, node.view_vec)))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[: max(1, int(top_k))]

    def get(self, place_id: int | str) -> PlaceNode | None:
        self.get_calls += 1
        return self._nodes.get(str(place_id))

    def list_places(self, *, fly_id: str = "fly_001", limit: int = 64) -> list[PlaceNode]:
        rows = [n for n in self._nodes.values() if n.fly_id == fly_id]
        return rows[: max(1, int(limit))]

    def _save_place(
        self,
        *,
        place_id: str,
        fly_id: str,
        view_vec: np.ndarray,
        walls: dict[str, dict[str, Any]],
        props: dict[str, Any],
        created_at: str | None = None,
    ) -> PlaceNode:
        self.save_calls += 1
        now = datetime.now(UTC)
        created = now
        if created_at:
            try:
                created = datetime.fromisoformat(created_at)
            except ValueError:
                created = now
        node = PlaceNode(
            id=place_id,
            fly_id=fly_id,
            view_vec=self._pad_vec(view_vec),
            walls=dict(walls),
            props=dict(props),
            created_at=created,
            updated_at=now,
        )
        self._nodes[place_id] = node
        return node

    def upsert_place(self, *args: Any, **kwargs: Any) -> PlaceNode:
        raise AssertionError("CachedPlaceGraph should call _save_place, not upsert_place")

    def create_place(
        self,
        view_vec: np.ndarray,
        *,
        fly_id: str = "fly_001",
        walls: dict[str, Any] | None = None,
        props: dict[str, Any] | None = None,
        step: int | None = None,
    ) -> PlaceNode:
        from uuid import uuid4

        return self._save_place(
            place_id=str(uuid4()),
            fly_id=fly_id,
            view_vec=view_vec,
            walls=walls or {},
            props=dict(props or {}),
        )

    def refresh_place(
        self,
        place_id: int | str,
        view_vec: np.ndarray,
        *,
        walls: dict[str, Any] | None = None,
        props: dict[str, Any] | None = None,
        step: int | None = None,
    ) -> PlaceNode | None:
        node = self.get(place_id)
        if node is None:
            return None
        return self._save_place(
            place_id=str(place_id),
            fly_id=node.fly_id,
            view_vec=view_vec,
            walls=walls if walls is not None else node.walls,
            props={**dict(node.props), **dict(props or {})},
            created_at=node.created_at.isoformat(),
        )

    def upsert_adjacent(
        self,
        src_id: int | str,
        dst_id: int | str,
        *,
        props: dict[str, Any] | None = None,
        step: int | None = None,
    ) -> PlaceEdge:
        for e in self._edges:
            if str(e.src_id) == str(src_id) and str(e.dst_id) == str(dst_id):
                e.props.update(dict(props or {}))
                return e
        edge = PlaceEdge(
            id=f"{src_id}->{dst_id}",
            src_id=src_id,
            dst_id=dst_id,
            props=dict(props or {}),
        )
        self._edges.append(edge)
        return edge

    def has_adjacent(self, src_id: int | str, dst_id: int | str) -> bool:
        return any(
            str(e.src_id) == str(src_id) and str(e.dst_id) == str(dst_id) for e in self._edges
        )

    def update_wall(self, *args: Any, **kwargs: Any) -> PlaceNode | None:
        return None

    def edges(self) -> list[PlaceEdge]:
        return list(self._edges)

    def neighbors(self, place_id: int | str, *, undirected: bool = True) -> list[PlaceEdge]:
        from flybrain.memory.place_graph import neighbors_from_edges

        return neighbors_from_edges(self.edges(), place_id, undirected=undirected)

    def find_places(
        self, *, fly_id: str = "fly_001", label: str | None = None, limit: int = 64
    ) -> list[PlaceNode]:
        rows = self.list_places(fly_id=fly_id, limit=max(1, int(limit)) * 4)
        if label is not None:
            rows = [n for n in rows if str((n.props or {}).get("label") or "") == str(label)]
        return rows[: max(1, int(limit))]

    def shortest_path(
        self,
        src_id: int | str,
        dst_id: int | str,
        *,
        undirected: bool = True,
        max_hops: int = 64,
    ) -> list[int | str] | None:
        from flybrain.memory.place_graph import shortest_path_bfs

        return shortest_path_bfs(
            self.edges(), src_id, dst_id, undirected=undirected, max_hops=max_hops
        )

    def close(self) -> None:
        return None


def test_l1_match_stops_no_neo4j_round_trip(tmp_path: Path) -> None:
    inner = FakeNeo4jPlaceGraph()
    cache = CachedPlaceGraph(
        inner=inner,
        config=PlaceCacheConfig(
            enabled=True,
            dir=str(tmp_path / "cache"),
            l1_ttl_seconds=10.0,
            l2_ttl_seconds=86400.0,
            match_min=0.9,
            l1_persist_every=0.0,
        ),
    )
    view = _view(1)
    first = cache.upsert_place(view, fly_id="fly_a", step=0)
    assert inner.save_calls == 1
    assert inner.nearest_calls == 1
    saves_after_create = inner.save_calls
    nearest_after_create = inner.nearest_calls

    # L1 hit: bump walls (decay path) — must not touch Neo4j or rewrite L2 neo flush.
    second = cache.upsert_place(
        view, fly_id="fly_a", step=1, walls={"north": {"familiarity": 0.25}}
    )
    assert str(second.id) == str(first.id)
    assert second.walls["north"]["familiarity"] == 0.25
    assert inner.nearest_calls == nearest_after_create
    assert inner.save_calls == saves_after_create  # zero Neo4j round-trips
    assert cache.stats.l1_hits >= 1
    # L2 still has the pre-bump snapshot until L1 expires / L2 path runs.
    assert str(first.id) in cache._l1
    assert cache._l1[str(first.id)].dirty is True
    cache.close()  # eviction flush on close
    assert inner.save_calls > saves_after_create


def test_l2_hit_saves_l1_l2_and_neo4j(tmp_path: Path) -> None:
    inner = FakeNeo4jPlaceGraph()
    cache = CachedPlaceGraph(
        inner=inner,
        config=PlaceCacheConfig(
            enabled=True,
            dir=str(tmp_path / "cache"),
            l1_ttl_seconds=0.01,
            l2_ttl_seconds=86400.0,
            match_min=0.9,
            l1_persist_every=0.0,
        ),
    )
    view = _view(2)
    node = cache.upsert_place(view, fly_id="fly_a", step=0)
    saves_after_create = inner.save_calls
    nearest_after_create = inner.nearest_calls

    # Expire L1 only; L2 still holds the place.
    for ent in list(cache._l1.values()):
        ent.touched_at -= 1.0
        ent.dirty = False  # avoid prune flush before nearest
    cache._l1.clear()

    third = cache.upsert_place(view, fly_id="fly_a", step=2, walls={"east": {"familiarity": 0.4}})
    assert str(third.id) == str(node.id)
    assert third.walls["east"]["familiarity"] == 0.4
    assert cache.stats.l2_hits >= 1
    assert inner.nearest_calls == nearest_after_create  # L2 hit, no Neo4j read
    assert inner.save_calls == saves_after_create + 1  # durable write
    assert str(node.id) in cache._l1
    assert str(node.id) in cache._l2
    cache.close()


def test_prune_flushes_several_expired_dirty_entries_once(tmp_path: Path) -> None:
    inner = FakeNeo4jPlaceGraph()
    cache = CachedPlaceGraph(
        inner=inner,
        config=PlaceCacheConfig(
            enabled=True,
            dir=str(tmp_path / "cache"),
            l1_ttl_seconds=10.0,
            l2_ttl_seconds=86400.0,
            match_min=0.99,
            l1_persist_every=0.0,
        ),
    )
    ids = [str(cache.upsert_place(_view(10 + i), fly_id="fly_a", step=i).id) for i in range(3)]
    for ent in cache._l1.values():
        ent.touched_at -= 60.0
        ent.dirty = True
    saves_before = inner.save_calls

    cache.nearest(_view(99), fly_id="fly_a", top_k=1)

    assert inner.save_calls == saves_before + len(ids)
    assert all(not cache._l1[i].dirty for i in ids)
    cache.close()


def _hot_cache(tmp_path: Path, inner: FakeNeo4jPlaceGraph, flush_every: float) -> CachedPlaceGraph:
    return CachedPlaceGraph(
        inner=inner,
        config=PlaceCacheConfig(
            enabled=True,
            dir=str(tmp_path / "cache"),
            l1_ttl_seconds=10.0,
            match_min=0.9,
            l1_persist_every=0.0,
            neo4j_flush_every=flush_every,
        ),
    )


def test_hot_place_is_saved_to_neo4j_on_a_cadence(tmp_path: Path) -> None:
    inner = FakeNeo4jPlaceGraph()
    cache = _hot_cache(tmp_path, inner, flush_every=5.0)
    node = cache.upsert_place(_view(1), fly_id="fly_a", step=0)
    saves = inner.save_calls

    cache.upsert_place(_view(1), fly_id="fly_a", step=1, walls={"north": {"familiarity": 0.3}})
    assert inner.save_calls == saves  # within the cadence: L1 only
    assert cache._l1[str(node.id)].dirty

    cache._l1[str(node.id)].flushed_at -= 6.0
    cache.upsert_place(_view(1), fly_id="fly_a", step=2, walls={"north": {"familiarity": 0.4}})
    assert inner.save_calls == saves + 1
    assert inner._nodes[str(node.id)].walls["north"]["familiarity"] == 0.4
    assert not cache._l1[str(node.id)].dirty
    cache.close()


def test_cache_revisit_keeps_the_place_view(tmp_path: Path) -> None:
    inner = FakeNeo4jPlaceGraph()
    cache = _hot_cache(tmp_path, inner, flush_every=0.0)
    node = cache.upsert_place(_view(1), fly_id="fly_a", step=0)
    anchor = np.asarray(node.view_vec).copy()
    nudged = _view(1) + 0.05 * _view(2)
    cache.upsert_place(nudged, fly_id="fly_a", step=1)
    cache.refresh_place(node.id, nudged, step=2)
    assert np.allclose(cache._l1[str(node.id)].place.view_vec, anchor)
    assert np.allclose(inner._nodes[str(node.id)].view_vec, anchor)
    cache.close()


def test_l2_resets_after_24h_window(tmp_path: Path) -> None:
    inner = FakeNeo4jPlaceGraph()
    cache = CachedPlaceGraph(
        inner=inner,
        config=PlaceCacheConfig(
            enabled=True,
            dir=str(tmp_path / "cache"),
            l1_ttl_seconds=10.0,
            l2_ttl_seconds=86400.0,
            match_min=0.9,
            l1_persist_every=0.0,
        ),
    )
    view = _view(3)
    cache.upsert_place(view, fly_id="fly_a", step=0)
    assert len(cache._l2) == 1
    cache._l2_started_at -= 86400.0 + 1.0
    cache.nearest(view, fly_id="fly_a", top_k=1)
    assert (tmp_path / "cache" / "l2_meta.json").exists()
    cache.close()


def test_l1_disk_reloads_within_ttl(tmp_path: Path) -> None:
    inner = FakeNeo4jPlaceGraph()
    cfg = PlaceCacheConfig(
        enabled=True,
        dir=str(tmp_path / "cache"),
        l1_ttl_seconds=10.0,
        l2_ttl_seconds=86400.0,
        match_min=0.9,
        l1_persist_every=0.0,
    )
    cache = CachedPlaceGraph(inner=inner, config=cfg)
    view = _view(4)
    node = cache.upsert_place(view, fly_id="fly_a", step=0)
    # Mark clean so close() doesn't require inner2.
    for ent in cache._l1.values():
        ent.dirty = False
    cache._persist_l1(force=True)
    cache._persist_l2()

    inner2 = FakeNeo4jPlaceGraph()
    cache2 = CachedPlaceGraph(inner=inner2, config=cfg)
    hits = cache2.nearest(view, fly_id="fly_a", top_k=1)
    assert hits and str(hits[0][0].id) == str(node.id)
    assert inner2.nearest_calls == 0
    assert inner2.save_calls == 0  # L1 stop — no Neo4j
    assert cache2.stats.l1_hits >= 1
    cache2.close()


def test_list_places_does_not_inflate_l2(tmp_path: Path) -> None:
    """Browse/list must not rewrite L2 JSON for every listed Neo4j row."""
    inner = FakeNeo4jPlaceGraph()
    for i in range(5):
        inner.create_place(_view(i + 1), fly_id="fly_a", props={"label": "place"})
    cache = CachedPlaceGraph(
        inner=inner,
        config=PlaceCacheConfig(
            enabled=True,
            dir=str(tmp_path / "cache"),
            l1_ttl_seconds=10.0,
            l2_ttl_seconds=86400.0,
            match_min=0.9,
            l1_persist_every=999.0,
        ),
    )
    before = len(cache._l2)
    rows = cache.list_places(fly_id="fly_a", limit=10)
    assert len(rows) == 5
    assert len(cache._l2) == before  # no L2 write on list
