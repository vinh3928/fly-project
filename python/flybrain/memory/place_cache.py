"""Two-layer place-graph cache in front of Neo4j.

L1 (10s working memory)
  - Kept in RAM so every sim step can resolve without Bolt I/O.
  - Mirrored to disk so the last ~10s survives brief process hiccups.
  - On L1 match: stop here — update L1 only, no L2/Neo4j round-trip.
    Sensor decay uses the L1 place node for the rest of the step.

L2 (24h short-term memory)
  - Disk-backed place index that resets every 24 hours.
  - Used only after L1 miss. On L2 hit or Neo4j fill: save L1 + L2 + Neo4j.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from flybrain.memory.place_graph import (
    PlaceGraphStore,
    PlaceNode,
    PlaceEdge,
    _clamp01,
    _cosine_similarity,
    _normalize_walls,
    default_walls,
    encode_place_view,
)
from flybrain.phase4 import PATCH_VEC_DIM


def _now() -> float:
    return time.time()


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PlaceCacheConfig:
    """Disk + TTL settings for the two-layer Neo4j place cache."""

    enabled: bool = True
    dir: str = "logs/place_cache"
    l1_ttl_seconds: float = 10.0
    l2_ttl_seconds: float = 86400.0  # 24h; full reset on this cadence
    match_min: float = 0.97
    l1_persist_every: float = 1.0  # seconds between L1 disk mirrors

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "PlaceCacheConfig":
        data = dict(raw or {})
        # write_behind_seconds kept as ignored alias for older configs.
        return cls(
            enabled=bool(data.get("enabled", True)),
            dir=str(data.get("dir", "logs/place_cache")),
            l1_ttl_seconds=float(data.get("l1_ttl_seconds", 10.0)),
            l2_ttl_seconds=float(data.get("l2_ttl_seconds", 86400.0)),
            match_min=float(data.get("match_min", 0.97)),
            l1_persist_every=float(data.get("l1_persist_every", 1.0)),
        )


@dataclass
class _CacheEntry:
    place: PlaceNode
    touched_at: float
    dirty: bool = False

    def to_disk(self) -> dict[str, Any]:
        return {
            "place": self.place.to_dict(),
            "touched_at": float(self.touched_at),
            "dirty": bool(self.dirty),
        }

    @classmethod
    def from_disk(cls, raw: dict[str, Any]) -> "_CacheEntry":
        return cls(
            place=PlaceNode.from_dict(raw["place"]),
            touched_at=float(raw.get("touched_at") or _now()),
            dirty=bool(raw.get("dirty", False)),
        )


@dataclass
class PlaceCacheStats:
    l1_hits: int = 0
    l2_hits: int = 0
    neo4j_reads: int = 0
    neo4j_writes: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "l1_hits": self.l1_hits,
            "l2_hits": self.l2_hits,
            "neo4j_reads": self.neo4j_reads,
            "neo4j_writes": self.neo4j_writes,
        }


@dataclass
class CachedPlaceGraph:
    """PlaceGraphStore facade: L1 stop-on-hit → L2 → Neo4j (persist all three)."""

    inner: PlaceGraphStore
    config: PlaceCacheConfig = field(default_factory=PlaceCacheConfig)
    view_dim: int = PATCH_VEC_DIM
    _l1: dict[str, _CacheEntry] = field(default_factory=dict, init=False, repr=False)
    _l2: dict[str, _CacheEntry] = field(default_factory=dict, init=False, repr=False)
    _adj: set[tuple[str, str]] = field(default_factory=set, init=False, repr=False)
    _l2_started_at: float = field(default_factory=_now, init=False, repr=False)
    _last_l1_persist: float = field(default=0.0, init=False, repr=False)
    stats: PlaceCacheStats = field(default_factory=PlaceCacheStats, init=False)

    def __post_init__(self) -> None:
        self._root = Path(self.config.dir).expanduser()
        if not self._root.is_absolute():
            self._root = Path.cwd() / self._root
        self._root.mkdir(parents=True, exist_ok=True)
        self._load_l2()
        self._load_l1()

    def _l1_path(self) -> Path:
        return self._root / "l1_working.json"

    def _l2_path(self) -> Path:
        return self._root / "l2_stm.json"

    def _l2_meta_path(self) -> Path:
        return self._root / "l2_meta.json"

    def _atomic_write(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, default=str), encoding="utf-8")
        tmp.replace(path)

    def _load_json(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _load_l1(self) -> None:
        raw = self._load_json(self._l1_path())
        if not raw:
            return
        now = _now()
        ttl = float(self.config.l1_ttl_seconds)
        for key, row in (raw.get("entries") or {}).items():
            try:
                ent = _CacheEntry.from_disk(row)
            except Exception:
                continue
            if now - ent.touched_at <= ttl:
                self._l1[str(key)] = ent

    def _persist_l1(self, *, force: bool = False) -> None:
        now = _now()
        if (
            not force
            and self._last_l1_persist
            and (now - self._last_l1_persist) < float(self.config.l1_persist_every)
        ):
            return
        self._prune_l1(now)
        payload = {
            "saved_at": _utc_iso(),
            "ttl_seconds": float(self.config.l1_ttl_seconds),
            "entries": {k: v.to_disk() for k, v in self._l1.items()},
        }
        self._atomic_write(self._l1_path(), payload)
        self._last_l1_persist = now

    def _load_l2(self) -> None:
        meta = self._load_json(self._l2_meta_path()) or {}
        started = float(meta.get("started_at") or 0.0)
        now = _now()
        ttl = float(self.config.l2_ttl_seconds)
        if started <= 0.0 or (now - started) >= ttl:
            self._reset_l2(now)
            return
        self._l2_started_at = started
        raw = self._load_json(self._l2_path())
        if not raw:
            return
        for key, row in (raw.get("entries") or {}).items():
            try:
                ent = _CacheEntry.from_disk(row)
            except Exception:
                continue
            if now - ent.touched_at <= ttl:
                self._l2[str(key)] = ent

    def _reset_l2(self, now: float | None = None) -> None:
        now = float(now if now is not None else _now())
        self._l2.clear()
        self._l2_started_at = now
        self._atomic_write(
            self._l2_meta_path(),
            {
                "started_at": now,
                "started_at_iso": _utc_iso(),
                "ttl_seconds": float(self.config.l2_ttl_seconds),
            },
        )
        self._atomic_write(self._l2_path(), {"saved_at": _utc_iso(), "entries": {}})

    def _persist_l2(self) -> None:
        now = _now()
        if (now - self._l2_started_at) >= float(self.config.l2_ttl_seconds):
            self._reset_l2(now)
            return
        self._prune_l2(now)
        self._atomic_write(
            self._l2_path(),
            {
                "saved_at": _utc_iso(),
                "started_at": float(self._l2_started_at),
                "ttl_seconds": float(self.config.l2_ttl_seconds),
                "entries": {k: v.to_disk() for k, v in self._l2.items()},
            },
        )

    def _prune_l1(self, now: float | None = None) -> None:
        """Drop expired L1 entries; flush dirty ones to L2 + Neo4j once."""
        now = float(now if now is not None else _now())
        ttl = float(self.config.l1_ttl_seconds)
        expired = [k for k, e in self._l1.items() if now - e.touched_at > ttl]
        for k in expired:
            ent = self._l1.pop(k)
            if ent.dirty:
                self._persist_all_layers(ent.place, flush_neo4j=True)

    def _prune_l2(self, now: float | None = None) -> None:
        now = float(now if now is not None else _now())
        ttl = float(self.config.l2_ttl_seconds)
        if (now - self._l2_started_at) >= ttl:
            self._reset_l2(now)
            return
        expired = [k for k, e in self._l2.items() if now - e.touched_at > ttl]
        for k in expired:
            self._l2.pop(k, None)

    def _pad_vec(self, view_vec: np.ndarray) -> np.ndarray:
        return encode_place_view(view_vec, self.view_dim)

    def _best_in(
        self,
        table: dict[str, _CacheEntry],
        view_vec: np.ndarray,
        *,
        fly_id: str,
        top_k: int,
        max_age: float,
        now: float,
    ) -> list[tuple[PlaceNode, float]]:
        scored: list[tuple[PlaceNode, float]] = []
        for ent in table.values():
            if ent.place.fly_id != fly_id:
                continue
            if now - ent.touched_at > max_age:
                continue
            sim = _cosine_similarity(view_vec, ent.place.view_vec)
            scored.append((ent.place, sim))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[: max(1, int(top_k))]

    def _put_l1_only(
        self,
        place: PlaceNode,
        *,
        dirty: bool = True,
        now: float | None = None,
    ) -> PlaceNode:
        """L1 hit path: RAM (+ optional L1 disk). No L2 / Neo4j I/O."""
        now = float(now if now is not None else _now())
        key = str(place.id)
        prev = self._l1.get(key)
        self._l1[key] = _CacheEntry(
            place=place,
            touched_at=now,
            dirty=dirty or bool(prev.dirty if prev else False),
        )
        self._persist_l1()
        return place

    def _persist_all_layers(
        self,
        place: PlaceNode,
        *,
        flush_neo4j: bool,
        now: float | None = None,
    ) -> PlaceNode:
        """L2 / cold path: write L1 + L2, and Neo4j when requested."""
        now = float(now if now is not None else _now())
        saved = place
        if flush_neo4j:
            saved = self._flush_neo4j(place)
            self.stats.neo4j_writes += 1
        key = str(saved.id)
        self._l1[key] = _CacheEntry(place=saved, touched_at=now, dirty=False)
        self._l2[key] = _CacheEntry(place=saved, touched_at=now, dirty=False)
        self._persist_l1(force=True)
        self._persist_l2()
        return saved

    def _flush_neo4j(self, place: PlaceNode) -> PlaceNode:
        save = getattr(self.inner, "_save_place", None)
        if not callable(save):
            raise RuntimeError(
                "CachedPlaceGraph requires an inner store with _save_place (Neo4j)"
            )
        created = place.created_at
        created_s = created.isoformat() if hasattr(created, "isoformat") else str(created)
        return save(
            place_id=str(place.id),
            fly_id=place.fly_id,
            view_vec=place.view_vec,
            walls=place.walls,
            props=place.props,
            created_at=created_s,
        )

    def flush(self) -> None:
        """Push all dirty L1 entries to L2 + Neo4j and mirror disks."""
        for ent in list(self._l1.values()):
            if ent.dirty:
                self._persist_all_layers(ent.place, flush_neo4j=True)
        self._persist_l1(force=True)
        self._persist_l2()

    def close(self) -> None:
        self.flush()
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()

    def _merge_place(
        self,
        node: PlaceNode,
        vec: np.ndarray,
        *,
        fly_id: str,
        walls: dict[str, Any] | None,
        prop_map: dict[str, Any],
    ) -> PlaceNode:
        merged_vec = encode_place_view(
            0.85 * np.asarray(node.view_vec, dtype=np.float32) + 0.15 * vec,
            self.view_dim,
        )
        return PlaceNode(
            id=node.id,
            fly_id=fly_id,
            view_vec=merged_vec,
            walls=_normalize_walls({**node.walls, **(walls or {})}),
            props={**dict(node.props), **prop_map},
            created_at=node.created_at,
            updated_at=datetime.now(timezone.utc),
        )

    # --- PlaceGraphStore API ---------------------------------------------------

    def nearest(
        self,
        view_vec: np.ndarray,
        *,
        fly_id: str = "fly_001",
        top_k: int = 5,
    ) -> list[tuple[PlaceNode, float]]:
        vec = self._pad_vec(view_vec)
        now = _now()
        self._prune_l1(now)
        if (now - self._l2_started_at) >= float(self.config.l2_ttl_seconds):
            self._reset_l2(now)

        match_min = float(self.config.match_min)
        l1_hits = self._best_in(
            self._l1,
            vec,
            fly_id=fly_id,
            top_k=top_k,
            max_age=float(self.config.l1_ttl_seconds),
            now=now,
        )
        if l1_hits and l1_hits[0][1] >= match_min:
            self.stats.l1_hits += 1
            key = str(l1_hits[0][0].id)
            if key in self._l1:
                self._l1[key].touched_at = now
            return l1_hits

        l2_hits = self._best_in(
            self._l2,
            vec,
            fly_id=fly_id,
            top_k=top_k,
            max_age=float(self.config.l2_ttl_seconds),
            now=now,
        )
        if l2_hits and l2_hits[0][1] >= match_min:
            self.stats.l2_hits += 1
            # L1 miss → L2 hit: promote and durable-save (L1 + L2 + Neo4j).
            node = l2_hits[0][0]
            saved = self._persist_all_layers(node, flush_neo4j=True, now=now)
            return [(saved, l2_hits[0][1])] + list(l2_hits[1:])

        self.stats.neo4j_reads += 1
        hits = self.inner.nearest(vec, fly_id=fly_id, top_k=top_k)
        out: list[tuple[PlaceNode, float]] = []
        for node, sim in hits:
            if sim >= match_min:
                saved = self._persist_all_layers(node, flush_neo4j=True, now=now)
                out.append((saved, sim))
            else:
                out.append((node, sim))
        return out if out else hits

    def get(self, place_id: int | str) -> PlaceNode | None:
        key = str(place_id)
        now = _now()
        self._prune_l1(now)
        ent = self._l1.get(key)
        if ent is not None and (now - ent.touched_at) <= float(self.config.l1_ttl_seconds):
            self.stats.l1_hits += 1
            ent.touched_at = now
            return ent.place
        ent = self._l2.get(key)
        if ent is not None and (now - ent.touched_at) <= float(self.config.l2_ttl_seconds):
            self.stats.l2_hits += 1
            return self._persist_all_layers(ent.place, flush_neo4j=True, now=now)
        self.stats.neo4j_reads += 1
        node = self.inner.get(place_id)
        if node is not None:
            return self._persist_all_layers(node, flush_neo4j=True, now=now)
        return node

    def list_places(self, *, fly_id: str = "fly_001", limit: int = 64) -> list[PlaceNode]:
        self.stats.neo4j_reads += 1
        rows = self.inner.list_places(fly_id=fly_id, limit=limit)
        now = _now()
        out: list[PlaceNode] = []
        for node in rows:
            out.append(self._persist_all_layers(node, flush_neo4j=False, now=now))
        return out

    def upsert_place(
        self,
        view_vec: np.ndarray,
        *,
        fly_id: str = "fly_001",
        match_min: float = 0.97,
        walls: dict[str, Any] | None = None,
        props: dict[str, Any] | None = None,
        step: int | None = None,
    ) -> PlaceNode:
        vec = self._pad_vec(view_vec)
        prop_map = dict(props or {})
        if step is not None:
            prop_map["last_step"] = int(step)
        now = _now()
        self._prune_l1(now)
        if (now - self._l2_started_at) >= float(self.config.l2_ttl_seconds):
            self._reset_l2(now)

        # 1) L1 match → stop. Update L1 only; sensor decay uses this node. No Neo4j.
        l1_hits = self._best_in(
            self._l1,
            vec,
            fly_id=fly_id,
            top_k=1,
            max_age=float(self.config.l1_ttl_seconds),
            now=now,
        )
        if l1_hits and l1_hits[0][1] >= float(match_min):
            self.stats.l1_hits += 1
            updated = self._merge_place(
                l1_hits[0][0], vec, fly_id=fly_id, walls=walls, prop_map=prop_map
            )
            return self._put_l1_only(updated, dirty=True, now=now)

        # 2) L2 match → save L1 + L2 + Neo4j.
        l2_hits = self._best_in(
            self._l2,
            vec,
            fly_id=fly_id,
            top_k=1,
            max_age=float(self.config.l2_ttl_seconds),
            now=now,
        )
        if l2_hits and l2_hits[0][1] >= float(match_min):
            self.stats.l2_hits += 1
            updated = self._merge_place(
                l2_hits[0][0], vec, fly_id=fly_id, walls=walls, prop_map=prop_map
            )
            return self._persist_all_layers(updated, flush_neo4j=True, now=now)

        # 3) Cold: Neo4j nearest / create, then save all layers.
        self.stats.neo4j_reads += 1
        matches = self.inner.nearest(vec, fly_id=fly_id, top_k=1)
        if matches and matches[0][1] >= float(match_min):
            updated = self._merge_place(
                matches[0][0], vec, fly_id=fly_id, walls=walls, prop_map=prop_map
            )
            return self._persist_all_layers(updated, flush_neo4j=True, now=now)

        place_id = str(uuid4())
        created = datetime.now(timezone.utc)
        fresh = PlaceNode(
            id=place_id,
            fly_id=fly_id,
            view_vec=vec,
            walls=_normalize_walls(walls),
            props=prop_map,
            created_at=created,
            updated_at=created,
        )
        return self._persist_all_layers(fresh, flush_neo4j=True, now=now)

    def create_place(
        self,
        view_vec: np.ndarray,
        *,
        fly_id: str = "fly_001",
        walls: dict[str, Any] | None = None,
        props: dict[str, Any] | None = None,
        step: int | None = None,
    ) -> PlaceNode:
        """Always create — never merge lookalikes (context-aware path)."""
        vec = self._pad_vec(view_vec)
        prop_map = dict(props or {})
        if step is not None:
            prop_map["last_step"] = int(step)
        now = _now()
        fresh = PlaceNode(
            id=str(uuid4()),
            fly_id=fly_id,
            view_vec=vec,
            walls=_normalize_walls(walls),
            props=prop_map,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        return self._persist_all_layers(fresh, flush_neo4j=True, now=now)

    def refresh_place(
        self,
        place_id: int | str,
        view_vec: np.ndarray,
        *,
        walls: dict[str, Any] | None = None,
        props: dict[str, Any] | None = None,
        step: int | None = None,
    ) -> PlaceNode | None:
        key = str(place_id)
        now = _now()
        self._prune_l1(now)
        vec = self._pad_vec(view_vec)
        prop_map = dict(props or {})
        if step is not None:
            prop_map["last_step"] = int(step)

        ent = self._l1.get(key)
        l1_hot = ent is not None and (now - ent.touched_at) <= float(self.config.l1_ttl_seconds)
        node = ent.place if l1_hot else self.get(place_id)
        if node is None:
            return None
        updated = self._merge_place(
            node, vec, fly_id=node.fly_id, walls=walls, prop_map=prop_map
        )
        if l1_hot:
            self.stats.l1_hits += 1
            return self._put_l1_only(updated, dirty=True, now=now)
        return self._persist_all_layers(updated, flush_neo4j=True, now=now)

    def update_wall(
        self,
        place_id: int | str,
        direction: str,
        *,
        familiarity: float | None = None,
        last_seen_step: int | None = None,
        notes: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> PlaceNode | None:
        # Prefer L1 without durable round-trips when the place is still hot.
        key = str(place_id)
        now = _now()
        self._prune_l1(now)
        ent = self._l1.get(key)
        node: PlaceNode | None
        l1_hot = False
        if ent is not None and (now - ent.touched_at) <= float(self.config.l1_ttl_seconds):
            self.stats.l1_hits += 1
            node = ent.place
            l1_hot = True
        else:
            node = self.get(place_id)
        if node is None:
            return None
        direction = str(direction or "").strip().lower()
        from flybrain.memory.place_graph import WALL_DIRS

        if direction not in WALL_DIRS:
            raise ValueError(f"direction must be one of {WALL_DIRS}, got {direction!r}")
        wall = dict(node.walls.get(direction) or default_walls()[direction])
        if familiarity is not None:
            wall["familiarity"] = _clamp01(float(familiarity))
        if last_seen_step is not None:
            wall["last_seen_step"] = int(last_seen_step)
        if notes is not None:
            wall["notes"] = str(notes)[:200]
        if extra:
            wall.update(extra)
        node.walls[direction] = wall
        node.updated_at = datetime.now(timezone.utc)
        if l1_hot:
            return self._put_l1_only(node, dirty=True, now=now)
        return self._persist_all_layers(node, flush_neo4j=True, now=now)

    def upsert_adjacent(
        self,
        src_id: int | str,
        dst_id: int | str,
        *,
        props: dict[str, Any] | None = None,
        step: int | None = None,
    ) -> PlaceEdge:
        edge = self.inner.upsert_adjacent(src_id, dst_id, props=props, step=step)
        self._adj.add((str(src_id), str(dst_id)))
        return edge

    def has_adjacent(self, src_id: int | str, dst_id: int | str) -> bool:
        key = (str(src_id), str(dst_id))
        if key in self._adj:
            return True
        if self.inner.has_adjacent(src_id, dst_id):
            self._adj.add(key)
            return True
        return False

    def edges(self) -> list[PlaceEdge]:
        rows = self.inner.edges()
        for e in rows:
            self._adj.add((str(e.src_id), str(e.dst_id)))
        return rows
