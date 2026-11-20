"""Spatial place graph: location nodes = view vectors of the chamber.

A place node is a mental map entry for \"being here looking at the box\".
The node's identity is its ``view_vec`` (stereo patch / place embedding).
Node details include the four chamber walls (north/east/south/west).
Edges between places are reserved for later navigation links.
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import numpy as np

from flybrain.phase4 import PATCH_VEC_DIM

WALL_DIRS: tuple[str, ...] = ("north", "east", "south", "west")


def neighbors_from_edges(
    edges: list[PlaceEdge],
    place_id: int | str,
    *,
    undirected: bool = True,
) -> list[PlaceEdge]:
    """Edges touching ``place_id`` (both directions when ``undirected``)."""
    key = str(place_id)
    out: list[PlaceEdge] = []
    for edge in edges:
        if str(edge.src_id) == key:
            out.append(edge)
        elif undirected and str(edge.dst_id) == key:
            out.append(edge)
    return out


def shortest_path_bfs(
    edges: list[PlaceEdge],
    src_id: int | str,
    dst_id: int | str,
    *,
    undirected: bool = True,
    max_hops: int = 64,
) -> list[int | str] | None:
    """BFS place-id path over ADJACENT edges (inclusive endpoints), or ``None``."""
    src, dst = str(src_id), str(dst_id)
    if src == dst:
        return [src_id]
    adj: dict[str, list[tuple[str, PlaceEdge]]] = {}
    for edge in edges:
        a, b = str(edge.src_id), str(edge.dst_id)
        adj.setdefault(a, []).append((b, edge))
        if undirected:
            adj.setdefault(b, []).append((a, edge))
    limit = max(1, int(max_hops))
    prev: dict[str, str | None] = {src: None}
    queue: deque[tuple[str, int]] = deque([(src, 0)])
    while queue:
        cur, depth = queue.popleft()
        if depth >= limit:
            continue
        for nxt, _edge in adj.get(cur, []):
            if nxt in prev:
                continue
            prev[nxt] = cur
            if nxt == dst:
                path = [dst]
                node: str | None = dst
                while node is not None and node != src:
                    node = prev[node]
                    if node is not None:
                        path.append(node)
                path.reverse()
                id_map = {str(e.src_id): e.src_id for e in edges}
                id_map.update({str(e.dst_id): e.dst_id for e in edges})
                id_map[src] = src_id
                id_map[dst] = dst_id
                return [id_map.get(p, p) for p in path]
            queue.append((nxt, depth + 1))
    return None


def edge_toward(
    edges: list[PlaceEdge],
    src_id: int | str,
    dst_id: int | str,
) -> PlaceEdge | None:
    """Directed edge ``src→dst`` if present, else the reverse (for undirected follow)."""
    src, dst = str(src_id), str(dst_id)
    forward = reverse = None
    for edge in edges:
        if str(edge.src_id) == src and str(edge.dst_id) == dst:
            forward = edge
        elif str(edge.src_id) == dst and str(edge.dst_id) == src:
            reverse = edge
    return forward if forward is not None else reverse


@dataclass
class PlaceGraphConfig:
    """Live mental-map settings (Neo4j place nodes + wall familiarity decay)."""

    enabled: bool = True
    backend: str = "in_memory"
    uri: str = "bolt://127.0.0.1:7687"
    user: str = "neo4j"
    password: str | None = None
    database: str = "neo4j"
    view_dim: int = PATCH_VEC_DIM
    # Mean-centered patch views need a higher bar than raw cosine (raw nonneg
    # patch_vec pairs sit near 0.99 even when visually distinct).
    match_min: float = 0.97
    update_every: int = 1
    familiarity_bump: float = 0.12
    hazard_decay_strength: float = 0.85
    hazard_decay_floor: float = 0.15
    # Path/reference disambiguation: identical views from different previous
    # places become distinct Place nodes linked by ADJACENT edges.
    context_enabled: bool = True
    context_top_k: int = 8
    # Leave same_place when landmarks / novelty say the scene changed even if
    # patch cosine still matches (open meadow otherwise glues one Place forever).
    split_on_novelty: bool = True
    novelty_split_min: float = 0.25
    split_on_landmark: bool = True
    opening_delta_min: float = 0.15
    wall_delta_min: float = 0.15
    cache: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> PlaceGraphConfig:
        data = dict(raw or {})
        return cls(
            enabled=bool(data.get("enabled", True)),
            backend=str(data.get("backend", "in_memory")),
            uri=str(data.get("uri", "bolt://127.0.0.1:7687")),
            user=str(data.get("user", "neo4j")),
            password=data.get("password"),
            database=str(data.get("database", "neo4j")),
            view_dim=int(data.get("view_dim", PATCH_VEC_DIM)),
            match_min=float(data.get("match_min", 0.97)),
            update_every=max(1, int(data.get("update_every", 1))),
            familiarity_bump=float(data.get("familiarity_bump", 0.12)),
            hazard_decay_strength=float(data.get("hazard_decay_strength", 0.85)),
            hazard_decay_floor=float(data.get("hazard_decay_floor", 0.15)),
            context_enabled=bool(data.get("context_enabled", True)),
            context_top_k=max(1, int(data.get("context_top_k", 8))),
            split_on_novelty=bool(data.get("split_on_novelty", True)),
            novelty_split_min=float(data.get("novelty_split_min", 0.25)),
            split_on_landmark=bool(data.get("split_on_landmark", True)),
            opening_delta_min=float(data.get("opening_delta_min", 0.15)),
            wall_delta_min=float(data.get("wall_delta_min", 0.15)),
            cache=dict(data["cache"]) if isinstance(data.get("cache"), dict) else None,
        )


def should_leave_same_place(
    *,
    visual_novelty: float = 0.0,
    new_stimulus: bool = False,
    opening: float = 0.0,
    wall: float = 0.0,
    prev_opening: float | None = None,
    prev_wall: float | None = None,
    split_on_novelty: bool = True,
    novelty_split_min: float = 0.25,
    split_on_landmark: bool = True,
    opening_delta_min: float = 0.15,
    wall_delta_min: float = 0.15,
) -> tuple[bool, str]:
    """Whether landmark / novelty cues should break a ``same_place`` match.

    Returns ``(leave, reason)`` with reason in
    ``novelty`` | ``new_stimulus`` | ``opening_delta`` | ``wall_delta`` | ``\"\"``.
    """
    if bool(split_on_novelty):
        if bool(new_stimulus):
            return True, "new_stimulus"
        if float(visual_novelty) >= float(novelty_split_min):
            return True, "novelty"
    if bool(split_on_landmark):
        if prev_opening is not None and abs(float(opening) - float(prev_opening)) >= float(
            opening_delta_min
        ):
            return True, "opening_delta"
        if prev_wall is not None and abs(float(wall) - float(prev_wall)) >= float(wall_delta_min):
            return True, "wall_delta"
    return False, ""


def default_walls() -> dict[str, dict[str, Any]]:
    """Chamber place always has four walls; familiarity starts at zero."""
    return {
        d: {
            "present": True,
            "label": d,
            "familiarity": 0.0,
            "last_seen_step": None,
            "notes": "",
        }
        for d in WALL_DIRS
    }


def _clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _pad_view(view_vec: np.ndarray, dim: int) -> np.ndarray:
    """Truncate/pad a raw view vector to ``dim`` without changing its scale."""
    v = np.asarray(view_vec, dtype=np.float32).reshape(-1)
    if v.size == int(dim):
        return v.copy()
    out = np.zeros(int(dim), dtype=np.float32)
    n = min(int(dim), int(v.size))
    if n > 0:
        out[:n] = v[:n]
    return out


def encode_place_view(view_vec: np.ndarray, dim: int = PATCH_VEC_DIM) -> np.ndarray:
    """Place identity: mean-center + L2-normalize.

    Stereo ``patch_vec`` entries are non-negative patch statistics. Raw cosine
    between two unrelated nonneg vectors is typically ~0.7–0.99, so a chamber
    collapses into one Place. Mean-centering (Pearson / centered cosine) puts
    unrelated views near 0 while keeping near-duplicates near 1. Storing the
    unit-length centered form keeps Neo4j's cosine vector index consistent
    with in-process matching.
    """
    v = _pad_view(view_vec, dim)
    v = v - float(v.mean())
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return v
    return (v / n).astype(np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    if aa.size == 0 or bb.size == 0 or aa.size != bb.size:
        return 0.0
    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(aa, bb) / (na * nb))


def place_similarity(
    a: np.ndarray,
    b: np.ndarray,
    *,
    dim: int | None = None,
) -> float:
    """Centered-cosine place match score in ``[-1, 1]`` (typically near 1 = same)."""
    d = int(dim) if dim is not None else max(int(np.asarray(a).size), int(np.asarray(b).size))
    return _cosine_similarity(encode_place_view(a, d), encode_place_view(b, d))


def _normalize_walls(raw: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    base = default_walls()
    if not isinstance(raw, dict):
        return base
    for d in WALL_DIRS:
        if d not in raw or not isinstance(raw[d], dict):
            continue
        merged = dict(base[d])
        for k, v in raw[d].items():
            merged[k] = v
        if "familiarity" in merged:
            merged["familiarity"] = _clamp01(float(merged["familiarity"] or 0.0))
        merged["present"] = bool(merged.get("present", True))
        merged["label"] = str(merged.get("label") or d)
        base[d] = merged
    return base


@dataclass
class PlaceNode:
    """One location on the mental map: a picture-of-the-box vector + wall details."""

    id: int | str
    fly_id: str
    view_vec: np.ndarray
    walls: dict[str, dict[str, Any]] = field(default_factory=default_walls)
    props: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "fly_id": self.fly_id,
            "view_vec": np.asarray(self.view_vec, dtype=np.float32).reshape(-1).tolist(),
            "walls": {d: dict(self.walls.get(d, {})) for d in WALL_DIRS},
            "props": dict(self.props),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PlaceNode:
        vec = np.asarray(raw.get("view_vec") or [], dtype=np.float32).reshape(-1)
        created = raw.get("created_at")
        updated = raw.get("updated_at")
        if isinstance(created, str):
            created = datetime.fromisoformat(created)
        if isinstance(updated, str):
            updated = datetime.fromisoformat(updated)
        return cls(
            id=raw["id"],
            fly_id=str(raw.get("fly_id") or "fly_001"),
            view_vec=encode_place_view(vec, max(PATCH_VEC_DIM, int(vec.size) or PATCH_VEC_DIM)),
            walls=_normalize_walls(raw.get("walls")),
            props=dict(raw.get("props") or {}),
            created_at=created or datetime.now(UTC),
            updated_at=updated or datetime.now(UTC),
        )


@dataclass
class PlaceEdge:
    """Navigation / reference link between places (``ADJACENT``)."""

    id: int | str
    src_id: int | str
    dst_id: int | str
    kind: str = "adjacent"
    props: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "src_id": self.src_id,
            "dst_id": self.dst_id,
            "kind": self.kind,
            "props": dict(self.props),
        }


# Hungry food search: sugar contact + sugar antenna odor only (not generic odor plumes).
SUGAR_SENSE_KINDS: tuple[str, ...] = (
    "sugar",
    "sugar_smell",
)
# Alias kept for callers; hunger routing uses sugar senses exclusively.
FOOD_SENSE_KINDS: tuple[str, ...] = SUGAR_SENSE_KINDS


@dataclass
class FoodPlaceHit:
    """A Place that carries a food/scent Sense (Neo4j ``HAS_SENSE`` or props.senses)."""

    place: PlaceNode
    kind: str
    value: float
    # Earliest known encounter step when available (else None).
    first_step: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "place_id": self.place.id,
            "kind": self.kind,
            "value": float(self.value),
            "first_step": self.first_step,
            "label": str((self.place.props or {}).get("label") or ""),
            "created_at": self.place.created_at.isoformat(),
        }


@dataclass
class PathHop:
    """One ADJACENT hop on a return path (node pair + edge props for steering)."""

    src_id: int | str
    dst_id: int | str
    props: dict[str, Any] = field(default_factory=dict)
    reversed: bool = False  # True when following the undirected reverse of a recorded edge

    def to_dict(self) -> dict[str, Any]:
        return {
            "src_id": self.src_id,
            "dst_id": self.dst_id,
            "props": dict(self.props),
            "reversed": bool(self.reversed),
        }


def food_sense_value(
    props: dict[str, Any] | None,
    *,
    kinds: tuple[str, ...] | list[str] | None = None,
) -> tuple[str, float, int | None]:
    """Best sugar-food channel on place props.senses → (kind, value, first_step)."""
    senses = (props or {}).get("senses") if isinstance(props, dict) else None
    if not isinstance(senses, dict):
        return "", 0.0, None
    best_kind = ""
    best_val = 0.0
    best_step: int | None = None
    for kind in kinds or SUGAR_SENSE_KINDS:
        row = senses.get(kind)
        if not isinstance(row, dict):
            continue
        val = float(row.get("value") or 0.0)
        if val > best_val:
            best_val = val
            best_kind = str(kind)
            step = row.get("last_seen_step")
            best_step = int(step) if step is not None else None
    return best_kind, best_val, best_step


def find_food_places_in_nodes(
    nodes: list[PlaceNode],
    *,
    min_value: float = 0.05,
    limit: int = 32,
    kinds: tuple[str, ...] | list[str] | None = None,
) -> list[FoodPlaceHit]:
    """Scan in-memory place props for sugar food senses; earliest experience first."""
    hit_kinds = tuple(kinds) if kinds is not None else SUGAR_SENSE_KINDS
    hits: list[FoodPlaceHit] = []
    for node in nodes:
        kind, value, step = food_sense_value(node.props, kinds=hit_kinds)
        if value < float(min_value) or not kind:
            continue
        hits.append(FoodPlaceHit(place=node, kind=kind, value=value, first_step=step))
    # First experience: lowest step, then oldest created_at, then strongest smell.
    hits.sort(
        key=lambda h: (
            h.first_step if h.first_step is not None else 10**12,
            h.place.created_at,
            -h.value,
        )
    )
    return hits[: max(1, int(limit))]


def path_hops_from_ids(
    edges: list[PlaceEdge],
    place_ids: list[int | str],
) -> list[PathHop]:
    """Resolve ADJACENT edge props along a place-id path (undirected)."""
    hops: list[PathHop] = []
    for i in range(len(place_ids) - 1):
        src, dst = place_ids[i], place_ids[i + 1]
        edge = edge_toward(edges, src, dst)
        if edge is None:
            hops.append(PathHop(src_id=src, dst_id=dst, props={}, reversed=False))
            continue
        reversed_hop = str(edge.src_id) == str(dst) and str(edge.dst_id) == str(src)
        hops.append(
            PathHop(
                src_id=src,
                dst_id=dst,
                props=dict(edge.props or {}),
                reversed=reversed_hop,
            )
        )
    return hops


class PlaceGraphStore(Protocol):
    """Graph store for place nodes and ADJACENT reference edges."""

    def upsert_place(
        self,
        view_vec: np.ndarray,
        *,
        fly_id: str = "fly_001",
        match_min: float = 0.97,
        walls: dict[str, Any] | None = None,
        props: dict[str, Any] | None = None,
        step: int | None = None,
    ) -> PlaceNode: ...

    def create_place(
        self,
        view_vec: np.ndarray,
        *,
        fly_id: str = "fly_001",
        walls: dict[str, Any] | None = None,
        props: dict[str, Any] | None = None,
        step: int | None = None,
    ) -> PlaceNode: ...

    def refresh_place(
        self,
        place_id: int | str,
        view_vec: np.ndarray,
        *,
        walls: dict[str, Any] | None = None,
        props: dict[str, Any] | None = None,
        step: int | None = None,
    ) -> PlaceNode | None: ...

    def get(self, place_id: int | str) -> PlaceNode | None: ...

    def nearest(
        self,
        view_vec: np.ndarray,
        *,
        fly_id: str = "fly_001",
        top_k: int = 5,
    ) -> list[tuple[PlaceNode, float]]: ...

    def list_places(self, *, fly_id: str = "fly_001", limit: int = 64) -> list[PlaceNode]: ...

    def update_wall(
        self,
        place_id: int | str,
        direction: str,
        *,
        familiarity: float | None = None,
        last_seen_step: int | None = None,
        notes: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> PlaceNode | None: ...

    def upsert_adjacent(
        self,
        src_id: int | str,
        dst_id: int | str,
        *,
        props: dict[str, Any] | None = None,
        step: int | None = None,
    ) -> PlaceEdge: ...

    def has_adjacent(self, src_id: int | str, dst_id: int | str) -> bool: ...

    def edges(self) -> list[PlaceEdge]: ...

    def neighbors(
        self,
        place_id: int | str,
        *,
        undirected: bool = True,
    ) -> list[PlaceEdge]: ...

    def find_places(
        self,
        *,
        fly_id: str = "fly_001",
        label: str | None = None,
        limit: int = 64,
    ) -> list[PlaceNode]: ...

    def find_food_places(
        self,
        *,
        fly_id: str = "fly_001",
        min_value: float = 0.05,
        kinds: tuple[str, ...] | list[str] | None = None,
        limit: int = 32,
    ) -> list[FoodPlaceHit]: ...

    def shortest_path(
        self,
        src_id: int | str,
        dst_id: int | str,
        *,
        undirected: bool = True,
        max_hops: int = 64,
    ) -> list[int | str] | None: ...

    def shortest_path_with_edges(
        self,
        src_id: int | str,
        dst_id: int | str,
        *,
        undirected: bool = True,
        max_hops: int = 64,
    ) -> tuple[list[int | str], list[PathHop]] | None: ...


@dataclass
class InMemoryPlaceGraph:
    """In-process place graph (tests / offline). Edges table held empty for later."""

    view_dim: int = PATCH_VEC_DIM
    _nodes: dict[int, PlaceNode] = field(default_factory=dict)
    _edges: dict[int, PlaceEdge] = field(default_factory=dict)
    _next_id: int = 1
    _next_edge_id: int = 1

    def reset(self) -> None:
        self._nodes.clear()
        self._edges.clear()
        self._next_id = 1
        self._next_edge_id = 1

    def _pad_vec(self, view_vec: np.ndarray) -> np.ndarray:
        return encode_place_view(view_vec, self.view_dim)

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
        best: PlaceNode | None = None
        best_sim = -1.0
        for node in self._nodes.values():
            if node.fly_id != fly_id:
                continue
            sim = _cosine_similarity(vec, node.view_vec)
            if sim > best_sim:
                best_sim = sim
                best = node
        now = datetime.now(UTC)
        if best is not None and best_sim >= float(match_min):
            if walls is not None:
                best.walls = _normalize_walls({**best.walls, **walls})
            if props:
                best.props.update(props)
            if step is not None:
                best.props["last_step"] = int(step)
            best.updated_at = now
            return best

        node = PlaceNode(
            id=self._next_id,
            fly_id=fly_id,
            view_vec=vec,
            walls=_normalize_walls(walls),
            props=dict(props or {}),
            created_at=now,
            updated_at=now,
        )
        if step is not None:
            node.props["last_step"] = int(step)
        self._next_id += 1
        self._nodes[node.id] = node
        return node

    def create_place(
        self,
        view_vec: np.ndarray,
        *,
        fly_id: str = "fly_001",
        walls: dict[str, Any] | None = None,
        props: dict[str, Any] | None = None,
        step: int | None = None,
    ) -> PlaceNode:
        """Always allocate a new Place (ignores view lookalikes)."""
        now = datetime.now(UTC)
        node = PlaceNode(
            id=self._next_id,
            fly_id=fly_id,
            view_vec=self._pad_vec(view_vec),
            walls=_normalize_walls(walls),
            props=dict(props or {}),
            created_at=now,
            updated_at=now,
        )
        if step is not None:
            node.props["last_step"] = int(step)
        self._next_id += 1
        self._nodes[node.id] = node
        return node

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
        if walls is not None:
            node.walls = _normalize_walls({**node.walls, **walls})
        if props:
            node.props.update(props)
        if step is not None:
            node.props["last_step"] = int(step)
        node.updated_at = datetime.now(UTC)
        return node

    def get(self, place_id: int | str) -> PlaceNode | None:
        try:
            return self._nodes.get(int(place_id))
        except (TypeError, ValueError):
            return self._nodes.get(place_id)  # type: ignore[arg-type]

    def nearest(
        self,
        view_vec: np.ndarray,
        *,
        fly_id: str = "fly_001",
        top_k: int = 5,
    ) -> list[tuple[PlaceNode, float]]:
        vec = self._pad_vec(view_vec)
        scored: list[tuple[PlaceNode, float]] = []
        for node in self._nodes.values():
            if node.fly_id != fly_id:
                continue
            scored.append((node, _cosine_similarity(vec, node.view_vec)))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[: max(1, int(top_k))]

    def list_places(self, *, fly_id: str = "fly_001", limit: int = 64) -> list[PlaceNode]:
        rows = [n for n in self._nodes.values() if n.fly_id == fly_id]
        rows.sort(key=lambda n: n.updated_at, reverse=True)
        return rows[: max(1, int(limit))]

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
        node = self.get(place_id)
        if node is None:
            return None
        d = str(direction or "").strip().lower()
        if d not in WALL_DIRS:
            raise ValueError(f"direction must be one of {WALL_DIRS}, got {direction!r}")
        wall = dict(node.walls.get(d) or default_walls()[d])
        if familiarity is not None:
            wall["familiarity"] = _clamp01(float(familiarity))
        if last_seen_step is not None:
            wall["last_seen_step"] = int(last_seen_step)
        if notes is not None:
            wall["notes"] = str(notes)[:200]
        if extra:
            wall.update(extra)
        node.walls[d] = wall
        node.updated_at = datetime.now(UTC)
        return node

    def upsert_adjacent(
        self,
        src_id: int | str,
        dst_id: int | str,
        *,
        props: dict[str, Any] | None = None,
        step: int | None = None,
    ) -> PlaceEdge:
        src = str(src_id)
        dst = str(dst_id)
        for edge in self._edges.values():
            if str(edge.src_id) == src and str(edge.dst_id) == dst:
                edge.props.update(dict(props or {}))
                if step is not None:
                    edge.props["last_step"] = int(step)
                edge.props["times"] = int(edge.props.get("times") or 0) + 1
                return edge
        edge = PlaceEdge(
            id=self._next_edge_id,
            src_id=src_id,
            dst_id=dst_id,
            kind="adjacent",
            props={
                **dict(props or {}),
                "times": 1,
                **({"last_step": int(step)} if step is not None else {}),
            },
        )
        self._edges[self._next_edge_id] = edge
        self._next_edge_id += 1
        return edge

    def has_adjacent(self, src_id: int | str, dst_id: int | str) -> bool:
        src = str(src_id)
        dst = str(dst_id)
        return any(str(e.src_id) == src and str(e.dst_id) == dst for e in self._edges.values())

    def edges(self) -> list[PlaceEdge]:
        return list(self._edges.values())

    def neighbors(
        self,
        place_id: int | str,
        *,
        undirected: bool = True,
    ) -> list[PlaceEdge]:
        return neighbors_from_edges(self.edges(), place_id, undirected=undirected)

    def find_places(
        self,
        *,
        fly_id: str = "fly_001",
        label: str | None = None,
        limit: int = 64,
    ) -> list[PlaceNode]:
        rows = self.list_places(fly_id=fly_id, limit=max(1, int(limit)) * 4)
        if label is not None:
            want = str(label)
            rows = [n for n in rows if str((n.props or {}).get("label") or "") == want]
        return rows[: max(1, int(limit))]

    def find_food_places(
        self,
        *,
        fly_id: str = "fly_001",
        min_value: float = 0.05,
        kinds: tuple[str, ...] | list[str] | None = None,
        limit: int = 32,
    ) -> list[FoodPlaceHit]:
        rows = self.list_places(fly_id=fly_id, limit=max(64, int(limit) * 8))
        return find_food_places_in_nodes(
            rows, min_value=min_value, limit=limit, kinds=kinds
        )

    def shortest_path(
        self,
        src_id: int | str,
        dst_id: int | str,
        *,
        undirected: bool = True,
        max_hops: int = 64,
    ) -> list[int | str] | None:
        return shortest_path_bfs(
            self.edges(), src_id, dst_id, undirected=undirected, max_hops=max_hops
        )

    def shortest_path_with_edges(
        self,
        src_id: int | str,
        dst_id: int | str,
        *,
        undirected: bool = True,
        max_hops: int = 64,
    ) -> tuple[list[int | str], list[PathHop]] | None:
        path = self.shortest_path(
            src_id, dst_id, undirected=undirected, max_hops=max_hops
        )
        if path is None:
            return None
        return path, path_hops_from_ids(self.edges(), path)


@dataclass
class PgPlaceGraph:
    """Postgres + pgvector place graph (fly_mind.place_nodes / place_edges)."""

    dsn: str
    view_dim: int = PATCH_VEC_DIM
    schema: str = "fly_mind"
    _conn: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install mind extras: pip install -e '.[mind]'") from exc
        self._conn = psycopg.connect(self.dsn)
        register_vector(self._conn)
        self._ensure_schema()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _ensure_schema(self) -> None:
        assert self._conn is not None
        walls_default = json.dumps(default_walls())
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.place_nodes (
                    id BIGSERIAL PRIMARY KEY,
                    fly_id TEXT NOT NULL DEFAULT 'fly_001',
                    view_vec vector({int(self.view_dim)}) NOT NULL,
                    walls JSONB NOT NULL DEFAULT %s::jsonb,
                    props JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """,
                (walls_default,),
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.place_edges (
                    id BIGSERIAL PRIMARY KEY,
                    src_id BIGINT NOT NULL REFERENCES {self.schema}.place_nodes(id) ON DELETE CASCADE,
                    dst_id BIGINT NOT NULL REFERENCES {self.schema}.place_nodes(id) ON DELETE CASCADE,
                    kind TEXT NOT NULL DEFAULT 'adjacent',
                    props JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
        self._conn.commit()

    def _pad_vec(self, view_vec: np.ndarray) -> np.ndarray:
        return encode_place_view(view_vec, self.view_dim)

    def _row_to_node(self, row: Any) -> PlaceNode:
        # id, fly_id, view_vec, walls, props, created_at, updated_at
        walls = row[3] if isinstance(row[3], dict) else json.loads(row[3])
        props = row[4] if isinstance(row[4], dict) else json.loads(row[4])
        vec = np.asarray(row[2], dtype=np.float32).reshape(-1)
        return PlaceNode(
            id=int(row[0]),
            fly_id=str(row[1]),
            view_vec=encode_place_view(vec, self.view_dim),
            walls=_normalize_walls(walls),
            props=dict(props or {}),
            created_at=row[5],
            updated_at=row[6],
        )

    def nearest(
        self,
        view_vec: np.ndarray,
        *,
        fly_id: str = "fly_001",
        top_k: int = 5,
    ) -> list[tuple[PlaceNode, float]]:
        assert self._conn is not None
        vec = self._pad_vec(view_vec)
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, fly_id, view_vec, walls, props, created_at, updated_at,
                       1 - (view_vec <=> %s::vector) AS sim
                FROM {self.schema}.place_nodes
                WHERE fly_id = %s
                ORDER BY view_vec <=> %s::vector
                LIMIT %s
                """,
                (vec.tolist(), fly_id, vec.tolist(), int(top_k)),
            )
            rows = cur.fetchall()
        return [(self._row_to_node(r[:7]), float(r[7])) for r in rows]

    def get(self, place_id: int) -> PlaceNode | None:
        assert self._conn is not None
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, fly_id, view_vec, walls, props, created_at, updated_at
                FROM {self.schema}.place_nodes WHERE id = %s
                """,
                (int(place_id),),
            )
            row = cur.fetchone()
        return self._row_to_node(row) if row else None

    def list_places(self, *, fly_id: str = "fly_001", limit: int = 64) -> list[PlaceNode]:
        assert self._conn is not None
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, fly_id, view_vec, walls, props, created_at, updated_at
                FROM {self.schema}.place_nodes
                WHERE fly_id = %s
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (fly_id, int(limit)),
            )
            rows = cur.fetchall()
        return [self._row_to_node(r) for r in rows]

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
        assert self._conn is not None
        vec = self._pad_vec(view_vec)
        neighbors = self.nearest(vec, fly_id=fly_id, top_k=1)
        prop_map = dict(props or {})
        if step is not None:
            prop_map["last_step"] = int(step)
        if neighbors and neighbors[0][1] >= float(match_min):
            node, _sim = neighbors[0]
            merged_vec = node.view_vec
            merged_walls = _normalize_walls({**node.walls, **(walls or {})})
            merged_props = {**node.props, **prop_map}
            with self._conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE {self.schema}.place_nodes
                    SET view_vec = %s::vector,
                        walls = %s::jsonb,
                        props = %s::jsonb,
                        updated_at = NOW()
                    WHERE id = %s
                    RETURNING id, fly_id, view_vec, walls, props, created_at, updated_at
                    """,
                    (
                        merged_vec.tolist(),
                        json.dumps(merged_walls),
                        json.dumps(merged_props),
                        int(node.id),
                    ),
                )
                row = cur.fetchone()
            self._conn.commit()
            assert row is not None
            return self._row_to_node(row)

        wall_json = json.dumps(_normalize_walls(walls))
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self.schema}.place_nodes (fly_id, view_vec, walls, props)
                VALUES (%s, %s::vector, %s::jsonb, %s::jsonb)
                RETURNING id, fly_id, view_vec, walls, props, created_at, updated_at
                """,
                (fly_id, vec.tolist(), wall_json, json.dumps(prop_map)),
            )
            row = cur.fetchone()
        self._conn.commit()
        assert row is not None
        return self._row_to_node(row)

    def create_place(
        self,
        view_vec: np.ndarray,
        *,
        fly_id: str = "fly_001",
        walls: dict[str, Any] | None = None,
        props: dict[str, Any] | None = None,
        step: int | None = None,
    ) -> PlaceNode:
        assert self._conn is not None
        vec = self._pad_vec(view_vec)
        prop_map = dict(props or {})
        if step is not None:
            prop_map["last_step"] = int(step)
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self.schema}.place_nodes (fly_id, view_vec, walls, props)
                VALUES (%s, %s::vector, %s::jsonb, %s::jsonb)
                RETURNING id, fly_id, view_vec, walls, props, created_at, updated_at
                """,
                (fly_id, vec.tolist(), json.dumps(_normalize_walls(walls)), json.dumps(prop_map)),
            )
            row = cur.fetchone()
        self._conn.commit()
        assert row is not None
        return self._row_to_node(row)

    def refresh_place(
        self,
        place_id: int | str,
        view_vec: np.ndarray,
        *,
        walls: dict[str, Any] | None = None,
        props: dict[str, Any] | None = None,
        step: int | None = None,
    ) -> PlaceNode | None:
        node = self.get(int(place_id))
        if node is None:
            return None
        merged_vec = node.view_vec
        merged_walls = _normalize_walls({**node.walls, **(walls or {})}) if walls else node.walls
        prop_map = {**dict(node.props), **dict(props or {})}
        if step is not None:
            prop_map["last_step"] = int(step)
        assert self._conn is not None
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {self.schema}.place_nodes
                SET view_vec = %s::vector,
                    walls = %s::jsonb,
                    props = %s::jsonb,
                    updated_at = NOW()
                WHERE id = %s
                RETURNING id, fly_id, view_vec, walls, props, created_at, updated_at
                """,
                (
                    merged_vec.tolist(),
                    json.dumps(_normalize_walls(merged_walls)),
                    json.dumps(prop_map),
                    int(place_id),
                ),
            )
            row = cur.fetchone()
        self._conn.commit()
        return self._row_to_node(row) if row else None

    def update_wall(
        self,
        place_id: int,
        direction: str,
        *,
        familiarity: float | None = None,
        last_seen_step: int | None = None,
        notes: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> PlaceNode | None:
        node = self.get(place_id)
        if node is None:
            return None
        d = str(direction or "").strip().lower()
        if d not in WALL_DIRS:
            raise ValueError(f"direction must be one of {WALL_DIRS}, got {direction!r}")
        wall = dict(node.walls.get(d) or default_walls()[d])
        if familiarity is not None:
            wall["familiarity"] = _clamp01(float(familiarity))
        if last_seen_step is not None:
            wall["last_seen_step"] = int(last_seen_step)
        if notes is not None:
            wall["notes"] = str(notes)[:200]
        if extra:
            wall.update(extra)
        node.walls[d] = wall
        assert self._conn is not None
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {self.schema}.place_nodes
                SET walls = %s::jsonb, updated_at = NOW()
                WHERE id = %s
                RETURNING id, fly_id, view_vec, walls, props, created_at, updated_at
                """,
                (json.dumps(_normalize_walls(node.walls)), int(place_id)),
            )
            row = cur.fetchone()
        self._conn.commit()
        return self._row_to_node(row) if row else None

    def upsert_adjacent(
        self,
        src_id: int | str,
        dst_id: int | str,
        *,
        props: dict[str, Any] | None = None,
        step: int | None = None,
    ) -> PlaceEdge:
        assert self._conn is not None
        prop_map = dict(props or {})
        if step is not None:
            prop_map["last_step"] = int(step)
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, src_id, dst_id, kind, props FROM {self.schema}.place_edges
                WHERE src_id = %s AND dst_id = %s AND kind = 'adjacent'
                LIMIT 1
                """,
                (int(src_id), int(dst_id)),
            )
            row = cur.fetchone()
            if row:
                old = row[4] if isinstance(row[4], dict) else json.loads(row[4] or "{}")
                merged = {**dict(old or {}), **prop_map}
                merged["times"] = int(merged.get("times") or 0) + 1
                cur.execute(
                    f"""
                    UPDATE {self.schema}.place_edges SET props = %s::jsonb WHERE id = %s
                    RETURNING id, src_id, dst_id, kind, props
                    """,
                    (json.dumps(merged), int(row[0])),
                )
                row = cur.fetchone()
            else:
                prop_map.setdefault("times", 1)
                cur.execute(
                    f"""
                    INSERT INTO {self.schema}.place_edges (src_id, dst_id, kind, props)
                    VALUES (%s, %s, 'adjacent', %s::jsonb)
                    RETURNING id, src_id, dst_id, kind, props
                    """,
                    (int(src_id), int(dst_id), json.dumps(prop_map)),
                )
                row = cur.fetchone()
        self._conn.commit()
        assert row is not None
        props_out = row[4] if isinstance(row[4], dict) else json.loads(row[4] or "{}")
        return PlaceEdge(
            id=int(row[0]),
            src_id=int(row[1]),
            dst_id=int(row[2]),
            kind=str(row[3]),
            props=dict(props_out or {}),
        )

    def has_adjacent(self, src_id: int | str, dst_id: int | str) -> bool:
        assert self._conn is not None
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT 1 FROM {self.schema}.place_edges
                WHERE src_id = %s AND dst_id = %s AND kind = 'adjacent'
                LIMIT 1
                """,
                (int(src_id), int(dst_id)),
            )
            return cur.fetchone() is not None

    def edges(self) -> list[PlaceEdge]:
        assert self._conn is not None
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT id, src_id, dst_id, kind, props FROM {self.schema}.place_edges"
            )
            rows = cur.fetchall()
        out: list[PlaceEdge] = []
        for r in rows:
            props = r[4] if isinstance(r[4], dict) else json.loads(r[4] or "{}")
            out.append(
                PlaceEdge(
                    id=int(r[0]),
                    src_id=int(r[1]),
                    dst_id=int(r[2]),
                    kind=str(r[3]),
                    props=dict(props or {}),
                )
            )
        return out

    def neighbors(
        self,
        place_id: int | str,
        *,
        undirected: bool = True,
    ) -> list[PlaceEdge]:
        return neighbors_from_edges(self.edges(), place_id, undirected=undirected)

    def find_places(
        self,
        *,
        fly_id: str = "fly_001",
        label: str | None = None,
        limit: int = 64,
    ) -> list[PlaceNode]:
        rows = self.list_places(fly_id=fly_id, limit=max(1, int(limit)) * 4)
        if label is not None:
            want = str(label)
            rows = [n for n in rows if str((n.props or {}).get("label") or "") == want]
        return rows[: max(1, int(limit))]

    def find_food_places(
        self,
        *,
        fly_id: str = "fly_001",
        min_value: float = 0.05,
        kinds: tuple[str, ...] | list[str] | None = None,
        limit: int = 32,
    ) -> list[FoodPlaceHit]:
        rows = self.list_places(fly_id=fly_id, limit=max(64, int(limit) * 8))
        return find_food_places_in_nodes(
            rows, min_value=min_value, limit=limit, kinds=kinds
        )

    def shortest_path(
        self,
        src_id: int | str,
        dst_id: int | str,
        *,
        undirected: bool = True,
        max_hops: int = 64,
    ) -> list[int | str] | None:
        return shortest_path_bfs(
            self.edges(), src_id, dst_id, undirected=undirected, max_hops=max_hops
        )

    def shortest_path_with_edges(
        self,
        src_id: int | str,
        dst_id: int | str,
        *,
        undirected: bool = True,
        max_hops: int = 64,
    ) -> tuple[list[int | str], list[PathHop]] | None:
        path = self.shortest_path(
            src_id, dst_id, undirected=undirected, max_hops=max_hops
        )
        if path is None:
            return None
        return path, path_hops_from_ids(self.edges(), path)


@dataclass
class Neo4jPlaceGraph:
    """Neo4j-backed mental map with indexed vector reads.

    Data model:

    ``(:Place)-[:HAS_WALL]->(:Wall)`` for chamber walls.
    ``(:Place)-[:HAS_SENSE]->(:Sense)`` for vision-correlated smell/taste details.
    ``(:Place)-[:ADJACENT]->(:Place)`` for path/reference transitions.
    """

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "flymindgraph"
    database: str = "neo4j"
    view_dim: int = PATCH_VEC_DIM
    _driver: Any = field(default=None, init=False, repr=False)

    VECTOR_INDEX = "place_view_vec"
    PLACE_ID_CONSTRAINT = "place_id_unique"
    WALL_ID_CONSTRAINT = "wall_id_unique"

    def __post_init__(self) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:  # pragma: no cover - optional mind dependency
            raise ImportError("Install mind extras: pip install -e '.[mind]'") from exc
        self._driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
        )
        self._ensure_schema()

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def _query(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[Any]:
        if self._driver is None:
            raise RuntimeError("Neo4j place graph is closed")
        records, _summary, _keys = self._driver.execute_query(
            cypher,
            parameters_=parameters or {},
            database_=self.database,
        )
        return list(records)

    def _ensure_schema(self) -> None:
        self._query(
            f"""
            CREATE CONSTRAINT {self.PLACE_ID_CONSTRAINT} IF NOT EXISTS
            FOR (p:Place) REQUIRE p.place_id IS UNIQUE
            """
        )
        self._query(
            f"""
            CREATE CONSTRAINT {self.WALL_ID_CONSTRAINT} IF NOT EXISTS
            FOR (w:Wall) REQUIRE w.wall_id IS UNIQUE
            """
        )
        self._query(
            """
            CREATE INDEX place_fly_updated IF NOT EXISTS
            FOR (p:Place) ON (p.fly_id, p.updated_at)
            """
        )
        self._query(
            f"""
            CREATE VECTOR INDEX {self.VECTOR_INDEX} IF NOT EXISTS
            FOR (p:Place) ON (p.view_vec)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {int(self.view_dim)},
                `vector.similarity_function`: 'cosine'
            }}}}
            """
        )
        self._query(
            "CALL db.awaitIndex($index_name, $timeout_seconds)",
            {"index_name": self.VECTOR_INDEX, "timeout_seconds": 30},
        )

    def _pad_vec(self, view_vec: np.ndarray) -> np.ndarray:
        return encode_place_view(view_vec, self.view_dim)

    @staticmethod
    def _parse_time(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and value:
            return datetime.fromisoformat(value)
        return datetime.now(UTC)

    def _record_to_node(self, record: Any) -> PlaceNode:
        walls_raw = record.get("walls") or []
        walls = default_walls()
        for raw in walls_raw:
            if not isinstance(raw, dict):
                raw = dict(raw)
            direction = str(raw.get("direction") or "").lower()
            if direction not in WALL_DIRS:
                continue
            extra_raw = raw.get("extra_json") or "{}"
            try:
                extra = json.loads(extra_raw) if isinstance(extra_raw, str) else dict(extra_raw)
            except (TypeError, ValueError, json.JSONDecodeError):
                extra = {}
            walls[direction].update(
                {
                    "present": bool(raw.get("present", True)),
                    "label": str(raw.get("label") or direction),
                    "familiarity": _clamp01(float(raw.get("familiarity", 0.0) or 0.0)),
                    "last_seen_step": raw.get("last_seen_step"),
                    "notes": str(raw.get("notes") or ""),
                    **extra,
                }
            )
        props_raw = record.get("props_json") or "{}"
        try:
            props = json.loads(props_raw) if isinstance(props_raw, str) else dict(props_raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            props = {}
        return PlaceNode(
            id=str(record["place_id"]),
            fly_id=str(record["fly_id"]),
            view_vec=encode_place_view(
                np.asarray(record["view_vec"], dtype=np.float32),
                self.view_dim,
            ),
            walls=walls,
            props=props,
            created_at=self._parse_time(record.get("created_at")),
            updated_at=self._parse_time(record.get("updated_at")),
        )

    @staticmethod
    def _return_clause(*, include_score: bool = False) -> str:
        score = ", score" if include_score else ""
        return f"""
            RETURN p.place_id AS place_id,
                   p.fly_id AS fly_id,
                   p.view_vec AS view_vec,
                   p.props_json AS props_json,
                   p.created_at AS created_at,
                   p.updated_at AS updated_at,
                   [(p)-[:HAS_WALL]->(w:Wall) | {{
                       direction: w.direction,
                       present: w.present,
                       label: w.label,
                       familiarity: w.familiarity,
                       last_seen_step: w.last_seen_step,
                       notes: w.notes,
                       extra_json: w.extra_json
                   }}] AS walls
                   {score}
        """

    def nearest(
        self,
        view_vec: np.ndarray,
        *,
        fly_id: str = "fly_001",
        top_k: int = 5,
    ) -> list[tuple[PlaceNode, float]]:
        vec = self._pad_vec(view_vec)
        k = max(1, int(top_k))
        rows = self._query(
            f"""
            CALL db.index.vector.queryNodes(
                '{self.VECTOR_INDEX}',
                $candidate_k,
                $view_vec
            )
            YIELD node AS p, score
            WHERE p.fly_id = $fly_id
            {self._return_clause(include_score=True)}
            ORDER BY score DESC
            LIMIT $top_k
            """,
            {
                "view_vec": vec.tolist(),
                "fly_id": fly_id,
                "candidate_k": max(16, k * 8),
                "top_k": k,
            },
        )
        if not rows:
            # The vector index cannot pre-filter by fly_id. Fall back to a
            # fly-indexed exact scan when over-fetching found only other flies.
            rows = self._query(
                f"""
                MATCH (p:Place {{fly_id: $fly_id}})
                WITH p, vector.similarity.cosine(p.view_vec, $view_vec) AS score
                {self._return_clause(include_score=True)}
                ORDER BY score DESC
                LIMIT $top_k
                """,
                {
                    "view_vec": vec.tolist(),
                    "fly_id": fly_id,
                    "top_k": k,
                },
            )
        return [(self._record_to_node(row), float(row["score"])) for row in rows]

    def get(self, place_id: int | str) -> PlaceNode | None:
        rows = self._query(
            f"""
            MATCH (p:Place {{place_id: $place_id}})
            {self._return_clause()}
            """,
            {"place_id": str(place_id)},
        )
        return self._record_to_node(rows[0]) if rows else None

    def list_places(self, *, fly_id: str = "fly_001", limit: int = 64) -> list[PlaceNode]:
        rows = self._query(
            f"""
            MATCH (p:Place {{fly_id: $fly_id}})
            WITH p
            ORDER BY p.updated_at DESC
            LIMIT $limit
            {self._return_clause()}
            """,
            {"fly_id": fly_id, "limit": max(1, int(limit))},
        )
        return [self._record_to_node(row) for row in rows]

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
        now = datetime.now(UTC).isoformat()
        wall_rows: list[dict[str, Any]] = []
        for direction in WALL_DIRS:
            wall = dict(walls[direction])
            standard = {
                "present",
                "label",
                "familiarity",
                "last_seen_step",
                "notes",
            }
            wall_rows.append(
                {
                    "wall_id": f"{place_id}:{direction}",
                    "direction": direction,
                    "present": bool(wall.get("present", True)),
                    "label": str(wall.get("label") or direction),
                    "familiarity": _clamp01(float(wall.get("familiarity", 0.0) or 0.0)),
                    "last_seen_step": wall.get("last_seen_step"),
                    "notes": str(wall.get("notes") or "")[:200],
                    "extra_json": json.dumps(
                        {key: value for key, value in wall.items() if key not in standard}
                    ),
                }
            )
        self._query(
            """
            MERGE (p:Place {place_id: $place_id})
            ON CREATE SET p.created_at = $created_at
            SET p.fly_id = $fly_id,
                p.view_vec = $view_vec,
                p.label = $label,
                p.props_json = $props_json,
                p.updated_at = $updated_at
            WITH p
            UNWIND $walls AS wall
            MERGE (w:Wall {wall_id: wall.wall_id})
            SET w.direction = wall.direction,
                w.present = wall.present,
                w.label = wall.label,
                w.familiarity = wall.familiarity,
                w.last_seen_step = wall.last_seen_step,
                w.notes = wall.notes,
                w.extra_json = wall.extra_json
            MERGE (p)-[:HAS_WALL {direction: wall.direction}]->(w)
            """,
            {
                "place_id": place_id,
                "fly_id": fly_id,
                "view_vec": np.asarray(view_vec, dtype=np.float32).tolist(),
                "label": str((props or {}).get("label") or "place"),
                "props_json": json.dumps(props),
                "created_at": created_at or now,
                "updated_at": now,
                "walls": wall_rows,
            },
        )
        # Vision-correlated sensor details as Sense nodes.
        from flybrain.sensors.place_senses import sense_rows_for_neo4j

        sense_rows = sense_rows_for_neo4j(place_id, (props or {}).get("senses"))
        if sense_rows:
            self._query(
                """
                MATCH (p:Place {place_id: $place_id})
                UNWIND $senses AS sense
                MERGE (s:Sense {sense_id: sense.sense_id})
                SET s.kind = sense.kind,
                    s.value = sense.value,
                    s.last_seen_step = sense.last_seen_step
                MERGE (p)-[r:HAS_SENSE]->(s)
                SET r.value = sense.value,
                    r.last_seen_step = sense.last_seen_step
                """,
                {"place_id": place_id, "senses": sense_rows},
            )
        saved = self.get(place_id)
        if saved is None:  # pragma: no cover - indicates DB write/read inconsistency
            raise RuntimeError(f"Neo4j failed to return saved place {place_id}")
        return saved

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
        matches = self.nearest(vec, fly_id=fly_id, top_k=1)
        prop_map = dict(props or {})
        if step is not None:
            prop_map["last_step"] = int(step)
        if matches and matches[0][1] >= float(match_min):
            node = matches[0][0]
            merged_walls = _normalize_walls({**node.walls, **(walls or {})})
            return self._save_place(
                place_id=str(node.id),
                fly_id=fly_id,
                view_vec=node.view_vec,
                walls=merged_walls,
                props={**node.props, **prop_map},
                created_at=node.created_at.isoformat(),
            )
        return self._save_place(
            place_id=str(uuid4()),
            fly_id=fly_id,
            view_vec=vec,
            walls=_normalize_walls(walls),
            props=prop_map,
        )

    def create_place(
        self,
        view_vec: np.ndarray,
        *,
        fly_id: str = "fly_001",
        walls: dict[str, Any] | None = None,
        props: dict[str, Any] | None = None,
        step: int | None = None,
    ) -> PlaceNode:
        vec = self._pad_vec(view_vec)
        prop_map = dict(props or {})
        if step is not None:
            prop_map["last_step"] = int(step)
        return self._save_place(
            place_id=str(uuid4()),
            fly_id=fly_id,
            view_vec=vec,
            walls=_normalize_walls(walls),
            props=prop_map,
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
        merged_walls = _normalize_walls({**node.walls, **(walls or {})}) if walls else node.walls
        prop_map = {**dict(node.props), **dict(props or {})}
        if step is not None:
            prop_map["last_step"] = int(step)
        return self._save_place(
            place_id=str(node.id),
            fly_id=node.fly_id,
            view_vec=node.view_vec,
            walls=merged_walls,
            props=prop_map,
            created_at=node.created_at.isoformat(),
        )

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
        node = self.get(place_id)
        if node is None:
            return None
        direction = str(direction or "").strip().lower()
        if direction not in WALL_DIRS:
            raise ValueError(f"direction must be one of {WALL_DIRS}, got {direction!r}")
        wall = dict(node.walls[direction])
        if familiarity is not None:
            wall["familiarity"] = _clamp01(float(familiarity))
        if last_seen_step is not None:
            wall["last_seen_step"] = int(last_seen_step)
        if notes is not None:
            wall["notes"] = str(notes)[:200]
        if extra:
            wall.update(extra)
        node.walls[direction] = wall
        return self._save_place(
            place_id=str(node.id),
            fly_id=node.fly_id,
            view_vec=node.view_vec,
            walls=node.walls,
            props=node.props,
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
        prop_map = dict(props or {})
        if step is not None:
            prop_map["last_step"] = int(step)
        now = datetime.now(UTC).isoformat()
        rows = self._query(
            """
            MATCH (src:Place {place_id: $src_id})
            MATCH (dst:Place {place_id: $dst_id})
            MERGE (src)-[e:ADJACENT]->(dst)
            ON CREATE SET e.kind = 'adjacent',
                          e.created_at = $now,
                          e.times = 1,
                          e.props_json = $props_json
            ON MATCH SET e.times = coalesce(e.times, 0) + 1,
                         e.updated_at = $now,
                         e.props_json = $props_json
            RETURN elementId(e) AS edge_id,
                   src.place_id AS src_id,
                   dst.place_id AS dst_id,
                   e.kind AS kind,
                   e.props_json AS props_json,
                   e.times AS times
            """,
            {
                "src_id": str(src_id),
                "dst_id": str(dst_id),
                "now": now,
                "props_json": json.dumps(prop_map),
            },
        )
        if not rows:
            raise RuntimeError(f"Neo4j failed to link places {src_id!r} -> {dst_id!r}")
        row = rows[0]
        props_out = json.loads(row.get("props_json") or "{}")
        if isinstance(props_out, dict):
            props_out["times"] = int(row.get("times") or props_out.get("times") or 1)
        return PlaceEdge(
            id=row["edge_id"],
            src_id=row["src_id"],
            dst_id=row["dst_id"],
            kind=str(row.get("kind") or "adjacent"),
            props=dict(props_out or {}),
        )

    def has_adjacent(self, src_id: int | str, dst_id: int | str) -> bool:
        rows = self._query(
            """
            MATCH (:Place {place_id: $src_id})-[:ADJACENT]->(:Place {place_id: $dst_id})
            RETURN 1 AS ok LIMIT 1
            """,
            {"src_id": str(src_id), "dst_id": str(dst_id)},
        )
        return bool(rows)

    def edges(self) -> list[PlaceEdge]:
        rows = self._query(
            """
            MATCH (src:Place)-[edge:ADJACENT]->(dst:Place)
            RETURN elementId(edge) AS edge_id,
                   src.place_id AS src_id,
                   dst.place_id AS dst_id,
                   edge.kind AS kind,
                   edge.props_json AS props_json
            """
        )
        out: list[PlaceEdge] = []
        for row in rows:
            props_raw = row.get("props_json") or "{}"
            props = json.loads(props_raw) if isinstance(props_raw, str) else dict(props_raw)
            out.append(
                PlaceEdge(
                    id=row["edge_id"],
                    src_id=row["src_id"],
                    dst_id=row["dst_id"],
                    kind=str(row.get("kind") or "adjacent"),
                    props=props,
                )
            )
        return out

    def neighbors(
        self,
        place_id: int | str,
        *,
        undirected: bool = True,
    ) -> list[PlaceEdge]:
        if undirected:
            rows = self._query(
                """
                MATCH (p:Place {place_id: $place_id})-[edge:ADJACENT]-(other:Place)
                RETURN elementId(edge) AS edge_id,
                       startNode(edge).place_id AS src_id,
                       endNode(edge).place_id AS dst_id,
                       edge.kind AS kind,
                       edge.props_json AS props_json
                """,
                {"place_id": str(place_id)},
            )
        else:
            rows = self._query(
                """
                MATCH (p:Place {place_id: $place_id})-[edge:ADJACENT]->(other:Place)
                RETURN elementId(edge) AS edge_id,
                       p.place_id AS src_id,
                       other.place_id AS dst_id,
                       edge.kind AS kind,
                       edge.props_json AS props_json
                """,
                {"place_id": str(place_id)},
            )
        out: list[PlaceEdge] = []
        for row in rows:
            props_raw = row.get("props_json") or "{}"
            props = json.loads(props_raw) if isinstance(props_raw, str) else dict(props_raw)
            out.append(
                PlaceEdge(
                    id=row["edge_id"],
                    src_id=row["src_id"],
                    dst_id=row["dst_id"],
                    kind=str(row.get("kind") or "adjacent"),
                    props=props,
                )
            )
        return out

    def find_places(
        self,
        *,
        fly_id: str = "fly_001",
        label: str | None = None,
        limit: int = 64,
    ) -> list[PlaceNode]:
        """Cypher-filter by props label when set (avoids over-fetch × client filter)."""
        lim = max(1, int(limit))
        if label is None:
            return self.list_places(fly_id=fly_id, limit=lim)
        # props_json stores {"label": "<name>", ...}; marker avoids loading all places.
        marker = f'"label": "{str(label)}"'
        rows = self._query(
            f"""
            MATCH (p:Place {{fly_id: $fly_id}})
            WHERE p.props_json CONTAINS $label_marker
            WITH p
            ORDER BY p.updated_at DESC
            LIMIT $limit
            {self._return_clause()}
            """,
            {
                "fly_id": fly_id,
                "label_marker": marker,
                "limit": lim,
            },
        )
        out = [self._record_to_node(row) for row in rows]
        # Exact label match (CONTAINS can false-positive on substrings).
        want = str(label)
        return [n for n in out if str((n.props or {}).get("label") or "") == want][:lim]

    def find_food_places(
        self,
        *,
        fly_id: str = "fly_001",
        min_value: float = 0.05,
        kinds: tuple[str, ...] | list[str] | None = None,
        limit: int = 32,
    ) -> list[FoodPlaceHit]:
        """Query Neo4j ``(:Place)-[:HAS_SENSE]->(:Sense)`` for food/scent.

        Returns earliest food experiences first (lowest ``last_seen_step``, then
        oldest ``created_at``) so hungry routing can reverse-path to the first
        food memory.
        """
        kind_list = [str(k) for k in (kinds or SUGAR_SENSE_KINDS)]
        # _return_clause() starts with RETURN; fold sense fields into the same RETURN.
        ret = self._return_clause().strip()
        if ret.upper().startswith("RETURN"):
            ret_body = ret[len("RETURN") :].strip()
        else:
            ret_body = ret
        rows = self._query(
            f"""
            MATCH (p:Place {{fly_id: $fly_id}})-[r:HAS_SENSE]->(s:Sense)
            WHERE s.kind IN $kinds AND coalesce(s.value, r.value, 0.0) >= $min_value
            WITH p, s, r,
                 coalesce(s.value, r.value, 0.0) AS value,
                 coalesce(s.last_seen_step, r.last_seen_step) AS first_step
            ORDER BY coalesce(first_step, 1000000000) ASC, p.created_at ASC, value DESC
            LIMIT $limit
            RETURN {ret_body},
                   s.kind AS sense_kind,
                   value AS sense_value,
                   first_step AS sense_step
            """,
            {
                "fly_id": str(fly_id),
                "kinds": kind_list,
                "min_value": float(min_value),
                "limit": max(1, int(limit)),
            },
        )
        hits: list[FoodPlaceHit] = []
        seen: set[str] = set()
        for row in rows:
            node = self._record_to_node(row)
            key = str(node.id)
            if key in seen:
                continue
            seen.add(key)
            step_raw = row.get("sense_step")
            hits.append(
                FoodPlaceHit(
                    place=node,
                    kind=str(row.get("sense_kind") or ""),
                    value=float(row.get("sense_value") or 0.0),
                    first_step=int(step_raw) if step_raw is not None else None,
                )
            )
        if hits:
            return hits
        # Fallback: props.senses on listed places (cache / pre-Sense writes).
        return find_food_places_in_nodes(
            self.list_places(fly_id=fly_id, limit=max(64, int(limit) * 8)),
            min_value=min_value,
            limit=limit,
            kinds=kind_list,
        )

    def shortest_path(
        self,
        src_id: int | str,
        dst_id: int | str,
        *,
        undirected: bool = True,
        max_hops: int = 64,
    ) -> list[int | str] | None:
        """Id-only shortestPath (no edge props) for cheap reachability checks."""
        if str(src_id) == str(dst_id):
            return [src_id]
        hops = max(1, int(max_hops))
        rel = f"-[:ADJACENT*1..{hops}]-" if undirected else f"-[:ADJACENT*1..{hops}]->"
        rows = self._query(
            f"""
            MATCH (a:Place {{place_id: $src_id}}), (b:Place {{place_id: $dst_id}})
            MATCH path = shortestPath((a){rel}(b))
            RETURN [n IN nodes(path) | n.place_id] AS place_ids
            """,
            {"src_id": str(src_id), "dst_id": str(dst_id)},
        )
        if not rows:
            return None
        ids = list(rows[0].get("place_ids") or [])
        return ids if ids else None

    def shortest_path_with_edges(
        self,
        src_id: int | str,
        dst_id: int | str,
        *,
        undirected: bool = True,
        max_hops: int = 64,
    ) -> tuple[list[int | str], list[PathHop]] | None:
        """Neo4j shortestPath over ADJACENT, returning place ids + edge props."""
        if str(src_id) == str(dst_id):
            return [src_id], []
        hops = max(1, int(max_hops))
        rel = f"-[:ADJACENT*1..{hops}]-" if undirected else f"-[:ADJACENT*1..{hops}]->"
        rows = self._query(
            f"""
            MATCH (a:Place {{place_id: $src_id}}), (b:Place {{place_id: $dst_id}})
            MATCH path = shortestPath((a){rel}(b))
            WITH nodes(path) AS ns, relationships(path) AS rs
            RETURN [n IN ns | n.place_id] AS place_ids,
                   [r IN rs | {{
                       src_id: startNode(r).place_id,
                       dst_id: endNode(r).place_id,
                       props_json: coalesce(r.props_json, '{{}}')
                   }}] AS edge_rows
            """,
            {"src_id": str(src_id), "dst_id": str(dst_id)},
        )
        if not rows:
            return None
        ids = list(rows[0].get("place_ids") or [])
        if not ids:
            return None
        edge_rows = rows[0].get("edge_rows") or []
        path_hops: list[PathHop] = []
        for i in range(len(ids) - 1):
            src, dst = ids[i], ids[i + 1]
            matched = None
            for raw in edge_rows:
                if not isinstance(raw, dict):
                    continue
                a, b = str(raw.get("src_id")), str(raw.get("dst_id"))
                if (a == str(src) and b == str(dst)) or (a == str(dst) and b == str(src)):
                    matched = raw
                    break
            props: dict[str, Any] = {}
            reversed_hop = False
            if matched is not None:
                props_raw = matched.get("props_json") or "{}"
                try:
                    props = (
                        json.loads(props_raw)
                        if isinstance(props_raw, str)
                        else dict(props_raw or {})
                    )
                except (TypeError, ValueError, json.JSONDecodeError):
                    props = {}
                reversed_hop = str(matched.get("src_id")) == str(dst)
            path_hops.append(
                PathHop(src_id=src, dst_id=dst, props=props, reversed=reversed_hop)
            )
        return ids, path_hops


def mean_wall_familiarity(walls: dict[str, Any] | None) -> float:
    """Average familiarity across the four chamber walls."""
    if not isinstance(walls, dict) or not walls:
        return 0.0
    vals = [
        _clamp01(float((walls.get(d) or {}).get("familiarity") or 0.0)) for d in WALL_DIRS
    ]
    return float(sum(vals) / max(1, len(vals)))


def decay_mapped_wall_hazard(
    wall_hazard: float,
    *,
    mapped: bool,
    familiarity: float,
    strength: float = 0.85,
    floor: float = 0.15,
) -> float:
    """Deprecated wrapper: wall attenuation now lives in ``MappedSensorFeedback``.

    Kept for tests/back-compat; prefer ``attenuate_mapped_reading``.
    """
    from flybrain.sensors.mapped_feedback import attenuate_mapped_reading

    reported, _scale = attenuate_mapped_reading(
        wall_hazard,
        mapped=mapped,
        familiarity=familiarity,
        strength=strength,
        floor=floor,
    )
    return reported


@dataclass
class PlaceMapObservation:
    """Result of upserting the current view into the place graph (map only)."""

    place: PlaceNode
    similarity: float
    is_new: bool
    mean_familiarity: float
    from_place_id: int | str | None = None
    match_reason: str = "created"  # same_place | adjacent | view_only | created
    edge_created: bool = False
    # Why same_place was refused (novelty / landmark); empty when not split.
    split_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "place_id": self.place.id,
            "similarity": float(self.similarity),
            "is_new": bool(self.is_new),
            "mapped": not bool(self.is_new),
            "mean_familiarity": float(self.mean_familiarity),
            "from_place_id": self.from_place_id,
            "match_reason": self.match_reason,
            "edge_created": bool(self.edge_created),
            "split_reason": self.split_reason,
            "walls": {d: dict(self.place.walls.get(d, {})) for d in WALL_DIRS},
            "items": dict((self.place.props or {}).get("items") or {}),
        }


def _bump_walls_for_revisit(
    prior: PlaceNode,
    *,
    familiarity_bump: float,
    step: int | None,
    walls: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if walls is not None:
        return walls
    bumped: dict[str, dict[str, Any]] = {}
    for d in WALL_DIRS:
        wall = dict(prior.walls.get(d) or default_walls()[d])
        wall["familiarity"] = _clamp01(
            float(wall.get("familiarity") or 0.0) + float(familiarity_bump)
        )
        if step is not None:
            wall["last_seen_step"] = int(step)
        bumped[d] = wall
    return bumped


def observe_place(
    store: PlaceGraphStore,
    view_vec: np.ndarray,
    *,
    fly_id: str = "fly_001",
    match_min: float = 0.97,
    step: int | None = None,
    familiarity_bump: float = 0.12,
    props: dict[str, Any] | None = None,
    walls: dict[str, Any] | None = None,
    from_place_id: int | str | None = None,
    context_enabled: bool = True,
    context_top_k: int = 8,
    edge_props: dict[str, Any] | None = None,
    leave_same_place: bool = False,
    split_reason: str = "",
) -> PlaceMapObservation:
    """Resolve current view into a Place, using path context when enabled.

    Matching order when ``context_enabled`` and ``from_place_id`` is set:

    1. Stay on the same place if the view still matches (micro-motion), unless
       ``leave_same_place`` (landmark / novelty split).
    2. Prefer a lookalike already linked by ``ADJACENT`` to/from the previous
       place (re-enter a known room via the same path, in either direction).
    3. Otherwise **create** a new Place even if a lookalike exists — identical
       rooms reached from different references stay distinct.

    When context is off / cold start (no ``from_place_id``), falls back to
    pure view matching.

    A place keeps the view it was created with: revisits never move it, so a
    gradual walk (box → tunnel → outside) leaves it and maps new places.
    """
    top_k = max(1, int(context_top_k if context_enabled else 1))
    candidates = store.nearest(view_vec, fly_id=fly_id, top_k=top_k)
    chosen: PlaceNode | None = None
    similarity = 0.0
    reason = "created"
    used_split = ""

    use_context = bool(context_enabled) and from_place_id is not None
    if use_context:
        if not bool(leave_same_place):
            for node, sim in candidates:
                if str(node.id) == str(from_place_id) and float(sim) >= float(match_min):
                    chosen = node
                    similarity = float(sim)
                    reason = "same_place"
                    break
        elif split_reason:
            used_split = str(split_reason)
        if chosen is None:
            best_adj: tuple[PlaceNode, float] | None = None
            for node, sim in candidates:
                if float(sim) < float(match_min):
                    continue
                if store.has_adjacent(from_place_id, node.id) or store.has_adjacent(
                    node.id, from_place_id
                ):
                    if best_adj is None or float(sim) > best_adj[1]:
                        best_adj = (node, float(sim))
            if best_adj is not None:
                chosen, similarity = best_adj
                reason = "adjacent"
        # Lookalike without reference edge → leave chosen None (create below).
    elif candidates and float(candidates[0][1]) >= float(match_min):
        chosen = candidates[0][0]
        similarity = float(candidates[0][1])
        reason = "view_only"

    prop_map = dict(props or {})
    if chosen is not None:
        walls_update = _bump_walls_for_revisit(
            chosen,
            familiarity_bump=familiarity_bump,
            step=step,
            walls=walls,
        )
        place = store.refresh_place(
            chosen.id,
            view_vec,
            walls=walls_update,
            props=prop_map,
            step=step,
        )
        if place is None:  # pragma: no cover
            place = store.create_place(
                view_vec, fly_id=fly_id, walls=walls_update, props=prop_map, step=step
            )
            reason = "created"
            is_new = True
        else:
            is_new = False
    else:
        if candidates:
            similarity = float(candidates[0][1])
        place = store.create_place(
            view_vec,
            fly_id=fly_id,
            walls=walls,
            props=prop_map,
            step=step,
        )
        is_new = True
        reason = "created"

    edge_created = False
    if from_place_id is not None and str(from_place_id) != str(place.id):
        existed = store.has_adjacent(from_place_id, place.id)
        store.upsert_adjacent(from_place_id, place.id, props=edge_props, step=step)
        edge_created = not existed

    return PlaceMapObservation(
        place=place,
        similarity=float(similarity),
        is_new=bool(is_new),
        mean_familiarity=mean_wall_familiarity(place.walls),
        from_place_id=from_place_id,
        match_reason=reason,
        edge_created=edge_created,
        split_reason=used_split if reason != "same_place" else "",
    )


def open_place_graph(
    *,
    backend: str = "neo4j",
    uri: str = "bolt://127.0.0.1:7687",
    user: str = "neo4j",
    password: str | None = None,
    database: str = "neo4j",
    view_dim: int = PATCH_VEC_DIM,
    cache: dict[str, Any] | Any | None = None,
) -> PlaceGraphStore:
    """Open the configured mental-map backend.

    Neo4j opens behind a two-layer disk cache (L1=10s working memory, L2=24h STM)
    unless ``cache.enabled`` is false.
    """
    from flybrain.memory.place_cache import CachedPlaceGraph, PlaceCacheConfig

    kind = str(backend or "neo4j").strip().lower()
    if kind == "in_memory":
        return InMemoryPlaceGraph(view_dim=int(view_dim))
    if kind != "neo4j":
        raise ValueError(f"unsupported place graph backend: {backend!r}")
    inner: PlaceGraphStore = Neo4jPlaceGraph(
        uri=uri,
        user=user,
        password=password or os.getenv("NEO4J_PASSWORD", "flymindgraph"),
        database=database,
        view_dim=int(view_dim),
    )
    if isinstance(cache, PlaceCacheConfig):
        cache_cfg = cache
    else:
        cache_cfg = PlaceCacheConfig.from_dict(cache)
    if not cache_cfg.enabled:
        return inner
    # Resolve relative cache dir against the python package parent (repo python/).
    if not Path(cache_cfg.dir).is_absolute():
        cache_cfg.dir = str(Path(__file__).resolve().parents[2] / cache_cfg.dir)
    return CachedPlaceGraph(inner=inner, config=cache_cfg, view_dim=int(view_dim))
