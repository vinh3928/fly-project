"""Follow remembered ADJACENT places back into the box when hungry.

Sugar chemotaxis aims straight at the food through walls. This tracker instead
walks the place graph (opening-aware shortest path) to a home place:

1. Prefer reachable contact sugar / enclosed sugar puddle when known.
2. Else reverse-path to the session trail smell peak — including outdoor
   ``odor_plume`` / ``outside`` places where the plume peaked (not permanent
   bookmarks; trail-only so a distant stale puddle cannot beat a nearby plume).
3. Else bookmarked enclosed smell site, graph sugar_smell, or the chamber place
   left just before the fly first stepped ``outside``.

While routing, rising smell reinforces the current heading (blend antenna bearing);
walls outside bias steering toward an opening (chamber entrance) instead of
pushing through the wall.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from flybrain.debug.route_path_channel import world_bearing_turn
from flybrain.memory.place_graph import (
    SUGAR_SENSE_KINDS,
    PathHop,
    PlaceEdge,
    PlaceGraphStore,
    PlaceNode,
    edge_toward,
    path_hops_from_ids,
)
from flybrain.thinking.composer import is_hungry

ROUTE_TURN_MAX = 1.0
OPENING_TURN_GAIN = 0.75
EDGE_TURN_MIN = 0.05
DEFAULT_HANDOFF_SUGAR = 0.35
# Contact taste (feet/proboscis) high enough to stop routing and let sugar_search feed.
# Matches Unity feed gate and SugarSearchConfig.feed_taste_min.
DEFAULT_ON_SUGAR_TASTE = 0.2
DEFAULT_FORWARD = 0.55
# Live sugar smell strong enough to bookmark the current place as food home.
DEFAULT_BOOKMARK_SUGAR_SMELL = 0.25
# Views this open are meadow/tunnel-mouth, not enclosed food sites.
DEFAULT_MAX_OPENING_FOR_HOME = 0.4
# Session trail: remember places with smell so hunger can reverse-path to them.
DEFAULT_TRAIL_MAX = 64
DEFAULT_TRAIL_SMELL_MIN = 0.08
DEFAULT_RISE_EPS = 0.002
DEFAULT_SMELL_BLEND = 0.55
DEFAULT_WALL_AVOID = 0.45
DEFAULT_ENTRANCE_OPENING_MIN = 0.25
DEFAULT_WALL_PATH_PENALTY = 1.5
DEFAULT_OPENING_PATH_BONUS = 0.75
# Landmark scan when hungry but home/path/edge is missing (degrees per mind tick).
# Continuous one-way body spin through a full circle; head leads ~one step in Unity.
DEFAULT_SCAN_STEP_DEGREES = 20.0
DEFAULT_SCAN_SPAN_DEGREES = 360.0
# Body yaw ≈ 20° over ~0.8 s mind_interval (40×0.02 s) at Unity turnSpeed=4 rad/s.
DEFAULT_SCAN_TURN = 0.12
DEFAULT_SCAN_LANDMARK_OPENING = 0.35
DEFAULT_SCAN_LANDMARK_NOVELTY = 0.25
# Once a route turn starts, hold its sign until ≥min body yaw AND hops drop
# (or max yaw if the place graph is thrashing A↔B).
DEFAULT_MIN_TURN_DEGREES = 180.0
DEFAULT_MAX_TURN_DEGREES = 360.0
# |turn| while resisting an opposite edge (may meet pivot_turn).
DEFAULT_COMMIT_TURN = 0.8
# Keep crawling while committed so hops can drop (hard pivot at 0.8 was a spin lock).
DEFAULT_COMMIT_FORWARD_MIN = 0.3
# After a full max-yaw commit with no hop progress, soft-steer this many ticks
# instead of immediately re-arming the opposite pivot.
DEFAULT_COMMIT_ESCAPE_TICKS = 2
# Soft |turn| cap during escape (below pivot_turn so forward stays up).
DEFAULT_COMMIT_ESCAPE_TURN = 0.55
# Body yaw degrees per mind tick at |turn|=1 with mind_interval≈20 (20×0.02s × 4 rad/s).
# Using the old 183 (mi=40) drained the 180° budget in ~2 ticks and re-armed the flip.
DEFAULT_DEG_PER_UNIT_TURN = 90.0
# Only these labels may become permanent food home from smell / sugar sense.
_FOOD_HOME_LABELS = frozenset({"puddle", "place"})
# Session-trail reverse may also target where an outdoor plume peaked.
_TRAIL_PLUME_LABELS = frozenset({"outside", "odor_plume"})
# Chamber return targets (odor_plume is often outside the box — excluded).
_CHAMBER_LABELS = frozenset({"place", "puddle", "wall"})


@dataclass
class RouteHomeConfig:
    enabled: bool = True
    max_hops: int = 64
    handoff_sugar_min: float = DEFAULT_HANDOFF_SUGAR
    on_sugar_taste_min: float = DEFAULT_ON_SUGAR_TASTE
    cruise_forward: float = DEFAULT_FORWARD
    pivot_turn: float = 0.8
    bookmark_sugar_smell: float = DEFAULT_BOOKMARK_SUGAR_SMELL
    max_opening_for_home: float = DEFAULT_MAX_OPENING_FOR_HOME
    # Reverse-path smell trail + reinforce-on-rise + wall/entrance steering.
    trail_max: int = DEFAULT_TRAIL_MAX
    trail_smell_min: float = DEFAULT_TRAIL_SMELL_MIN
    rise_eps: float = DEFAULT_RISE_EPS
    smell_blend: float = DEFAULT_SMELL_BLEND
    wall_avoid: float = DEFAULT_WALL_AVOID
    entrance_opening_min: float = DEFAULT_ENTRANCE_OPENING_MIN
    wall_path_penalty: float = DEFAULT_WALL_PATH_PENALTY
    opening_path_bonus: float = DEFAULT_OPENING_PATH_BONUS
    # Neo4j / graph Sense query when hungry — sugar + sugar_smell only (no generic odor).
    graph_food_enabled: bool = True
    graph_food_min: float = 0.05
    prefer_first_food: bool = True  # earliest sugar experience over strongest smell
    # Reverse-path to outdoor/odor_plume session-trail smell peaks (not bookmarks).
    trail_plume_home: bool = True
    # Outdoor plume must be at least this strong to become a trail reverse target.
    trail_plume_smell_min: float = DEFAULT_BOOKMARK_SUGAR_SMELL
    # Hungry but no home / unreachable / missing edge → body spins for landmarks.
    landmark_scan: bool = True
    scan_step_degrees: float = DEFAULT_SCAN_STEP_DEGREES
    scan_span_degrees: float = DEFAULT_SCAN_SPAN_DEGREES
    scan_turn: float = DEFAULT_SCAN_TURN
    scan_landmark_opening: float = DEFAULT_SCAN_LANDMARK_OPENING
    scan_landmark_novelty: float = DEFAULT_SCAN_LANDMARK_NOVELTY
    # Hold turn sign until ≥min yaw and hops progress (0 disables). Cap at max.
    min_turn_degrees: float = DEFAULT_MIN_TURN_DEGREES
    max_turn_degrees: float = DEFAULT_MAX_TURN_DEGREES
    commit_turn: float = DEFAULT_COMMIT_TURN
    commit_forward_min: float = DEFAULT_COMMIT_FORWARD_MIN
    commit_escape_ticks: int = DEFAULT_COMMIT_ESCAPE_TICKS
    commit_escape_turn: float = DEFAULT_COMMIT_ESCAPE_TURN
    deg_per_unit_turn: float = DEFAULT_DEG_PER_UNIT_TURN
    # Prefer Unity world XYZ bearing to the next place over remembered edge turns.
    world_steer: bool = True
    world_steer_gain: float = 1.0
    world_arrive_m: float = 0.08  # checkpoint radius in XZ (meters)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> RouteHomeConfig:
        data = dict(raw or {})
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class SmellTrailEntry:
    """One visited place with the smell seen there (for reverse pathing)."""

    place_id: int | str
    smell: float
    label: str = ""
    wall: float = 0.0
    opening: float = 0.0


@dataclass
class RouteHomeState:
    mode: str = "off"  # off | route | handoff | scan
    turn: float | None = None
    forward: float = DEFAULT_FORWARD
    home_id: int | str | None = None
    next_id: int | str | None = None
    hops_left: int = 0
    path: list[int | str] = field(default_factory=list)
    path_edges: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""
    smell_delta: float = 0.0
    food_kind: str = ""
    food_value: float = 0.0
    scan_yaw_degrees: float = 0.0
    commit_degrees_left: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "turn": self.turn,
            "forward": self.forward,
            "home_id": self.home_id,
            "next_id": self.next_id,
            "hops_left": self.hops_left,
            "path": [str(p) for p in self.path],
            "path_edges": list(self.path_edges),
            "reason": self.reason,
            "smell_delta": float(self.smell_delta),
            "food_kind": self.food_kind,
            "food_value": float(self.food_value),
            "scan_yaw_degrees": float(self.scan_yaw_degrees),
            "commit_degrees_left": float(self.commit_degrees_left),
        }

    @property
    def active(self) -> bool:
        return self.mode in ("route", "scan") and self.turn is not None


def _live_smell(signals: dict[str, Any]) -> float:
    """Sugar-odor intensity only (antenna sugar_smell or contact sugar)."""
    return float(signals.get("sugar_smell", signals.get("sugar", 0.0)) or 0.0)


def _contact_sugar(signals: dict[str, Any]) -> float:
    """Feet/proboscis sugar taste (not antenna smell)."""
    return max(
        float(signals.get("sugar", 0.0) or 0.0),
        float(signals.get("taste_sugar", 0.0) or 0.0),
    )


def _world_xyz_of(node: PlaceNode | None) -> np.ndarray | None:
    """Place props ``world_xyz`` / ``position`` → length-3 array."""
    if node is None:
        return None
    props = node.props or {}
    raw = props.get("world_xyz", props.get("position"))
    if raw is None:
        return None
    try:
        arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    return arr[:3].copy()


def _fly_pose_from_signals(
    signals: dict[str, Any],
) -> tuple[np.ndarray | None, float | None]:
    """Optional live fly XYZ + yaw from signals (Unity-mapped)."""
    xyz = signals.get("fly_xyz")
    yaw = signals.get("fly_yaw_deg")
    pos: np.ndarray | None = None
    if xyz is not None:
        try:
            arr = np.asarray(xyz, dtype=np.float32).reshape(-1)
            if arr.size >= 3 and np.all(np.isfinite(arr[:3])):
                pos = arr[:3].copy()
        except Exception:
            pos = None
    yaw_f: float | None = None
    if yaw is not None:
        try:
            yaw_f = float(yaw)
            if not np.isfinite(yaw_f):
                yaw_f = None
        except Exception:
            yaw_f = None
    return pos, yaw_f


def _xz_dist_sq(a: np.ndarray, b: np.ndarray) -> float:
    dx = float(a[0]) - float(b[0])
    dz = float(a[2]) - float(b[2])
    return dx * dx + dz * dz


def shortest_path_wall_aware(
    edges: list[PlaceEdge],
    src_id: int | str,
    dst_id: int | str,
    *,
    undirected: bool = True,
    max_hops: int = 64,
    wall_of: Callable[[int | str], float] | None = None,
    wall_penalty: float = DEFAULT_WALL_PATH_PENALTY,
    opening_bonus: float = DEFAULT_OPENING_PATH_BONUS,
) -> list[int | str] | None:
    """Dijkstra path that prefers openings and avoids high-wall places.

    Hop base cost is 1; destination wall raises cost; edge ``opening_strength``
    lowers it so the tunnel mouth into the chamber wins over meadow detours.
    """
    src, dst = str(src_id), str(dst_id)
    if src == dst:
        return [src_id]
    adj: dict[str, list[tuple[str, PlaceEdge]]] = {}
    id_map: dict[str, int | str] = {src: src_id, dst: dst_id}
    for edge in edges:
        a, b = str(edge.src_id), str(edge.dst_id)
        id_map[a] = edge.src_id
        id_map[b] = edge.dst_id
        adj.setdefault(a, []).append((b, edge))
        if undirected:
            adj.setdefault(b, []).append((a, edge))

    wall_fn = wall_of or (lambda _pid: 0.0)
    limit = max(1, int(max_hops))
    # cost, depth, node
    best: dict[str, tuple[float, int]] = {src: (0.0, 0)}
    prev: dict[str, str | None] = {src: None}
    # Linear scan (graphs are tiny); avoid heapq import noise for tests.
    frontier = {src}
    while frontier:
        cur = min(frontier, key=lambda n: best[n][0])
        frontier.remove(cur)
        cost, depth = best[cur]
        if cur == dst:
            break
        if depth >= limit:
            continue
        for nxt, edge in adj.get(cur, []):
            opening = float((edge.props or {}).get("opening_strength") or 0.0)
            wall = float(wall_fn(id_map.get(nxt, nxt)))
            step_cost = 1.0 + wall_penalty * max(0.0, wall) - opening_bonus * max(0.0, opening)
            step_cost = max(0.05, step_cost)
            new_cost = cost + step_cost
            new_depth = depth + 1
            prev_best = best.get(nxt)
            if prev_best is None or new_cost < prev_best[0] - 1e-9:
                best[nxt] = (new_cost, new_depth)
                prev[nxt] = cur
                frontier.add(nxt)

    if dst not in prev and src != dst:
        return None
    if src == dst:
        return [src_id]
    if dst not in prev:
        return None
    path_ids = [dst]
    node: str | None = dst
    while node is not None and node != src:
        node = prev.get(node)
        if node is not None:
            path_ids.append(node)
    if not path_ids or path_ids[-1] != src:
        return None
    path_ids.reverse()
    return [id_map.get(p, p) for p in path_ids]


@dataclass
class RouteHomeTracker:
    """Session path-home planner over a place graph store."""

    config: RouteHomeConfig = field(default_factory=RouteHomeConfig)
    fly_id: str = "fly_001"
    _chamber_anchor_id: int | str | None = field(default=None, init=False, repr=False)
    _home_place_id: int | str | None = field(default=None, init=False, repr=False)
    _last_place_id: int | str | None = field(default=None, init=False, repr=False)
    _last_label: str = field(default="", init=False, repr=False)
    _trail: deque[SmellTrailEntry] = field(default_factory=deque, init=False, repr=False)
    _peak_smell_id: int | str | None = field(default=None, init=False, repr=False)
    _peak_smell: float = field(default=0.0, init=False, repr=False)
    _last_route_smell: float | None = field(default=None, init=False, repr=False)
    _path_cache_key: tuple[str, str] | None = field(default=None, init=False, repr=False)
    _path_cache: tuple[list[int | str], list[PathHop]] | None = field(
        default=None, init=False, repr=False
    )
    # One return path per hungry bout — low_energy then hunger must not spawn a
    # second home/polyline while this is set.
    _active_home_id: int | str | None = field(default=None, init=False, repr=False)
    _active_home_reason: str = field(default="", init=False, repr=False)
    _active_food_kind: str = field(default="", init=False, repr=False)
    _active_food_value: float = field(default=0.0, init=False, repr=False)
    _scan_side: float = field(default=1.0, init=False, repr=False)
    _scan_yaw_degrees: float = field(default=0.0, init=False, repr=False)
    _prev_scan_opening: float | None = field(default=None, init=False, repr=False)
    _commit_sign: float = field(default=0.0, init=False, repr=False)
    _commit_yaw: float = field(default=0.0, init=False, repr=False)
    _commit_hops_at_start: int | None = field(default=None, init=False, repr=False)
    _commit_degrees_left: float = field(default=0.0, init=False, repr=False)
    _escape_ticks_left: int = field(default=0, init=False, repr=False)
    _commit_escaped_this_tick: bool = field(default=False, init=False, repr=False)
    _state: RouteHomeState = field(default_factory=RouteHomeState, init=False)

    @property
    def route_in_progress(self) -> bool:
        """True while a return path is locked or actively steering/scanning.

        Stimulus interrupts must not freeze the body during this window.
        """
        if self._active_home_id is not None:
            return True
        return self._state.active

    def reset(self) -> None:
        self._chamber_anchor_id = None
        self._home_place_id = None
        self._last_place_id = None
        self._last_label = ""
        self._trail = deque(maxlen=max(4, int(self.config.trail_max)))
        self._peak_smell_id = None
        self._peak_smell = 0.0
        self._last_route_smell = None
        self._path_cache_key = None
        self._path_cache = None
        self._clear_active_route()
        self._reset_scan()
        self._stop_turn_commit()
        self._state = RouteHomeState()

    def _reset_scan(self) -> None:
        self._scan_side = 1.0
        self._scan_yaw_degrees = 0.0
        self._prev_scan_opening = None

    def _reset_turn_commit(self) -> None:
        self._commit_sign = 0.0
        self._commit_yaw = 0.0
        self._commit_hops_at_start = None
        self._commit_degrees_left = 0.0

    def _clear_commit_escape(self) -> None:
        self._escape_ticks_left = 0
        self._commit_escaped_this_tick = False

    def _stop_turn_commit(self) -> None:
        """Drop commit and any soft-escape window (handoff / scan / off)."""
        self._reset_turn_commit()
        self._clear_commit_escape()

    def _soft_escape_turn(self, raw: float) -> float:
        """Cap |turn| below pivot so cruise forward stays nonzero."""
        cap = float(
            np.clip(
                float(self.config.commit_escape_turn),
                EDGE_TURN_MIN,
                max(EDGE_TURN_MIN, float(self.config.pivot_turn) - 1e-3),
            )
        )
        return float(np.clip(raw, -cap, cap))

    def _commit_turn(self, turn: float, *, hops_left: int | None = None) -> float:
        """Hold turn sign through meadow place-flip thrash.

        Release only when:
        - commanded yaw ≥ ``min_turn_degrees`` AND ``hops_left`` dropped, or
        - commanded yaw ≥ ``max_turn_degrees`` (safety if the graph never advances).

        A max-yaw release with no hop progress starts a short soft-escape window
        so the opposite edge cannot immediately re-arm another fwd=0 pivot.

        Degree accounting uses ``deg_per_unit_turn`` (~90 for mind_interval 20).
        """
        self._commit_escaped_this_tick = False
        min_deg = float(self.config.min_turn_degrees)
        raw = float(np.clip(turn, -ROUTE_TURN_MAX, ROUTE_TURN_MAX))
        if min_deg <= 0.0:
            return raw

        max_deg = max(min_deg, float(self.config.max_turn_degrees))
        deg_per = max(1.0, float(self.config.deg_per_unit_turn))
        floor = float(np.clip(float(self.config.commit_turn), EDGE_TURN_MIN, ROUTE_TURN_MAX))

        if self._commit_sign != 0.0:
            progressed = (
                hops_left is not None
                and self._commit_hops_at_start is not None
                and int(hops_left) < int(self._commit_hops_at_start)
            )
            at_max = self._commit_yaw >= max_deg
            if (self._commit_yaw >= min_deg and progressed) or at_max:
                if at_max and not progressed:
                    self._escape_ticks_left = max(1, int(self.config.commit_escape_ticks))
                self._reset_turn_commit()

        if self._escape_ticks_left > 0:
            self._escape_ticks_left -= 1
            self._commit_escaped_this_tick = True
            self._commit_degrees_left = 0.0
            return self._soft_escape_turn(raw)

        if self._commit_sign == 0.0:
            if abs(raw) < EDGE_TURN_MIN:
                self._commit_degrees_left = 0.0
                return raw
            self._commit_sign = 1.0 if raw >= 0.0 else -1.0
            self._commit_yaw = 0.0
            self._commit_hops_at_start = None if hops_left is None else int(hops_left)
            cmd = raw
        else:
            sign = float(self._commit_sign)
            same_way = abs(raw) >= EDGE_TURN_MIN and (
                (raw >= 0.0 and sign > 0.0) or (raw < 0.0 and sign < 0.0)
            )
            cmd = raw if same_way else float(np.clip(sign * floor, -ROUTE_TURN_MAX, ROUTE_TURN_MAX))

        self._commit_yaw = float(self._commit_yaw) + abs(cmd) * deg_per
        # Remaining until we *may* release (min), then until hard cap (max).
        if self._commit_yaw < min_deg:
            self._commit_degrees_left = min_deg - self._commit_yaw
        else:
            self._commit_degrees_left = max(0.0, max_deg - self._commit_yaw)
        return float(cmd)

    def _route_forward(self, turn: float, *, smell_delta: float) -> float:
        """Cruise forward with smell bias; crawl while committed / escaping."""
        forward = float(self.config.cruise_forward)
        if smell_delta >= float(self.config.rise_eps):
            forward = float(np.clip(forward + 0.15, 0.0, 1.0))
        elif smell_delta <= -float(self.config.rise_eps):
            forward = float(np.clip(forward * 0.7, 0.0, 1.0))
        pivot = max(1e-6, float(self.config.pivot_turn))
        if abs(turn) >= pivot:
            scaled = 0.0
        else:
            scaled = float(np.clip(forward * (1.0 - abs(turn) / pivot), 0.0, 1.0))
        floor = 0.0
        if self._commit_sign != 0.0 or self._commit_escaped_this_tick:
            floor = float(np.clip(float(self.config.commit_forward_min), 0.0, 1.0))
        return float(np.clip(max(scaled, floor), 0.0, 1.0))

    def __post_init__(self) -> None:
        self._trail = deque(maxlen=max(4, int(self.config.trail_max)))
        self._path_cache_key = None
        self._path_cache = None

    @property
    def state(self) -> RouteHomeState:
        return self._state

    @property
    def trail(self) -> list[SmellTrailEntry]:
        return list(self._trail)

    def _sugar_on_place(self, place: PlaceNode) -> float:
        senses = (place.props or {}).get("senses") or {}
        if not isinstance(senses, dict):
            return 0.0
        for key in ("sugar", "sugar_smell"):
            row = senses.get(key) or {}
            if isinstance(row, dict) and float(row.get("value") or 0.0) > 0.0:
                return float(row.get("value") or 0.0)
        return 0.0

    def _smell_on_place(self, place: PlaceNode | None) -> float:
        if place is None:
            return 0.0
        senses = (place.props or {}).get("senses") or {}
        if not isinstance(senses, dict):
            return 0.0
        best = 0.0
        for key in SUGAR_SENSE_KINDS:
            row = senses.get(key) or {}
            if isinstance(row, dict):
                best = max(best, float(row.get("value") or 0.0))
        return best

    def _wall_on_place(self, place: PlaceNode | None) -> float:
        if place is None:
            return 0.0
        senses = (place.props or {}).get("senses") or {}
        if not isinstance(senses, dict):
            return 0.0
        row = senses.get("wall_hazard") or {}
        if not isinstance(row, dict):
            return 0.0
        return float(row.get("value") or 0.0)

    def _opening_on_place(
        self, place: PlaceNode | None, signals: dict[str, Any] | None = None
    ) -> float:
        if signals is not None:
            live = signals.get("opening_strength")
            if live is not None:
                return float(live or 0.0)
        if place is None:
            return 0.0
        senses = (place.props or {}).get("senses") or {}
        if not isinstance(senses, dict):
            return 0.0
        row = senses.get("opening_strength") or {}
        if not isinstance(row, dict):
            return 0.0
        return float(row.get("value") or 0.0)

    def _place_label(self, place: PlaceNode | None, fallback: str = "") -> str:
        if place is None:
            return str(fallback or "")
        return str((place.props or {}).get("label") or fallback or "")

    def _is_enclosed_food_site(
        self,
        place: PlaceNode | None,
        label: str,
        signals: dict[str, Any] | None = None,
    ) -> bool:
        """True for chamber puddle/place with a mostly enclosed view (not meadow plume)."""
        lab = self._place_label(place, label)
        if lab == "outside" or lab not in _FOOD_HOME_LABELS:
            return False
        opening = self._opening_on_place(place, signals)
        return opening < float(self.config.max_opening_for_home)

    def _is_valid_home_node(
        self,
        store: PlaceGraphStore,
        place_id: int | str,
        *,
        require_food_label: bool,
    ) -> bool:
        """Permanent / chamber homes: enclosed only (never outdoor plume bookmarks)."""
        node = store.get(place_id)
        if node is None:
            return False
        lab = self._place_label(node)
        if lab in _TRAIL_PLUME_LABELS or lab == "outside":
            return False
        if require_food_label and lab not in _FOOD_HOME_LABELS:
            return False
        if not require_food_label and lab not in _CHAMBER_LABELS:
            return False
        return self._opening_on_place(node) < float(self.config.max_opening_for_home)

    def _is_trail_reverse_node(
        self,
        store: PlaceGraphStore,
        place_id: int | str,
        *,
        smell: float | None = None,
    ) -> bool:
        """Session-trail reverse target: enclosed food site or strong outdoor plume."""
        node = store.get(place_id)
        if node is None:
            return False
        lab = self._place_label(node)
        if lab in _TRAIL_PLUME_LABELS:
            if not bool(self.config.trail_plume_home):
                return False
            if smell is None:
                return True
            return float(smell) >= float(self.config.trail_plume_smell_min)
        return self._is_valid_home_node(store, place_id, require_food_label=True)

    def _record_trail(
        self,
        place: PlaceNode,
        label: str,
        signals: dict[str, Any],
    ) -> None:
        """Append a trail sample when the place changes or smell is notable."""
        place_id = place.id
        smell = _live_smell(signals)
        wall = float(signals.get("wall_hazard", 0.0) or 0.0)
        opening = float(signals.get("opening_strength", 0.0) or 0.0)
        lab = self._place_label(place, label)
        changed = self._last_place_id is None or str(self._last_place_id) != str(place_id)
        notable = smell >= float(self.config.trail_smell_min)
        if not changed and not notable:
            return
        if (
            self._trail
            and str(self._trail[-1].place_id) == str(place_id)
            and abs(self._trail[-1].smell - smell) < 1e-4
        ):
            return
        entry = SmellTrailEntry(
            place_id=place_id,
            smell=smell,
            label=lab,
            wall=wall,
            opening=opening,
        )
        if self._trail and str(self._trail[-1].place_id) == str(place_id):
            self._trail[-1] = entry
        else:
            self._trail.append(entry)
        # Peak smell (enclosed food or strong outdoor plume) → reverse-path target.
        enclosed_peak = smell >= float(self.config.trail_smell_min) and self._is_enclosed_food_site(
            place, lab, signals
        )
        plume_peak = (
            bool(self.config.trail_plume_home)
            and lab in _TRAIL_PLUME_LABELS
            and smell >= float(self.config.trail_plume_smell_min)
        )
        if (enclosed_peak or plume_peak) and smell >= self._peak_smell - 1e-6:
            self._peak_smell = smell
            self._peak_smell_id = place_id

    def _update_anchors(
        self,
        place: PlaceNode | None,
        label: str,
        signals: dict[str, Any],
        store: PlaceGraphStore | None,
    ) -> None:
        if place is None:
            return
        place_id = place.id
        sugar_smell = float(signals.get("sugar_smell", signals.get("sugar", 0.0)) or 0.0)
        tasted = float(signals.get("taste_sugar", 0.0) or 0.0)
        lab = self._place_label(place, label)

        self._record_trail(place, lab, signals)

        # Food home: only enclosed puddle/place — never outdoor odor_plume (last run's bug).
        if self._is_enclosed_food_site(place, lab, signals):
            if lab == "puddle" and self._sugar_on_place(place) > 0.0:
                self._home_place_id = place_id
            elif self._sugar_on_place(place) >= 0.05 or tasted >= 0.05:
                self._home_place_id = place_id
            elif sugar_smell >= float(self.config.bookmark_sugar_smell):
                self._home_place_id = place_id

        # Chamber anchors: enclosed place/puddle/wall only (not odor_plume).
        if lab in _CHAMBER_LABELS and self._opening_on_place(place, signals) < float(
            self.config.max_opening_for_home
        ):
            if self._chamber_anchor_id is None:
                self._chamber_anchor_id = place_id

        # First step onto outside this outing → remember the previous chamber place.
        if label == "outside" and self._last_label != "outside" and self._last_place_id is not None:
            if self._chamber_anchor_id is None and self._last_label in _CHAMBER_LABELS:
                self._chamber_anchor_id = self._last_place_id

        # Run started already outside: adopt a reachable interior place from the graph.
        if label == "outside" and self._chamber_anchor_id is None and store is not None:
            self._chamber_anchor_id = self._find_chamber_in_graph(store, place_id)

        self._last_place_id = place_id
        self._last_label = label

    def _find_chamber_in_graph(
        self, store: PlaceGraphStore, from_id: int | str
    ) -> int | str | None:
        """Nearest reachable enclosed chamber place (puddle/place/wall)."""
        for label in ("puddle", "place", "wall"):
            for node in store.find_places(fly_id=self.fly_id, label=label, limit=32):
                if str(node.id) == str(from_id):
                    continue
                if self._opening_on_place(node) >= float(self.config.max_opening_for_home):
                    continue
                if self._reachable(store, from_id, node.id) and str(node.id) != str(from_id):
                    return node.id
        for label in ("puddle", "place", "wall"):
            for node in store.find_places(fly_id=self.fly_id, label=label, limit=8):
                if str(node.id) == str(from_id):
                    continue
                if self._opening_on_place(node) >= float(self.config.max_opening_for_home):
                    continue
                return node.id
        return None

    def _clear_path_cache(self) -> None:
        self._path_cache_key = None
        self._path_cache = None

    def _clear_active_route(self) -> None:
        self._active_home_id = None
        self._active_home_reason = ""
        self._active_food_kind = ""
        self._active_food_value = 0.0

    def _commit_active_home(
        self,
        home_id: int | str,
        reason: str,
        food_kind: str,
        food_value: float,
    ) -> None:
        """Lock the first home of this hungry bout (idempotent while locked)."""
        if self._active_home_id is not None:
            return
        self._active_home_id = home_id
        self._active_home_reason = str(reason or "")
        self._active_food_kind = str(food_kind or "")
        self._active_food_value = float(food_value or 0.0)

    def _has_checkpoint_route(self, home_id: int | str) -> bool:
        """True when a cached checkpoint path toward ``home_id`` is still active."""
        if self._path_cache is None or self._path_cache_key is None:
            return False
        if self._path_cache_key[1] != str(home_id):
            return False
        return len(self._path_cache[0]) >= 1

    def _reachable(self, store: PlaceGraphStore, src_id: int | str, dst_id: int | str) -> bool:
        """Cheap reachability via id-only shortest_path (no full edge dump)."""
        if str(src_id) == str(dst_id):
            return True
        path = store.shortest_path(
            src_id, dst_id, undirected=True, max_hops=int(self.config.max_hops)
        )
        return path is not None

    def _path_with_edges(
        self, store: PlaceGraphStore, src_id: int | str, dst_id: int | str
    ) -> tuple[list[int | str], list[PathHop]] | None:
        """Prefer Neo4j ADJACENT shortestPath + edge props; cache (src→dst)."""
        key = (str(src_id), str(dst_id))
        if self._path_cache_key == key and self._path_cache is not None:
            return self._path_cache

        detailed = getattr(store, "shortest_path_with_edges", None)
        if callable(detailed):
            got = detailed(
                src_id,
                dst_id,
                undirected=True,
                max_hops=int(self.config.max_hops),
            )
            if got is not None:
                self._path_cache_key = key
                self._path_cache = got
                return got

        # Fallback (in-memory): load edges once for wall-aware + hop props.
        edges = store.edges()

        def wall_of(pid: int | str) -> float:
            return self._wall_on_place(store.get(pid))

        path = shortest_path_wall_aware(
            edges,
            src_id,
            dst_id,
            undirected=True,
            max_hops=int(self.config.max_hops),
            wall_of=wall_of,
            wall_penalty=float(self.config.wall_path_penalty),
            opening_bonus=float(self.config.opening_path_bonus),
        )
        if path is None:
            path = store.shortest_path(
                src_id, dst_id, undirected=True, max_hops=int(self.config.max_hops)
            )
        if path is None:
            return None
        got = (path, path_hops_from_ids(edges, path))
        self._path_cache_key = key
        self._path_cache = got
        return got

    def _store_path_cache(
        self,
        current_id: int | str,
        home_id: int | str,
        path: list[int | str],
        hops: list[PathHop],
    ) -> tuple[list[int | str], list[PathHop]]:
        trimmed_hops = list(hops)
        if len(trimmed_hops) > max(0, len(path) - 1):
            trimmed_hops = trimmed_hops[: max(0, len(path) - 1)]
        got = (list(path), trimmed_hops)
        self._path_cache_key = (str(current_id), str(home_id))
        self._path_cache = got
        return got

    def _trim_cached_path(
        self, current_id: int | str, home_id: int | str
    ) -> tuple[list[int | str], list[PathHop]] | None:
        """Reuse cached path when still en route to the same home."""
        if self._path_cache is None or self._path_cache_key is None:
            return None
        if self._path_cache_key[1] != str(home_id):
            return None
        path, hops = self._path_cache
        cur = str(current_id)
        for i, pid in enumerate(path):
            if str(pid) == cur:
                trimmed_path = list(path[i:])
                trimmed_hops = list(hops[i:]) if i < len(hops) else []
                return self._store_path_cache(current_id, home_id, trimmed_path, trimmed_hops)
        return None

    def _snap_path_to_nearest_checkpoint(
        self,
        store: PlaceGraphStore,
        path: list[int | str],
        hops: list[PathHop],
        signals: dict[str, Any],
    ) -> tuple[list[int | str], list[PathHop]] | None:
        """If place-id left the route, resume at the nearest world checkpoint."""
        if len(path) < 2 or not bool(self.config.world_steer):
            return None
        fly_xyz, _ = _fly_pose_from_signals(signals)
        if fly_xyz is None:
            return None
        best_i: int | None = None
        best_d = float("inf")
        for i, pid in enumerate(path):
            target = _world_xyz_of(store.get(pid))
            if target is None:
                continue
            dist = _xz_dist_sq(fly_xyz, target)
            if dist < best_d:
                best_d = dist
                best_i = i
        if best_i is None:
            return None
        # Resume with the nearest node as path[0]; _advance_checkpoints then
        # consumes it only when the *next* checkpoint is reached.
        trimmed_path = list(path[best_i:])
        trimmed_hops = list(hops[best_i:]) if best_i < len(hops) else []
        if len(trimmed_path) < 1:
            return None
        return trimmed_path, trimmed_hops

    def _advance_checkpoints(
        self,
        store: PlaceGraphStore,
        path: list[int | str],
        hops: list[PathHop],
        *,
        current_id: int | str,
        signals: dict[str, Any],
    ) -> tuple[list[int | str], list[PathHop]]:
        """Consume path nodes as checkpoints via place match or world proximity.

        New outdoor place ids used to force a full replan (hops climbed forever).
        Arriving within ``world_arrive_m`` of the next remembered XYZ advances
        even when the matcher minted a fresh place node.
        """
        if len(path) < 2:
            return list(path), list(hops)
        fly_xyz, _ = _fly_pose_from_signals(signals)
        arrive_sq = float(self.config.world_arrive_m) ** 2
        out_path = list(path)
        out_hops = list(hops)
        # Cap pops so a bad radius cannot wipe the whole route in one tick.
        for _ in range(max(1, len(path) - 1)):
            if len(out_path) < 2:
                break
            next_id = out_path[1]
            arrived = str(current_id) == str(next_id)
            if not arrived and fly_xyz is not None and bool(self.config.world_steer):
                target = _world_xyz_of(store.get(next_id))
                if target is not None and _xz_dist_sq(fly_xyz, target) <= arrive_sq:
                    arrived = True
            if not arrived:
                break
            out_path = out_path[1:]
            out_hops = out_hops[1:] if out_hops else []
            current_id = next_id
        if len(out_hops) > max(0, len(out_path) - 1):
            out_hops = out_hops[: max(0, len(out_path) - 1)]
        return out_path, out_hops

    def _resolve_route_path(
        self,
        store: PlaceGraphStore,
        current_id: int | str,
        home_id: int | str,
        signals: dict[str, Any],
    ) -> tuple[list[int | str], list[PathHop]] | None:
        """Path to home: place-id trim, else world checkpoint snap, else replan.

        Never replace an active shorter checkpoint path with a longer replan —
        that was the mid-route "second polyline" (hops 2 → 25) thrash.
        """
        cached: tuple[list[int | str], list[PathHop]] | None = None
        if self._path_cache is not None and self._path_cache_key is not None:
            if self._path_cache_key[1] == str(home_id):
                cached = self._path_cache

        detailed = self._trim_cached_path(current_id, home_id)
        if detailed is None and cached is not None:
            path, hops = cached
            snapped = self._snap_path_to_nearest_checkpoint(store, path, hops, signals)
            if snapped is not None:
                detailed = snapped
        if detailed is None:
            replanned = self._path_with_edges(store, current_id, home_id)
            if replanned is not None and cached is not None and len(replanned[0]) > len(cached[0]):
                path, hops = cached
                snapped = self._snap_path_to_nearest_checkpoint(store, path, hops, signals)
                detailed = snapped if snapped is not None else (list(path), list(hops))
            else:
                detailed = replanned
        if not detailed:
            return None
        path, hops = detailed
        path, hops = self._advance_checkpoints(
            store, path, hops, current_id=current_id, signals=signals
        )
        return self._store_path_cache(current_id, home_id, path, hops)

    def _graph_food_home(
        self,
        store: PlaceGraphStore,
        current_id: int | str,
        *,
        kinds: tuple[str, ...] | None = None,
    ) -> tuple[int | str | None, str, str, float]:
        """Query graph (Neo4j Sense nodes) for food/scent; prefer first experience."""
        if not bool(self.config.graph_food_enabled):
            return None, "", "", 0.0
        finder = getattr(store, "find_food_places", None)
        if not callable(finder):
            return None, "", "", 0.0
        hit_kinds = tuple(kinds) if kinds is not None else SUGAR_SENSE_KINDS
        hits = finder(
            fly_id=self.fly_id,
            min_value=float(self.config.graph_food_min),
            kinds=hit_kinds,
            limit=32,
        )
        if not hits:
            return None, "", "", 0.0
        # prefer_first_food: hits already ordered earliest→latest by the store.
        ordered = list(hits)
        if not bool(self.config.prefer_first_food):
            ordered.sort(key=lambda h: h.value, reverse=True)
        for hit in ordered:
            place = hit.place
            lab = self._place_label(place)
            # Contact sugar / puddle always OK; smell homes must be enclosed food sites.
            if hit.kind == "sugar" or lab == "puddle":
                if lab in _TRAIL_PLUME_LABELS:
                    continue
                if self._opening_on_place(place) >= float(self.config.max_opening_for_home):
                    continue
            elif not self._is_enclosed_food_site(place, lab):
                continue
            if not self._reachable(store, current_id, place.id):
                continue
            self._home_place_id = place.id
            return place.id, "graph_food", str(hit.kind), float(hit.value)
        return None, "", "", 0.0

    def _trail_peak_home(
        self,
        store: PlaceGraphStore,
        current_id: int | str,
        *,
        plume_only: bool = False,
    ) -> tuple[int | str | None, str]:
        """Strongest session-trail smell place (enclosed and/or outdoor plume)."""
        candidates: list[tuple[float, int | str, str]] = []
        if self._peak_smell_id is not None and self._is_trail_reverse_node(
            store, self._peak_smell_id, smell=self._peak_smell
        ):
            lab = self._place_label(store.get(self._peak_smell_id))
            candidates.append((self._peak_smell, self._peak_smell_id, lab))
        # Session trail only — stale graph puddles are sugar_puddle / graph_food.
        for entry in reversed(self._trail):
            min_smell = (
                float(self.config.trail_plume_smell_min)
                if entry.label in _TRAIL_PLUME_LABELS
                else float(self.config.trail_smell_min)
            )
            if entry.smell < min_smell:
                continue
            if not self._is_trail_reverse_node(store, entry.place_id, smell=entry.smell):
                continue
            candidates.append((entry.smell, entry.place_id, entry.label))

        # Enclosed food sites beat outdoor plume peaks; then strongest smell.
        def _rank(row: tuple[float, int | str, str]) -> tuple[int, float]:
            smell, _pid, lab = row
            outdoor = 1 if lab in _TRAIL_PLUME_LABELS else 0
            return (outdoor, -float(smell))

        candidates.sort(key=_rank)
        seen: set[str] = set()
        for smell, place_id, lab in candidates:
            key = str(place_id)
            if key in seen:
                continue
            seen.add(key)
            is_plume = lab in _TRAIL_PLUME_LABELS
            if plume_only and not is_plume:
                continue
            if not plume_only and is_plume:
                continue
            # Outdoor: only reverse toward a *left* plume place (not live smell here).
            if is_plume and str(place_id) == str(current_id):
                continue
            if not self._reachable(store, current_id, place_id):
                continue
            # Outdoor plume is trail-only — never a permanent bookmark.
            if not is_plume:
                self._home_place_id = place_id
            self._peak_smell_id = place_id
            self._peak_smell = max(self._peak_smell, smell)
            reason = "plume_trail" if is_plume else "smell_trail"
            return place_id, reason
        return None, ""

    def _pick_home(
        self, store: PlaceGraphStore, current_id: int | str
    ) -> tuple[int | str | None, str, str, float]:
        """Return (home_id, reason, food_kind, food_value)."""
        # Drop illegal outdoor bookmarks left over from older sessions / bugs.
        if self._home_place_id is not None and not self._is_valid_home_node(
            store, self._home_place_id, require_food_label=True
        ):
            self._home_place_id = None

        # 0) Already committed this hungry bout — keep one active path/home.
        if self._active_home_id is not None:
            aid = self._active_home_id
            if self._reachable(store, current_id, aid) or self._has_checkpoint_route(aid):
                reason = self._active_home_reason or "active_route"
                return (
                    aid,
                    reason,
                    self._active_food_kind,
                    float(self._active_food_value),
                )
            self._clear_active_route()

        # 1) Session outdoor/odor_plume trail peak (left behind) before distant puddles.
        plume_id, plume_reason = self._trail_peak_home(store, current_id, plume_only=True)
        if plume_id is not None:
            return plume_id, plume_reason, "sugar_smell", float(self._peak_smell)

        # 2) Graph contact sugar + enclosed sugar puddles.
        food_id, food_reason, food_kind, food_value = self._graph_food_home(
            store, current_id, kinds=("sugar",)
        )
        if food_id is not None:
            return food_id, food_reason, food_kind, food_value

        puddles = store.find_places(fly_id=self.fly_id, label="puddle", limit=32)
        sugar_homes = [p for p in puddles if self._sugar_on_place(p) > 0.0]
        for candidate in sugar_homes:
            if not self._is_enclosed_food_site(candidate, self._place_label(candidate)):
                continue
            if self._reachable(store, current_id, candidate.id):
                self._home_place_id = candidate.id
                return candidate.id, "sugar_puddle", "sugar", self._sugar_on_place(candidate)

        smell_id, smell_reason, smell_kind, smell_value = self._graph_food_home(
            store, current_id, kinds=("sugar_smell",)
        )
        if smell_id is not None:
            return smell_id, smell_reason, smell_kind, smell_value

        # 3) Enclosed session smell trail (after real food homes).
        trail_id, trail_reason = self._trail_peak_home(store, current_id, plume_only=False)
        if trail_id is not None:
            return trail_id, trail_reason, "sugar_smell", float(self._peak_smell)

        if self._home_place_id is not None:
            if self._reachable(store, current_id, self._home_place_id):
                return self._home_place_id, "bookmarked_home", "", 0.0
            # New outdoor place ids often aren't ADJACENT-linked yet; keep steering
            # along the remembered checkpoint path instead of dropping into scan.
            if bool(self.config.world_steer) and self._has_checkpoint_route(self._home_place_id):
                home_node = store.get(self._home_place_id)
                return (
                    self._home_place_id,
                    "checkpoint_home",
                    "sugar" if self._sugar_on_place(home_node) > 0.0 else "",
                    float(self._sugar_on_place(home_node)),
                )
        if self._chamber_anchor_id is not None:
            if not self._is_valid_home_node(
                store, self._chamber_anchor_id, require_food_label=False
            ):
                self._chamber_anchor_id = None
            elif self._reachable(store, current_id, self._chamber_anchor_id):
                return self._chamber_anchor_id, "chamber_anchor", "", 0.0
        alt = self._find_chamber_in_graph(store, current_id)
        if alt is not None and self._reachable(store, current_id, alt):
            self._chamber_anchor_id = alt
            return alt, "chamber_graph", "", 0.0
        return None, "", "", 0.0

    def _scan_step_degrees(self) -> float:
        return float(np.clip(float(self.config.scan_step_degrees), 10.0, 20.0))

    def _landmark_visible(self, signals: dict[str, Any]) -> bool:
        """True when a landmark cue appears during a scan (stop and re-route next tick)."""
        opening = float(signals.get("opening_strength", 0.0) or 0.0)
        novelty = float(signals.get("visual_novelty", 0.0) or 0.0)
        wall = float(signals.get("wall_hazard", 0.0) or 0.0)
        if novelty >= float(self.config.scan_landmark_novelty):
            return True
        if opening >= float(self.config.scan_landmark_opening):
            prev = self._prev_scan_opening
            if prev is None or opening - float(prev) >= 0.1:
                return True
        # Hitting a wall while scanning can be an entrance/corner landmark.
        if wall >= float(self.config.wall_avoid) and opening >= float(
            self.config.entrance_opening_min
        ):
            return True
        return False

    def _edge_missing(self, hop: PathHop | None, signals: dict[str, Any]) -> bool:
        """No usable remembered edge turn and no live opening to steer by."""
        if hop is None:
            return True
        props = dict(hop.props or {})
        has_turn = (
            props.get("turn") is not None and abs(float(props.get("turn") or 0.0)) >= EDGE_TURN_MIN
        )
        opening = float(signals.get("opening_strength", 0.0) or 0.0)
        edge_open = float(props.get("opening_strength") or 0.0)
        has_open = max(opening, edge_open) >= float(self.config.entrance_opening_min)
        return not has_turn and not has_open

    def _landmark_scan(
        self,
        signals: dict[str, Any],
        *,
        reason: str,
        home_id: int | str | None = None,
        food_kind: str = "",
        food_value: float = 0.0,
    ) -> RouteHomeState:
        """Spin the body in 10–20° steps through a full circle when memory fails.

        Head leads by about one step in Unity while the thorax yaws continuously
        (not a ±span zigzag), so both body and eyes sweep landmarks.
        """
        if not bool(self.config.landmark_scan):
            self._reset_scan()
            return RouteHomeState(
                mode="off",
                reason=reason,
                home_id=home_id,
                food_kind=food_kind,
                food_value=food_value,
            )
        opening = float(signals.get("opening_strength", 0.0) or 0.0)
        if self._landmark_visible(signals):
            self._prev_scan_opening = opening
            # Hold still this tick so place-split / re-route can catch the landmark.
            self._state = RouteHomeState(
                mode="handoff",
                turn=None,
                forward=0.0,
                home_id=home_id,
                reason="scan_landmark",
                food_kind=food_kind,
                food_value=food_value,
                scan_yaw_degrees=float(self._scan_yaw_degrees),
            )
            return self._state

        step = self._scan_step_degrees()
        span = max(step, float(self.config.scan_span_degrees))
        # Continuous one-way spin; wrap after a full circle and keep scanning.
        self._scan_side = 1.0
        self._scan_yaw_degrees = (float(self._scan_yaw_degrees) + step) % span
        turn = float(np.clip(float(self.config.scan_turn), 0.0, ROUTE_TURN_MAX))
        self._prev_scan_opening = opening
        self._state = RouteHomeState(
            mode="scan",
            turn=turn,
            forward=0.0,
            home_id=home_id,
            reason=reason,
            food_kind=food_kind,
            food_value=food_value,
            scan_yaw_degrees=float(self._scan_yaw_degrees),
        )
        return self._state

    def _world_turn_to_place(
        self,
        store: PlaceGraphStore,
        place_id: int | str,
        signals: dict[str, Any],
    ) -> float | None:
        """Turn toward a place's remembered Unity XYZ, or None if unmapped."""
        if not bool(self.config.world_steer):
            return None
        fly_xyz, fly_yaw = _fly_pose_from_signals(signals)
        if fly_xyz is None or fly_yaw is None:
            return None
        target = _world_xyz_of(store.get(place_id))
        if target is None:
            return None
        dx = float(target[0] - fly_xyz[0])
        dz = float(target[2] - fly_xyz[2])
        if dx * dx + dz * dz <= float(self.config.world_arrive_m) ** 2:
            return 0.0
        return world_bearing_turn(
            fly_xyz,
            fly_yaw,
            target,
            gain=float(self.config.world_steer_gain),
        )

    def _steer_for_hop(
        self,
        store: PlaceGraphStore,
        current_id: int | str,
        next_id: int | str,
        signals: dict[str, Any],
        *,
        smell_delta: float = 0.0,
        hop: PathHop | None = None,
    ) -> float:
        # Prefer physical Unity bearing to the next remembered place when mapped.
        world_turn = self._world_turn_to_place(store, next_id, signals)
        if world_turn is not None:
            turn = float(world_turn)
            # Rising sugar still blends so chemotaxis can finish the approach.
            if smell_delta >= float(self.config.rise_eps):
                sugar_bearing = float(signals.get("sugar_bearing", 0.0) or 0.0)
                blend = float(self.config.smell_blend)
                turn = (1.0 - blend) * turn + blend * sugar_bearing
            return float(np.clip(turn, -ROUTE_TURN_MAX, ROUTE_TURN_MAX))

        # Fallback: Neo4j/path hop props (oriented along the return path).
        turn = 0.0
        edge_props: dict[str, Any] = {}
        reversed_hop = False
        if hop is not None:
            edge_props = dict(hop.props or {})
            reversed_hop = bool(hop.reversed)
            if edge_props.get("turn") is not None:
                turn = float(edge_props["turn"])
                if reversed_hop:
                    turn = -turn
        else:
            # Prefer neighbors(current) over dumping the full edge table.
            edge = None
            neigh = getattr(store, "neighbors", None)
            if callable(neigh):
                for e in neigh(current_id, undirected=True):
                    other = e.dst_id if str(e.src_id) == str(current_id) else e.src_id
                    if str(other) == str(next_id):
                        edge = e
                        break
            if edge is None:
                edge = edge_toward(store.edges(), current_id, next_id)
            if edge is not None and edge.props.get("turn") is not None:
                turn = float(edge.props["turn"])
                if str(edge.src_id) == str(next_id) and str(edge.dst_id) == str(current_id):
                    turn = -turn
                    reversed_hop = True
                edge_props = dict(edge.props or {})

        opening = float(signals.get("opening_strength", 0.0) or 0.0)
        bearing = float(signals.get("opening_bearing", 0.0) or 0.0)
        wall = float(signals.get("wall_hazard", 0.0) or 0.0)
        label = str(signals.get("place_label") or "")

        # Outside + wall ahead: prefer the opening (chamber entrance) over wall push.
        if (
            label == "outside"
            and wall >= float(self.config.wall_avoid)
            and opening >= float(self.config.entrance_opening_min)
        ):
            turn = OPENING_TURN_GAIN * bearing
        elif abs(turn) < EDGE_TURN_MIN:
            if opening >= 0.1:
                turn = OPENING_TURN_GAIN * bearing
            else:
                bearing = float(edge_props.get("opening_bearing") or 0.0)
                if reversed_hop:
                    bearing = -bearing
                turn = OPENING_TURN_GAIN * bearing

        # Sugar smell getting stronger → home in with sugar antenna bearing only.
        if smell_delta >= float(self.config.rise_eps):
            sugar_bearing = float(signals.get("sugar_bearing", 0.0) or 0.0)
            blend = float(self.config.smell_blend)
            turn = (1.0 - blend) * turn + blend * sugar_bearing

        return float(np.clip(turn, -ROUTE_TURN_MAX, ROUTE_TURN_MAX))

    def observe(
        self,
        *,
        store: PlaceGraphStore | None,
        place: PlaceNode | None,
        signals: dict[str, Any],
    ) -> RouteHomeState:
        if not self.config.enabled or store is None:
            self._state = RouteHomeState(mode="off", reason="disabled")
            return self._state

        label = str(
            signals.get("place_label")
            or ((place.props or {}).get("label") if place is not None else "")
            or ""
        )
        self._update_anchors(place, label, signals, store)

        if place is None or not is_hungry(signals):
            self._last_route_smell = None
            self._clear_path_cache()
            self._clear_active_route()
            self._reset_scan()
            self._stop_turn_commit()
            self._state = RouteHomeState(mode="off", reason="not_hungry" if place else "no_place")
            return self._state

        current_id = place.id
        home_id, home_reason, food_kind, food_value = self._pick_home(store, current_id)
        if home_id is None:
            self._clear_path_cache()
            self._stop_turn_commit()
            return self._landmark_scan(
                signals,
                reason="no_home",
                food_kind=food_kind,
                food_value=food_value,
            )
        self._commit_active_home(home_id, home_reason, food_kind, food_value)

        # Follow remembered place nodes as checkpoints back to the scent home.
        detailed = self._resolve_route_path(store, current_id, home_id, signals)
        if not detailed:
            self._stop_turn_commit()
            return self._landmark_scan(
                signals,
                reason="unreachable",
                home_id=home_id,
                food_kind=food_kind,
                food_value=food_value,
            )
        path, hops = detailed
        edge_dicts = [h.to_dict() for h in hops]

        sugar_smell = float(signals.get("sugar_smell", signals.get("sugar", 0.0)) or 0.0)
        live_smell = _live_smell(signals)
        prev = self._last_route_smell
        smell_delta = 0.0 if prev is None else live_smell - float(prev)
        self._last_route_smell = live_smell

        hops_left = max(0, len(path) - 1)
        contact_sugar = _contact_sugar(signals)
        # Standing on food (even a different puddle than the routed home): stop
        # so sugar_search can hold a feed bout instead of forcing p0.0 past it.
        if contact_sugar >= float(self.config.on_sugar_taste_min):
            if label == "puddle" or self._sugar_on_place(place) > 0.0:
                self._home_place_id = current_id
                home_id = current_id
                food_kind = "sugar"
                food_value = max(float(food_value), contact_sugar)
                path = [current_id]
                edge_dicts = []
                hops_left = 0
                self._clear_path_cache()
                self._clear_active_route()
            self._reset_scan()
            self._stop_turn_commit()
            self._state = RouteHomeState(
                mode="handoff",
                turn=None,
                home_id=home_id,
                next_id=None,
                hops_left=hops_left,
                path=list(path),
                path_edges=edge_dicts,
                reason="on_sugar",
                smell_delta=smell_delta,
                food_kind=food_kind,
                food_value=food_value,
            )
            return self._state

        at_home = hops_left == 0
        near_home = hops_left <= 1 and sugar_smell >= float(self.config.handoff_sugar_min)
        if at_home or near_home:
            self._reset_scan()
            self._stop_turn_commit()
            self._state = RouteHomeState(
                mode="handoff",
                turn=None,
                home_id=home_id,
                next_id=None,
                hops_left=hops_left,
                path=list(path),
                path_edges=edge_dicts,
                reason="at_home" if at_home else "near_home_sugar",
                smell_delta=smell_delta,
                food_kind=food_kind,
                food_value=food_value,
            )
            return self._state

        # Route when outside or still more than one hop from home.
        if label != "outside" and hops_left <= 1:
            self._reset_scan()
            self._stop_turn_commit()
            self._state = RouteHomeState(
                mode="handoff",
                turn=None,
                home_id=home_id,
                hops_left=hops_left,
                path=list(path),
                path_edges=edge_dicts,
                reason="inside_near_home",
                smell_delta=smell_delta,
                food_kind=food_kind,
                food_value=food_value,
            )
            return self._state

        next_id = path[1]
        hop = hops[0] if hops else None
        if self._edge_missing(hop, signals) and smell_delta < float(self.config.rise_eps):
            self._stop_turn_commit()
            return self._landmark_scan(
                signals,
                reason="missing_edge",
                home_id=home_id,
                food_kind=food_kind,
                food_value=food_value,
            )
        self._reset_scan()
        turn = self._steer_for_hop(
            store,
            current_id,
            next_id,
            signals,
            smell_delta=smell_delta,
            hop=hop,
        )
        turn = self._commit_turn(turn, hops_left=hops_left)
        forward = self._route_forward(turn, smell_delta=smell_delta)
        self._state = RouteHomeState(
            mode="route",
            turn=turn,
            forward=forward,
            home_id=home_id,
            next_id=next_id,
            hops_left=hops_left,
            path=list(path),
            path_edges=edge_dicts,
            reason=home_reason or "route",
            smell_delta=smell_delta,
            food_kind=food_kind,
            food_value=food_value,
            commit_degrees_left=float(self._commit_degrees_left),
        )
        return self._state
