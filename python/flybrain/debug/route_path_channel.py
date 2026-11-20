"""ML-Agents SideChannel: route path + place-graph map for Unity debug viz.

Python → Unity messages (first float = version):

  version=1 path polyline:
    [1, mode, current_index, n_points, x0,y0,z0, ...]
    mode: 0=off, 1=route, 2=scan

  version=2 place-graph snapshot (latent memory laid on Unity XYZ):
    [2, n_nodes, n_edges,
     (label, flags, x, y, z, smell) * n_nodes,
     (src_index, dst_index) * n_edges]
    label: 0=unk 1=place 2=puddle 3=outside 4=odor_plume 5=wall
    flags bit0=current bit1=home bit2=on_route

Unity → Python (fly world pose for anchoring places):
  [x, y, z] or [x, y, z, yawDegrees]
  yawDegrees is Unity transform.eulerAngles.y (forward = +Z).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID

import numpy as np

# Fixed Guid — must match RoutePathDebugChannel.cs
ROUTE_PATH_CHANNEL_ID = UUID("c8f0a2b4-5d6e-4a1b-9c3d-7e8f90123456")

MODE_OFF = 0
MODE_ROUTE = 1
MODE_SCAN = 2
PROTOCOL_VERSION = 1.0
GRAPH_PROTOCOL_VERSION = 2.0
# Keep snapshots small: route + home/current only (no full-graph Neo4j dump).
MAX_GRAPH_NODES = 24
MAX_GRAPH_EDGES = 24
# Skip identical graph payloads; refresh at least this often while routing.
GRAPH_SEND_EVERY_STEPS = 8

LABEL_UNKNOWN = 0
LABEL_PLACE = 1
LABEL_PUDDLE = 2
LABEL_OUTSIDE = 3
LABEL_ODOR_PLUME = 4
LABEL_WALL = 5
_LABEL_CODES = {
    "place": LABEL_PLACE,
    "puddle": LABEL_PUDDLE,
    "outside": LABEL_OUTSIDE,
    "odor_plume": LABEL_ODOR_PLUME,
    "wall": LABEL_WALL,
}
FLAG_CURRENT = 1
FLAG_HOME = 2
FLAG_ROUTE = 4


def pack_path_message(
    points: Sequence[Sequence[float]],
    *,
    mode: int = MODE_ROUTE,
    current_index: int = -1,
) -> list[float]:
    """Flatten path points into the SideChannel float list."""
    pts = [tuple(float(x) for x in p[:3]) for p in points if len(p) >= 3]
    n = len(pts)
    cur = int(current_index)
    if n == 0:
        cur = -1
    elif cur >= n:
        cur = n - 1
    flat: list[float] = [
        float(PROTOCOL_VERSION),
        float(int(mode)),
        float(cur),
        float(n),
    ]
    for x, y, z in pts:
        flat.extend([x, y, z])
    return flat


def parse_path_message(values: Sequence[float]) -> dict[str, Any]:
    """Parse a Python→Unity path payload (also used by tests / Unity mirror)."""
    data = [float(v) for v in values]
    if len(data) < 4:
        return {"version": 0, "mode": MODE_OFF, "current_index": -1, "points": []}
    version = data[0]
    if version >= 2.0:
        return parse_graph_message(data)
    mode = int(data[1])
    current_index = int(data[2])
    n = max(0, int(data[3]))
    points: list[tuple[float, float, float]] = []
    offset = 4
    for _ in range(n):
        if offset + 2 >= len(data):
            break
        points.append((data[offset], data[offset + 1], data[offset + 2]))
        offset += 3
    return {
        "version": version,
        "mode": mode,
        "current_index": current_index,
        "points": points,
    }


def label_code(label: str | None) -> int:
    return int(_LABEL_CODES.get(str(label or "").lower(), LABEL_UNKNOWN))


def pack_graph_message(
    nodes: Sequence[dict[str, Any]],
    edges: Sequence[tuple[int, int]],
) -> list[float]:
    """Flatten place-graph nodes/edges into the version=2 SideChannel list."""
    capped = list(nodes)[:MAX_GRAPH_NODES]
    flat: list[float] = [
        float(GRAPH_PROTOCOL_VERSION),
        float(len(capped)),
        float(min(len(edges), MAX_GRAPH_EDGES)),
    ]
    for node in capped:
        flat.extend(
            [
                float(int(node.get("label", LABEL_UNKNOWN))),
                float(int(node.get("flags", 0))),
                float(node.get("x", 0.0)),
                float(node.get("y", 0.0)),
                float(node.get("z", 0.0)),
                float(np.clip(float(node.get("smell", 0.0) or 0.0), 0.0, 1.0)),
            ]
        )
    for i, (src, dst) in enumerate(edges):
        if i >= MAX_GRAPH_EDGES:
            break
        flat.extend([float(int(src)), float(int(dst))])
    return flat


def parse_graph_message(values: Sequence[float]) -> dict[str, Any]:
    """Parse a version=2 place-graph payload."""
    data = [float(v) for v in values]
    if len(data) < 3:
        return {"version": 0, "nodes": [], "edges": []}
    n_nodes = max(0, int(data[1]))
    n_edges = max(0, int(data[2]))
    nodes: list[dict[str, Any]] = []
    offset = 3
    for _ in range(n_nodes):
        if offset + 5 >= len(data):
            break
        nodes.append(
            {
                "label": int(data[offset]),
                "flags": int(data[offset + 1]),
                "x": float(data[offset + 2]),
                "y": float(data[offset + 3]),
                "z": float(data[offset + 4]),
                "smell": float(data[offset + 5]),
            }
        )
        offset += 6
    edges: list[tuple[int, int]] = []
    for _ in range(n_edges):
        if offset + 1 >= len(data):
            break
        edges.append((int(data[offset]), int(data[offset + 1])))
        offset += 2
    return {"version": float(data[0]), "nodes": nodes, "edges": edges}


def _pose_from_props(props: dict[str, Any]) -> np.ndarray | None:
    raw = props.get("world_xyz", props.get("position"))
    if raw is None:
        return None
    try:
        pose = np.asarray(raw, dtype=np.float32).reshape(-1)[:3]
    except Exception:
        return None
    if pose.size < 3 or not np.all(np.isfinite(pose)):
        return None
    return pose


def _smell_from_props(props: dict[str, Any]) -> float:
    senses = props.get("senses") if isinstance(props.get("senses"), dict) else {}
    smell = 0.0
    for key in ("sugar_smell", "sugar"):
        row = senses.get(key) if isinstance(senses, dict) else None
        if isinstance(row, dict):
            smell = max(smell, float(row.get("value") or 0.0))
    return float(smell)


def build_graph_snapshot(
    store: Any,
    *,
    place_poses: dict[str, np.ndarray],
    current_place_id: Any | None = None,
    home_id: Any | None = None,
    route_path: Sequence[Any] | None = None,
    fly_id: str | None = None,
    max_nodes: int = MAX_GRAPH_NODES,
) -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
    """Collect route/home/current places with XYZ — no Neo4j full-graph scan.

    ``fly_id`` is accepted for API compatibility; snapshots are seeded from
    recorded poses + the active route only (keeps the SideChannel cheap).
    """
    del fly_id  # unused — avoid accidental store-wide queries
    route_list = [str(p) for p in (route_path or []) if p is not None]
    route_set = set(route_list)
    current_s = str(current_place_id) if current_place_id is not None else ""
    home_s = str(home_id) if home_id is not None else ""

    # Only ids we already know about: poses + route/current/home.
    id_set: set[str] = set(place_poses.keys())
    id_set.update(route_set)
    if current_s:
        id_set.add(current_s)
    if home_s:
        id_set.add(home_s)

    getter = getattr(store, "get", None) if store is not None else None
    scored: list[tuple[float, str]] = []
    for pid in id_set:
        if not pid:
            continue
        node = getter(pid) if callable(getter) else None
        props = (getattr(node, "props", None) or {}) if node is not None else {}
        pose = place_poses.get(pid)
        if pose is None:
            pose = _pose_from_props(props)
        if pose is None or np.asarray(pose).size < 3:
            continue
        smell = _smell_from_props(props)
        pri = smell
        if pid == current_s:
            pri += 100.0
        if pid == home_s:
            pri += 50.0
        if pid in route_set:
            pri += 10.0
        scored.append((pri, pid))

    scored.sort(key=lambda row: -row[0])
    nodes_out: list[dict[str, Any]] = []
    index_of: dict[str, int] = {}
    for _pri, pid in scored[: max(1, int(max_nodes))]:
        node = getter(pid) if callable(getter) else None
        props = (getattr(node, "props", None) or {}) if node is not None else {}
        pose = place_poses.get(pid)
        if pose is None:
            pose = _pose_from_props(props)
        if pose is None:
            continue
        pose = np.asarray(pose, dtype=np.float32).reshape(-1)
        flags = 0
        if pid == current_s:
            flags |= FLAG_CURRENT
        if pid == home_s:
            flags |= FLAG_HOME
        if pid in route_set:
            flags |= FLAG_ROUTE
        index_of[pid] = len(nodes_out)
        nodes_out.append(
            {
                "id": pid,
                "label": label_code(str(props.get("label") or "")),
                "flags": flags,
                "x": float(pose[0]),
                "y": float(pose[1]),
                "z": float(pose[2]),
                "smell": _smell_from_props(props),
            }
        )

    # Route polyline edges only (skip store.edges() — that dump was a major hitch).
    edges_out: list[tuple[int, int]] = []
    prev: str | None = None
    for pid in route_list:
        if pid in index_of and prev is not None and prev in index_of:
            edges_out.append((index_of[prev], index_of[pid]))
        if pid in index_of:
            prev = pid
    return nodes_out, edges_out[:MAX_GRAPH_EDGES]


def parse_pose_message(values: Sequence[float]) -> np.ndarray | None:
    """Parse Unity→Python fly pose ``[x,y,z]`` (yaw ignored; see ``parse_pose_yaw``)."""
    if values is None or len(values) < 3:
        return None
    return np.asarray([float(values[0]), float(values[1]), float(values[2])], dtype=np.float32)


def parse_pose_yaw(values: Sequence[float]) -> float | None:
    """Optional yaw degrees from Unity pose payload index 3."""
    if values is None or len(values) < 4:
        return None
    return float(values[3])


def world_bearing_turn(
    fly_xyz: Sequence[float],
    fly_yaw_deg: float,
    target_xyz: Sequence[float],
    *,
    gain: float = 1.0,
) -> float:
    """Motor turn in ``[-1,1]`` toward ``target_xyz`` in Unity XZ (positive = right)."""
    fx, fz = float(fly_xyz[0]), float(fly_xyz[2])
    tx, tz = float(target_xyz[0]), float(target_xyz[2])
    dx, dz = tx - fx, tz - fz
    dist_sq = dx * dx + dz * dz
    if dist_sq < 1e-8:
        return 0.0
    inv = 1.0 / float(np.sqrt(dist_sq))
    dx *= inv
    dz *= inv
    yaw = np.deg2rad(float(fly_yaw_deg))
    # Unity forward from euler Y: (sin yaw, cos yaw) in XZ.
    fwd_x = float(np.sin(yaw))
    fwd_z = float(np.cos(yaw))
    # Signed angle around +Y; positive = target to the right of forward.
    cross = fwd_z * dx - fwd_x * dz
    dot = fwd_x * dx + fwd_z * dz
    err = float(np.arctan2(cross, dot))
    return float(np.clip(float(gain) * err / (0.5 * np.pi), -1.0, 1.0))


def mode_from_route(route: dict[str, Any] | None) -> int:
    if not isinstance(route, dict):
        return MODE_OFF
    m = str(route.get("mode") or "")
    if m in ("route", "handoff"):
        return MODE_ROUTE
    if m == "scan":
        return MODE_SCAN
    return MODE_OFF


class RoutePathDebugChannel:
    """SideChannel wrapper; safe no-op if mlagents_envs is unavailable."""

    def __init__(self) -> None:
        self.last_pose: np.ndarray | None = None
        self.last_yaw_deg: float | None = None
        self._place_poses: dict[str, np.ndarray] = {}
        self._impl: Any = None
        self._graph_step: int = 0
        self._last_graph_sig: tuple[Any, ...] | None = None
        try:
            from mlagents_envs.side_channel.side_channel import (
                IncomingMessage,
                OutgoingMessage,
                SideChannel,
            )

            channel_id = ROUTE_PATH_CHANNEL_ID
            outer = self

            class _Impl(SideChannel):
                def __init__(self) -> None:
                    super().__init__(channel_id)

                def on_message_received(self, msg: IncomingMessage) -> None:
                    try:
                        values = msg.read_float32_list()
                    except Exception:
                        return
                    pose = parse_pose_message(values)
                    if pose is not None:
                        outer.last_pose = pose
                        outer.last_yaw_deg = parse_pose_yaw(values)

            self._impl = _Impl()
            self._OutgoingMessage = OutgoingMessage
        except Exception:
            self._impl = None
            self._OutgoingMessage = None

    @property
    def side_channel(self) -> Any:
        """Underlying ML-Agents SideChannel instance (or None)."""
        return self._impl

    def record_place_pose(self, place_id: Any, pose: np.ndarray | None = None) -> None:
        """Remember world XYZ for a place id (from Unity pose or explicit)."""
        p = pose if pose is not None else self.last_pose
        if place_id is None or p is None or np.asarray(p).size < 3:
            return
        self._place_poses[str(place_id)] = np.asarray(p, dtype=np.float32)[:3].copy()

    def pose_for(self, place_id: Any) -> np.ndarray | None:
        return self._place_poses.get(str(place_id))

    def build_path_points(
        self,
        route: dict[str, Any] | None,
        *,
        current_place_id: Any | None = None,
        include_fly: bool = True,
    ) -> tuple[list[tuple[float, float, float]], int]:
        """Resolve route path ids to XYZ; optionally prepend live fly pose."""
        points: list[tuple[float, float, float]] = []
        current_index = -1
        if include_fly and self.last_pose is not None:
            fly = self.last_pose
            points.append((float(fly[0]), float(fly[1]), float(fly[2])))
            current_index = 0
        path = list((route or {}).get("path") or [])
        for pid in path:
            pose = self._place_poses.get(str(pid))
            if pose is None:
                continue
            pt = (float(pose[0]), float(pose[1]), float(pose[2]))
            # Skip duplicate of fly tip when current place already matches.
            if points and np.allclose(points[-1], pt, atol=1e-4):
                if current_place_id is not None and str(pid) == str(current_place_id):
                    current_index = len(points) - 1
                continue
            points.append(pt)
            if current_place_id is not None and str(pid) == str(current_place_id):
                current_index = len(points) - 1
        return points, current_index

    def _queue(self, packed: list[float]) -> list[float]:
        if self._impl is not None and self._OutgoingMessage is not None:
            msg = self._OutgoingMessage()
            msg.write_float32_list(packed)
            self._impl.queue_message_to_send(msg)
        return packed

    def send_route(
        self,
        route: dict[str, Any] | None,
        *,
        current_place_id: Any | None = None,
    ) -> list[float]:
        """Queue a path message for the next ``env.step``; return the packed floats."""
        mode = mode_from_route(route)
        if mode == MODE_OFF:
            packed = pack_path_message([], mode=MODE_OFF, current_index=-1)
        else:
            points, cur = self.build_path_points(route, current_place_id=current_place_id)
            packed = pack_path_message(points, mode=mode, current_index=cur)
        return self._queue(packed)

    def send_place_graph(
        self,
        store: Any,
        *,
        current_place_id: Any | None = None,
        home_id: Any | None = None,
        route_path: Sequence[Any] | None = None,
        fly_id: str | None = None,
        force: bool = False,
        step: int | None = None,
    ) -> list[float] | None:
        """Queue a version=2 place-graph map; skip unchanged / off-throttle ticks."""
        path_t = tuple(str(p) for p in (route_path or ()))
        sig = (
            str(current_place_id) if current_place_id is not None else "",
            str(home_id) if home_id is not None else "",
            path_t,
        )
        tick = int(step) if step is not None else self._graph_step + 1
        self._graph_step = tick
        due = force or self._last_graph_sig != sig or (tick % GRAPH_SEND_EVERY_STEPS == 0)
        if not due:
            return None
        self._last_graph_sig = sig
        nodes, edges = build_graph_snapshot(
            store,
            place_poses=self._place_poses,
            current_place_id=current_place_id,
            home_id=home_id,
            route_path=route_path,
            fly_id=fly_id,
        )
        return self._queue(pack_graph_message(nodes, edges))

    def clear(self) -> None:
        self.send_route(None)
        self._queue(pack_graph_message([], []))
        self._place_poses.clear()
        self.last_pose = None
        self.last_yaw_deg = None
        self._last_graph_sig = None
        self._graph_step = 0
