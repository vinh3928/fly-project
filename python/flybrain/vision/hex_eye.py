"""Pointy-top hexagonal eye aperture with 6 triangular facets.

Geometry matches Unity ``FlyEyeHexGeometry``: a regular hexagon inscribed in the
eye frame, subdivided by diagonals through the center (center → each vertex).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from flybrain.phase4 import HEX_FACET_COUNT


def hex_center_and_radius(height: int, width: int) -> tuple[float, float, float]:
    """Pixel-center and inradius/apothem-style outer radius for a pointy-top hex."""
    h = max(1, int(height))
    w = max(1, int(width))
    cx = 0.5 * float(w - 1)
    cy = 0.5 * float(h - 1)
    # Inscribe in the square frame with a small inset so corners stay outside.
    radius = 0.5 * float(min(h, w) - 1) * 0.98
    return cx, cy, radius


def hex_vertices_px(height: int, width: int) -> np.ndarray:
    """Return (6, 2) vertex pixel coords; vertex 0 at top, CCW in image space.

    Math angles use y-up; image ``y`` increases downward, so sin is flipped.
    """
    cx, cy, radius = hex_center_and_radius(height, width)
    verts = np.zeros((HEX_FACET_COUNT, 2), dtype=np.float64)
    for i in range(HEX_FACET_COUNT):
        # Pointy-top: first vertex at top (−π/2 from +x in y-up).
        angle = -0.5 * np.pi + i * (np.pi / 3.0)
        verts[i, 0] = cx + radius * np.cos(angle)
        verts[i, 1] = cy - radius * np.sin(angle)
    return verts


def _point_in_triangle(
    px: np.ndarray,
    py: np.ndarray,
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Barycentric point-in-triangle for broadcast pixel grids."""
    v0x, v0y = cx - ax, cy - ay
    v1x, v1y = bx - ax, by - ay
    v2x = px - ax
    v2y = py - ay
    dot00 = v0x * v0x + v0y * v0y
    dot01 = v0x * v1x + v0y * v1y
    dot02 = v0x * v2x + v0y * v2y
    dot11 = v1x * v1x + v1y * v1y
    dot12 = v1x * v2x + v1y * v2y
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-12:
        return np.zeros_like(px, dtype=bool)
    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    return (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1.0 + 1e-6)


@lru_cache(maxsize=32)
def hex_aperture_mask(height: int, width: int) -> np.ndarray:
    """Bool (H, W) True inside the pointy-top hexagon."""
    masks = hex_triangle_masks(height, width)
    out = np.zeros((height, width), dtype=bool)
    for m in masks:
        out |= m
    return out


@lru_cache(maxsize=32)
def hex_triangle_masks(height: int, width: int) -> tuple[np.ndarray, ...]:
    """Six bool (H, W) masks: facet ``i`` is triangle (center, vertex_i, vertex_{i+1})."""
    h = max(1, int(height))
    w = max(1, int(width))
    cx, cy, _ = hex_center_and_radius(h, w)
    verts = hex_vertices_px(h, w)
    ys, xs = np.mgrid[0:h, 0:w]
    px = xs.astype(np.float64)
    py = ys.astype(np.float64)
    masks: list[np.ndarray] = []
    for i in range(HEX_FACET_COUNT):
        j = (i + 1) % HEX_FACET_COUNT
        ax, ay = float(verts[i, 0]), float(verts[i, 1])
        bx, by = float(verts[j, 0]), float(verts[j, 1])
        masks.append(_point_in_triangle(px, py, cx, cy, ax, ay, bx, by))
    return tuple(masks)


def facet_centroid_bearing(
    facet: int, *, eye: str = "left", eye_bearing_offset: float = 0.2
) -> float:
    """Azimuth in [-1, 1] from triangular facet centroid (pointy-top layout)."""
    i = int(facet) % HEX_FACET_COUNT
    # Centroid angle of facet i (y-up): midway between vertex i and i+1.
    angle = -0.5 * np.pi + (i + 0.5) * (np.pi / 3.0)
    # Horizontal component: cos(angle) in y-up → +x right; map to bearing.
    x = float(np.cos(angle))
    offset = float(eye_bearing_offset)
    if eye == "left":
        x = x - offset
    elif eye == "right":
        x = x + offset
    return float(np.clip(x, -1.0, 1.0))


def apply_hex_aperture(frame: np.ndarray) -> np.ndarray:
    """Zero pixels outside the hexagonal aperture. Accepts (H,W), (C,H,W), or (H,W,C)."""
    x = np.asarray(frame, dtype=np.float32).copy()
    if x.ndim == 2:
        h, w = x.shape
        mask = hex_aperture_mask(h, w)
        x[~mask] = 0.0
        return x
    if x.ndim == 3 and x.shape[0] in (1, 3):
        _c, h, w = x.shape
        mask = hex_aperture_mask(h, w)
        x[:, ~mask] = 0.0
        return x
    if x.ndim == 3 and x.shape[-1] in (1, 3):
        h, w, _c = x.shape
        mask = hex_aperture_mask(h, w)
        x[~mask, :] = 0.0
        return x
    return x
