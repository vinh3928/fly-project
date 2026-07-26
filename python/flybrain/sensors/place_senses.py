"""Vision-keyed place details: smell / taste / hazard correlated onto Place props.

The Place node is identified by vision (``view_vec``). Sensor and smell readings
at that viewpoint are stored as place *details* (not as separate place keys),
e.g. a puddle Place with ``senses.bitter = 0.49``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from flybrain.phase4 import SMELL_VEC_DIM
from flybrain.sensors.modal_gate import pad_modal_vec

# Channels persisted on Place.props["senses"].
SENSE_CHANNELS: tuple[str, ...] = (
    "wall_hazard",
    "odor",
    "odor_left",
    "odor_right",
    "bitter",
    "sugar",
)

_CHEMICAL_CHANNELS: tuple[str, ...] = (
    "bitter",
    "sugar",
    "odor",
    "odor_left",
    "odor_right",
)


def _clamp01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def senses_vector(raw: dict[str, float]) -> np.ndarray:
    """Fixed-order vector of sense channels for ModalChangeGate diffs."""
    return np.asarray(
        [_clamp01(float(raw.get(ch, 0.0) or 0.0)) for ch in SENSE_CHANNELS],
        dtype=np.float32,
    )


def infer_place_label(
    raw: dict[str, float],
    *,
    present_min: float = 0.05,
    wall_min: float = 0.55,
) -> str:
    """Label a vision Place from co-located sensor evidence.

    Vision keys the node; chemistry names it when a puddle/plume is present at
    that view. Example: bitter 0.49 at a chromatic patch → ``\"puddle\"``.
    """
    bitter = float(raw.get("bitter", 0.0) or 0.0)
    sugar = float(raw.get("sugar", 0.0) or 0.0)
    odor = max(
        float(raw.get("odor", 0.0) or 0.0),
        float(raw.get("odor_left", 0.0) or 0.0),
        float(raw.get("odor_right", 0.0) or 0.0),
    )
    wall = float(raw.get("wall_hazard", 0.0) or 0.0)

    if bitter >= float(present_min) or sugar >= float(present_min):
        return "puddle"
    if odor >= float(present_min):
        return "odor_plume"
    if wall >= float(wall_min):
        return "wall"
    return "place"


def merge_place_senses(
    props: dict[str, Any] | None,
    raw: dict[str, float],
    *,
    step: int | None = None,
    smell_vec: np.ndarray | None = None,
    present_min: float = 0.05,
    label: str | None = None,
) -> dict[str, Any]:
    """Merge current sensor/smell readings into place props (vision-correlated)."""
    out = dict(props or {})
    inferred = str(label or infer_place_label(raw, present_min=present_min))
    # Keep a more specific prior label unless we now have chemical evidence.
    prior = str(out.get("label") or "")
    if inferred in ("puddle", "odor_plume") or not prior or prior == "place":
        out["label"] = inferred
    elif inferred == "wall" and prior == "place":
        out["label"] = "wall"

    prior_senses = out.get("senses") if isinstance(out.get("senses"), dict) else {}
    senses: dict[str, Any] = dict(prior_senses)
    for channel in SENSE_CHANNELS:
        value = _clamp01(float(raw.get(channel, 0.0) or 0.0))
        # Always refresh present channels; keep last reading for quiet channels.
        if value >= float(present_min) or channel in senses:
            row = dict(senses.get(channel) or {})
            if value >= float(present_min) or channel not in senses:
                row["value"] = value
            if step is not None and value >= float(present_min):
                row["last_seen_step"] = int(step)
            senses[channel] = row
    out["senses"] = senses

    if smell_vec is not None:
        out["smell_vec"] = pad_modal_vec(smell_vec, SMELL_VEC_DIM).tolist()
    return out


def sense_rows_for_neo4j(
    place_id: str,
    senses: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Flatten props.senses into Neo4j Sense node rows."""
    if not isinstance(senses, dict):
        return []
    rows: list[dict[str, Any]] = []
    for kind, raw in senses.items():
        if not isinstance(raw, dict):
            continue
        value = _clamp01(float(raw.get("value") or 0.0))
        rows.append(
            {
                "sense_id": f"{place_id}:{kind}",
                "kind": str(kind),
                "value": value,
                "last_seen_step": raw.get("last_seen_step"),
            }
        )
    return rows
