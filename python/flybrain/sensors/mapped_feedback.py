"""Sensor feedback: report mapped Neo4j items at a decayed rate.

Raw transducers (eyes, antennae, taste) stay honest. Once the current place
(and its walls / other items) exist in the place graph, this loop scales the
*reported* sensor channels so downstream mind/motor see habituated rates.

Vision keys the Place node; smell/sensor readings are stored as correlated
details on that node (and Neo4j ``HAS_SENSE`` edges). ``ModalChangeGate`` keeps
unchanged smell/sensor snapshots on L1 so Neo4j is not touched every step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from flybrain.memory.place_graph import (
    WALL_DIRS,
    PlaceGraphConfig,
    PlaceGraphStore,
    PlaceNode,
    mean_wall_familiarity,
    observe_place,
)
from flybrain.phase4 import SMELL_VEC_DIM
from flybrain.sensors.modal_gate import ModalChangeGate
from flybrain.sensors.place_senses import (
    SENSE_CHANNELS,
    infer_place_label,
    merge_place_senses,
    senses_vector,
)

# Sensor channel → place-graph item key (walls are special-cased).
CHANNEL_ITEMS: dict[str, str] = {
    "wall_hazard": "wall",
    "odor": "odor",
    "odor_left": "odor",
    "odor_right": "odor",
    "bitter": "bitter",
    "sugar": "sugar",
    "toxin_taste": "toxin",
}

# Channels attenuated when their mapped item is known.
DEFAULT_FEEDBACK_CHANNELS: tuple[str, ...] = (
    "wall_hazard",
    "odor",
    "odor_left",
    "odor_right",
    "bitter",
    "sugar",
)


def _clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def attenuate_mapped_reading(
    raw: float,
    *,
    mapped: bool,
    familiarity: float,
    strength: float = 0.85,
    floor: float = 0.15,
) -> tuple[float, float]:
    """Return ``(reported, scale)`` for a sensor once its item is mapped."""
    value = _clamp01(float(raw))
    if not mapped:
        return value, 1.0
    fam = _clamp01(float(familiarity))
    scale = max(float(floor), 1.0 - float(strength) * fam)
    return _clamp01(value * scale), float(scale)


@dataclass
class MappedSensorReading:
    """One sensor channel after place-map feedback."""

    channel: str
    item: str
    raw: float
    reported: float
    mapped: bool
    familiarity: float
    scale: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "item": self.item,
            "raw": float(self.raw),
            "reported": float(self.reported),
            "mapped": bool(self.mapped),
            "familiarity": float(self.familiarity),
            "scale": float(self.scale),
        }


@dataclass
class MappedSensorReport:
    """Full sensor report for one tick after Neo4j place feedback."""

    place: PlaceNode | None
    is_new: bool
    similarity: float
    mean_familiarity: float
    readings: dict[str, MappedSensorReading] = field(default_factory=dict)
    label: str = "place"
    senses_l1_hit: bool = False
    smell_l1_hit: bool = False
    match_reason: str = ""

    def reported(self, channel: str, default: float = 0.0) -> float:
        r = self.readings.get(channel)
        return float(r.reported) if r is not None else float(default)

    def raw(self, channel: str, default: float = 0.0) -> float:
        r = self.readings.get(channel)
        return float(r.raw) if r is not None else float(default)

    def to_place_info(self) -> dict[str, Any]:
        """Log-friendly place + wall snapshot (includes wall sensor raw/reported)."""
        wall = self.readings.get("wall_hazard")
        props = dict(self.place.props) if self.place is not None else {}
        out: dict[str, Any] = {
            "place_id": self.place.id if self.place is not None else None,
            "label": str(props.get("label") or self.label or "place"),
            "similarity": float(self.similarity),
            "is_new": bool(self.is_new),
            "mapped": not bool(self.is_new) and self.place is not None,
            "mean_familiarity": float(self.mean_familiarity),
            "match_reason": self.match_reason,
            "senses_l1_hit": bool(self.senses_l1_hit),
            "smell_l1_hit": bool(self.smell_l1_hit),
            "walls": (
                {d: dict(self.place.walls.get(d, {})) for d in WALL_DIRS}
                if self.place is not None
                else {}
            ),
            "items": dict(props.get("items") or {}),
            "senses": dict(props.get("senses") or {}),
            "sensors": {k: v.to_dict() for k, v in self.readings.items()},
        }
        if wall is not None:
            out["wall_hazard_raw"] = float(wall.raw)
            out["wall_hazard"] = float(wall.reported)
        return out


def item_familiarity(place: PlaceNode | None, channel: str, *, mapped: bool) -> float:
    """Familiarity for the graph item backing a sensor channel."""
    if not mapped or place is None:
        return 0.0
    item = CHANNEL_ITEMS.get(channel, channel)
    if item == "wall":
        return mean_wall_familiarity(place.walls)
    items = place.props.get("items") if isinstance(place.props, dict) else None
    if not isinstance(items, dict):
        return 0.0
    row = items.get(item)
    if not isinstance(row, dict):
        return 0.0
    return _clamp01(float(row.get("familiarity") or 0.0))


def _bump_detected_items(
    place: PlaceNode,
    raw: dict[str, float],
    *,
    bump: float,
    step: int | None,
    present_min: float = 0.05,
) -> dict[str, Any]:
    """Raise familiarity on place.props.items for channels that are currently present."""
    items = dict(place.props.get("items") or {}) if isinstance(place.props, dict) else {}
    for channel, value in raw.items():
        item = CHANNEL_ITEMS.get(channel)
        if item is None or item == "wall":
            continue
        if float(value) < float(present_min):
            continue
        row = dict(items.get(item) or {"familiarity": 0.0, "last_seen_step": None})
        row["familiarity"] = _clamp01(float(row.get("familiarity") or 0.0) + float(bump))
        if step is not None:
            row["last_seen_step"] = int(step)
        items[item] = row
    return items


@dataclass
class MappedSensorFeedback:
    """Feedback loop: map place → attach senses → attenuate known items."""

    store: PlaceGraphStore
    config: PlaceGraphConfig
    fly_id: str = "fly_001"
    channels: tuple[str, ...] = DEFAULT_FEEDBACK_CHANNELS
    # L1 sample-diff gates: unchanged smell/sensors stay off Neo4j.
    sense_gate_ttl_seconds: float = 10.0
    smell_gate_ttl_seconds: float = 10.0
    _last_place_id: int | str | None = field(default=None, init=False, repr=False)
    _sense_gate: ModalChangeGate = field(init=False, repr=False)
    _smell_gate: ModalChangeGate = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._sense_gate = ModalChangeGate(
            modality="sensors",
            ttl_seconds=float(self.sense_gate_ttl_seconds),
            step_interval=1,
            change_min=0.05,
            refresh_seconds=None,
            compare="max_abs",
            dim=len(SENSE_CHANNELS),
            first_nonzero_is_change=True,
        )
        self._smell_gate = ModalChangeGate(
            modality="smell",
            ttl_seconds=float(self.smell_gate_ttl_seconds),
            step_interval=1,
            change_min=0.05,
            refresh_seconds=None,
            compare="max_abs",
            dim=int(SMELL_VEC_DIM),
            first_nonzero_is_change=True,
        )

    def reset(self) -> None:
        self._last_place_id = None
        self._sense_gate.reset()
        self._smell_gate.reset()

    def sense(
        self,
        *,
        raw: dict[str, float],
        view_vec: np.ndarray | None = None,
        smell_vec: np.ndarray | None = None,
        step: int | None = None,
        now_ms: float | None = None,
    ) -> MappedSensorReport:
        """Upsert vision Place, attach correlated senses, attenuate known items."""
        cfg = self.config
        place: PlaceNode | None = None
        is_new = True
        similarity = 0.0
        mean_fam = 0.0
        mapped = False
        match_reason = ""
        label = infer_place_label(raw)
        senses_l1 = False
        smell_l1 = False

        sense_sample = self._sense_gate.observe(
            senses_vector(raw),
            now_ms=now_ms if now_ms is not None else float(step or 0),
            step=step,
        )
        smell_sample = self._smell_gate.observe(
            smell_vec,
            now_ms=now_ms if now_ms is not None else float(step or 0) + 0.001,
            step=step,
        )
        senses_l1 = bool(sense_sample.l1_hit)
        smell_l1 = bool(smell_sample.l1_hit)

        if view_vec is not None and cfg.enabled:
            obs = observe_place(
                self.store,
                view_vec,
                fly_id=self.fly_id,
                match_min=float(cfg.match_min),
                step=step,
                familiarity_bump=float(cfg.familiarity_bump),
                from_place_id=self._last_place_id,
                context_enabled=bool(cfg.context_enabled),
                context_top_k=int(cfg.context_top_k),
            )
            place = obs.place
            is_new = bool(obs.is_new)
            similarity = float(obs.similarity)
            mean_fam = float(obs.mean_familiarity)
            mapped = not is_new
            match_reason = str(obs.match_reason)
            self._last_place_id = place.id if place is not None else self._last_place_id

            if place is not None:
                # Unchanged smell/sensors on a revisit → stay on L1 (skip durable write).
                senses_stable = senses_l1 and (smell_vec is None or smell_l1)
                should_write = bool(
                    is_new
                    or sense_sample.changed
                    or smell_sample.changed
                    or not senses_stable
                )
                if should_write:
                    items = _bump_detected_items(
                        place,
                        raw,
                        bump=float(cfg.familiarity_bump),
                        step=step,
                    )
                    merged = merge_place_senses(
                        {**dict(place.props), "items": items},
                        raw,
                        step=step,
                        smell_vec=smell_vec,
                        label=label,
                    )
                    label = str(merged.get("label") or label)
                    refreshed = self.store.refresh_place(
                        place.id,
                        place.view_vec,
                        walls=place.walls,
                        props=merged,
                        step=step,
                    )
                    if refreshed is not None:
                        place = refreshed
                        mean_fam = mean_wall_familiarity(place.walls)
                else:
                    label = str((place.props or {}).get("label") or label)

        readings: dict[str, MappedSensorReading] = {}
        for channel in self.channels:
            if channel not in raw:
                continue
            raw_v = float(raw[channel] or 0.0)
            fam = item_familiarity(place, channel, mapped=mapped)
            item = CHANNEL_ITEMS.get(channel, channel)
            channel_mapped = mapped and (item == "wall" or fam > 0.0)
            reported, scale = attenuate_mapped_reading(
                raw_v,
                mapped=channel_mapped,
                familiarity=fam if channel_mapped else 0.0,
                strength=float(cfg.hazard_decay_strength),
                floor=float(cfg.hazard_decay_floor),
            )
            readings[channel] = MappedSensorReading(
                channel=channel,
                item=item,
                raw=_clamp01(raw_v),
                reported=reported,
                mapped=channel_mapped,
                familiarity=fam,
                scale=scale,
            )

        return MappedSensorReport(
            place=place,
            is_new=is_new,
            similarity=similarity,
            mean_familiarity=mean_fam,
            readings=readings,
            label=label,
            senses_l1_hit=senses_l1,
            smell_l1_hit=smell_l1,
            match_reason=match_reason,
        )
