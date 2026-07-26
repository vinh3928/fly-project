"""Sensor feedback: report mapped Neo4j items at a decayed rate.

Raw transducers (eyes, antennae, taste) stay honest. Once the current place
(and its walls / other items) exist in the place graph, this loop scales the
*reported* sensor channels so downstream mind/motor see habituated rates.
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

    def reported(self, channel: str, default: float = 0.0) -> float:
        r = self.readings.get(channel)
        return float(r.reported) if r is not None else float(default)

    def raw(self, channel: str, default: float = 0.0) -> float:
        r = self.readings.get(channel)
        return float(r.raw) if r is not None else float(default)

    def to_place_info(self) -> dict[str, Any]:
        """Log-friendly place + wall snapshot (includes wall sensor raw/reported)."""
        wall = self.readings.get("wall_hazard")
        out: dict[str, Any] = {
            "place_id": self.place.id if self.place is not None else None,
            "similarity": float(self.similarity),
            "is_new": bool(self.is_new),
            "mapped": not bool(self.is_new) and self.place is not None,
            "mean_familiarity": float(self.mean_familiarity),
            "walls": (
                {d: dict(self.place.walls.get(d, {})) for d in WALL_DIRS}
                if self.place is not None
                else {}
            ),
            "items": dict((self.place.props or {}).get("items") or {})
            if self.place is not None
            else {},
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
    """Feedback loop: map place → attenuate reported sensors for known items."""

    store: PlaceGraphStore
    config: PlaceGraphConfig
    fly_id: str = "fly_001"
    channels: tuple[str, ...] = DEFAULT_FEEDBACK_CHANNELS
    _last_place_id: int | str | None = field(default=None, init=False, repr=False)

    def sense(
        self,
        *,
        raw: dict[str, float],
        view_vec: np.ndarray | None = None,
        step: int | None = None,
    ) -> MappedSensorReport:
        """Upsert place from view (if any), then report attenuated sensor rates."""
        cfg = self.config
        place: PlaceNode | None = None
        is_new = True
        similarity = 0.0
        mean_fam = 0.0
        mapped = False

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
            self._last_place_id = place.id if place is not None else self._last_place_id

            # Persist non-wall item familiarity on the place node.
            if mapped and place is not None:
                items = _bump_detected_items(
                    place,
                    raw,
                    bump=float(cfg.familiarity_bump),
                    step=step,
                )
                prior_items = place.props.get("items") or {}
                if items != prior_items:
                    refreshed = self.store.refresh_place(
                        place.id,
                        place.view_vec,
                        walls=place.walls,
                        props={**dict(place.props), "items": items},
                        step=step,
                    )
                    if refreshed is not None:
                        place = refreshed
                        mean_fam = mean_wall_familiarity(place.walls)

        readings: dict[str, MappedSensorReading] = {}
        for channel in self.channels:
            if channel not in raw:
                continue
            raw_v = float(raw[channel] or 0.0)
            fam = item_familiarity(place, channel, mapped=mapped)
            # Wall uses wall familiarity; other items need their own fam > 0 to decay.
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
        )
