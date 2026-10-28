"""Sugar search: steer by whether the smell is getting stronger over time.

Left/right antenna comparison alone cannot tell "sugar ahead" from "sugar behind"
(both read a zero bearing), and it never notices that the fly is not getting closer.
This tracker compares sugar smell across mind ticks:

- rising   → approach: gentle left/right antenna steering
- flat     → search: turn harder each tick the smell fails to rise
- dropping sharply → reverse: near U-turn
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from flybrain.thinking.composer import FEEDING_UNTIL_SATED_KEY, is_hungry
from flybrain.thinking.two_layer_future import SUGAR_SMELL_MIN, sugar_approach_turn


@dataclass
class SugarSearchConfig:
    enabled: bool = True
    # Relative smell change per mind tick counted as "getting closer".
    rise_eps: float = 0.002
    # Relative drop per mind tick treated as heading away → reverse.
    sharp_drop: float = 0.05
    # Search turn starts here and grows each non-improving tick, up to max_turn.
    base_turn: float = 0.25
    turn_step: float = 0.1
    max_turn: float = 0.75
    reverse_turn: float = 1.0
    # Bearing magnitude needed to trust the antennae for picking a side.
    bearing_min: float = 0.05
    # Per-motor-step orient reflex. With antenna tips 38 mm apart and
    # ANTENNA_BEARING_GAIN the bearing is ≈ 1.75·sin(angle to sugar), saturating
    # at ±1 beyond ~35°.
    orient_enabled: bool = True
    steer_gain: float = 1.0
    cruise_forward: float = 0.6
    # |turn| at or above this ⇒ forward 0 (pivot in place).
    pivot_turn: float = 0.8
    # Arrive: at this smell (≈11 cm in the 1 m plume) extend the proboscis and
    # slow by up to arrive_slow as smell → 1.
    probe_smell: float = 0.6
    probe_extension: float = 1.0
    arrive_slow: float = 0.6
    # Sugar taste (feet or proboscis) that counts as "on food" (Unity feeds above 0.2).
    feed_taste_min: float = 0.2
    # Feed bout: motor stopped, proboscis down, for this many motor ticks
    # (= one mind_interval). Repeats while the proboscis tastes sugar and hungry;
    # a bout with no proboscis contact pauses feet-only starts for as long.
    feed_ticks: int = 40

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> SugarSearchConfig:
        data = dict(raw or {})
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class SugarSearchState:
    mode: str  # "off" | "start" | "approach" | "search" | "reverse"
    turn: float | None
    change: float = 0.0
    streak: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "turn": None if self.turn is None else float(self.turn),
            "change": float(self.change),
            "streak": int(self.streak),
        }


@dataclass
class SugarOrient:
    forward: float
    turn: float
    proboscis: float
    mode: str  # "orient" | "probe" | "feed"
    ticks_left: int = 0  # remaining motor ticks in the current feed bout

    def to_list(self) -> list[float]:
        return [float(self.forward), float(self.turn), float(self.proboscis)]


class SugarSearchTracker:
    def __init__(self, config: SugarSearchConfig | None = None) -> None:
        self.config = config or SugarSearchConfig()
        self.reset()

    def reset(self) -> None:
        self._last_smell: float | None = None
        self._streak = 0
        self._side = 1.0
        self._state = SugarSearchState(mode="off", turn=None)
        self._locked = False
        self._bout_left = 0
        self._bout_fed = False
        self._cooldown = 0
        self._feeding_session = False

    @property
    def feeding_until_sated(self) -> bool:
        """A feed bout has started; food seeking holds until energy reaches ENERGY_SATED."""
        return self._feeding_session

    def _pick_side(self, bearing: float) -> float:
        if abs(bearing) >= self.config.bearing_min:
            self._side = float(np.sign(bearing))
        return self._side

    def observe(self, signals: dict[str, Any]) -> SugarSearchState:
        """Call once per mind tick with the thinking signals."""
        self._state = self._observe(signals)
        self._locked = False
        return self._state

    @property
    def active(self) -> bool:
        return self.config.orient_enabled and self._state.turn is not None

    @property
    def feed_ticks_left(self) -> int:
        return int(self._bout_left)

    def _tastes_sugar(self, signals: dict[str, Any]) -> bool:
        """Sugar under the fly: feet or proboscis taste (summary ``sugar``)."""
        return float(signals.get("sugar", 0.0) or 0.0) >= self.config.feed_taste_min

    def _proboscis_on_sugar(self, signals: dict[str, Any]) -> bool:
        return float(signals.get("proboscis_contact", 0.0) or 0.0) >= 0.5 and self._tastes_sugar(
            signals
        )

    def _should_start_bout(self, signals: dict[str, Any]) -> bool:
        if self._proboscis_on_sugar(signals):
            return True
        # Feet-only taste starts a bout unless the last one never reached the proboscis.
        return self._cooldown <= 0 and self._tastes_sugar(signals)

    def feeding_now(self, signals: dict[str, Any]) -> bool:
        """In a feed bout, or hungry and standing on sugar (bout about to start)."""
        return self._bout_left > 0 or (self.active and self._should_start_bout(signals))

    def steer(self, signals: dict[str, Any]) -> SugarOrient | None:
        """Per-motor-step command toward sugar, or None when not searching.

        Turn follows the live antenna bearing so the heading is corrected every
        step; forward shrinks as the turn grows, pivoting in place for sharp
        angles instead of orbiting. While the smell trend says search/reverse,
        the held search turn is used until the antennae pick a side. Near the
        source the proboscis extends and the fly slows. Once on top of sugar the
        motor stops for a feed bout of ``feed_ticks`` steps (proboscis down); a
        new bout follows while the proboscis keeps tasting sugar and the fly is
        still hungry.
        """
        cfg = self.config
        state = self._state
        if not self.active:
            return None
        if self._bout_left <= 0 and self._should_start_bout(signals):
            self._bout_left = max(1, int(cfg.feed_ticks))
            self._bout_fed = False
            self._feeding_session = True
        if self._bout_left > 0:
            self._bout_left -= 1
            self._bout_fed = self._bout_fed or self._proboscis_on_sugar(signals)
            if self._bout_left == 0 and not self._bout_fed:
                self._cooldown = max(1, int(cfg.feed_ticks))
            return SugarOrient(
                forward=0.0,
                turn=0.0,
                proboscis=cfg.probe_extension,
                mode="feed",
                ticks_left=self._bout_left,
            )
        if self._cooldown > 0:
            self._cooldown -= 1
        bearing = float(signals.get("sugar_bearing", 0.0) or 0.0)
        smell = float(signals.get("sugar_smell", 0.0) or 0.0)
        bearing_turn = float(np.clip(cfg.steer_gain * bearing, -1.0, 1.0))
        if abs(bearing) >= cfg.bearing_min:
            self._locked = True
        if state.mode in ("search", "reverse") and not self._locked:
            turn = float(state.turn)
        else:
            turn = bearing_turn
        pivot = max(cfg.pivot_turn, 1e-6)
        forward = cfg.cruise_forward * float(np.clip(1.0 - abs(turn) / pivot, 0.0, 1.0))
        near = float(np.clip((smell - cfg.probe_smell) / max(1.0 - cfg.probe_smell, 1e-6), 0.0, 1.0))
        if smell >= cfg.probe_smell:
            forward *= 1.0 - cfg.arrive_slow * near
            return SugarOrient(forward=forward, turn=turn, proboscis=cfg.probe_extension, mode="probe")
        return SugarOrient(forward=forward, turn=turn, proboscis=0.0, mode="orient")

    def _observe(self, signals: dict[str, Any]) -> SugarSearchState:
        cfg = self.config
        smell = float(signals.get("sugar_smell", 0.0) or 0.0)
        bearing = float(signals.get("sugar_bearing", 0.0) or 0.0)
        if self._feeding_session:
            signals = {**signals, FEEDING_UNTIL_SATED_KEY: True}
        if not cfg.enabled or not is_hungry(signals) or smell < SUGAR_SMELL_MIN:
            self.reset()
            return SugarSearchState(mode="off", turn=None)

        last = self._last_smell
        self._last_smell = smell
        if last is None:
            self._pick_side(bearing)
            return SugarSearchState(mode="start", turn=sugar_approach_turn(bearing))

        change = (smell - last) / max(last, 1e-6)
        side = self._pick_side(bearing)
        if change > cfg.rise_eps:
            self._streak = 0
            return SugarSearchState(
                mode="approach", turn=sugar_approach_turn(bearing), change=change
            )
        if change <= -cfg.sharp_drop:
            self._streak = 0
            return SugarSearchState(
                mode="reverse", turn=side * cfg.reverse_turn, change=change
            )
        self._streak += 1
        magnitude = min(cfg.max_turn, cfg.base_turn + cfg.turn_step * (self._streak - 1))
        return SugarSearchState(
            mode="search", turn=side * magnitude, change=change, streak=self._streak
        )
