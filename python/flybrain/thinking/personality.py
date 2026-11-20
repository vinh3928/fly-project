"""Live personality traits: seeded once, continuously rewritten by later systems.

Judgement reads ``snapshot()`` each mind tick and never owns trait learning.
Later systems (outcomes, retrospective, developmental, …) call ``adjust`` /
``set_traits`` so accept/reject behavior drifts without changing judgement rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, float(x))))


TRAIT_NAMES = (
    "risk_appetite",
    "playfulness",
    "curiosity_to_test",
    "intelligence",
    "wisdom",
)


@dataclass
class PersonalityTraits:
    """Point-in-time trait vector. All values in [0, 1]."""

    risk_appetite: float = 0.45  # 0 = averse .. 1 = seeking
    playfulness: float = 0.55
    curiosity_to_test: float = 0.70
    intelligence: float = 0.50
    wisdom: float = 0.30

    def clamped(self) -> PersonalityTraits:
        return PersonalityTraits(**{name: _clamp01(getattr(self, name)) for name in TRAIT_NAMES})

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in TRAIT_NAMES}

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> PersonalityTraits:
        data = dict(raw or {})
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: float(v) for k, v in data.items() if k in known}).clamped()


@dataclass
class PersonalityState:
    """Seeded once; continuously rewritten by later systems. Judgement only reads."""

    traits: PersonalityTraits = field(default_factory=PersonalityTraits)
    history_limit: int = 32
    _history: list[dict[str, Any]] = field(default_factory=list, repr=False)

    @classmethod
    def from_config(cls, raw: dict[str, Any] | None) -> PersonalityState:
        data = dict(raw or {})
        history_limit = int(data.pop("history_limit", 32))
        return cls(traits=PersonalityTraits.from_dict(data), history_limit=history_limit)

    def snapshot(self) -> PersonalityTraits:
        return PersonalityTraits(**self.traits.as_dict())

    def adjust(self, deltas: dict[str, float], *, source: str = "external") -> PersonalityTraits:
        applied: dict[str, float] = {}
        for name, delta in deltas.items():
            if name not in TRAIT_NAMES:
                continue
            before = float(getattr(self.traits, name))
            after = _clamp01(before + float(delta))
            setattr(self.traits, name, after)
            applied[name] = after - before
        self._record(source=source, kind="adjust", applied=applied)
        return self.snapshot()

    def set_traits(
        self, values: dict[str, float], *, source: str = "external"
    ) -> PersonalityTraits:
        applied: dict[str, float] = {}
        for name, value in values.items():
            if name not in TRAIT_NAMES:
                continue
            before = float(getattr(self.traits, name))
            after = _clamp01(value)
            setattr(self.traits, name, after)
            applied[name] = after - before
        self._record(source=source, kind="set", applied=applied)
        return self.snapshot()

    def history(self, n: int = 8) -> list[dict[str, Any]]:
        if n <= 0:
            return []
        return list(self._history[-n:])

    def _record(self, *, source: str, kind: str, applied: dict[str, float]) -> None:
        if not applied:
            return
        self._history.append(
            {
                "source": str(source),
                "kind": kind,
                "applied": dict(applied),
                "traits": self.traits.as_dict(),
            }
        )
        if len(self._history) > self.history_limit:
            self._history = self._history[-self.history_limit :]

    def as_dict(self) -> dict[str, Any]:
        return {
            "traits": self.traits.as_dict(),
            "last_source": (self._history[-1]["source"] if self._history else None),
        }
