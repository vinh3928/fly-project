"""Hazard conditioning: repeated devil risks remembered across deep-thought ticks.

Conditioned risks feed a memory block into devil drafts. Acknowledged risks stay
suppressed from that block / Present until the next non-deep judgement cycle.
Consecutive agreeing accept/acknowledge judgements can early-unlock a freeze.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

RISK_KINDS = ("wall", "toxin", "odor", "fall", "energy")
RiskKind = Literal["wall", "toxin", "odor", "fall", "energy"]
EARLY_UNLOCK_DECISIONS = frozenset({"accept", "acknowledge"})


@dataclass
class ConditioningConfig:
    """Hit threshold for conditioned status; confirmations for early unlock."""

    hit_threshold: int = 3
    judgement_confirmations: int = 2

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> ConditioningConfig:
        data = dict(raw or {})
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class HazardEntry:
    risk: str
    hits: int = 0
    strength: float = 0.0
    last_reason: str = ""
    acknowledged: bool = False

    @property
    def conditioned(self) -> bool:
        return self.hits > 0  # strength set when hits meet threshold externally

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk": self.risk,
            "hits": int(self.hits),
            "strength": float(self.strength),
            "last_reason": self.last_reason,
            "acknowledged": bool(self.acknowledged),
            "conditioned": bool(self.strength > 0.0 or self.hits > 0),
        }


@dataclass
class HazardConditioning:
    """Accumulate repeated hazards; gate display after acknowledge; confirm judgements."""

    config: ConditioningConfig = field(default_factory=ConditioningConfig)
    _entries: dict[str, HazardEntry] = field(default_factory=dict)
    _confirm_key: tuple[str, str] | None = None
    _confirm_count: int = 0
    _freeze_acknowledged: set[str] = field(default_factory=set)

    def reset(self) -> None:
        self._entries.clear()
        self.reset_confirmation()
        self._freeze_acknowledged.clear()

    def reset_confirmation(self) -> None:
        self._confirm_key = None
        self._confirm_count = 0

    def begin_freeze(self) -> None:
        """Reset confirmation streak for a new deep-thought episode; keep entries."""
        self.reset_confirmation()
        self._freeze_acknowledged.clear()

    def end_freeze(self) -> None:
        """Freeze over: keep conditioning memory; confirmation streak ends."""
        self.reset_confirmation()

    def begin_judgement_cycle(self) -> None:
        """Non-deep judgement: allow previously acknowledged risks to surface again."""
        for entry in self._entries.values():
            entry.acknowledged = False
        self._freeze_acknowledged.clear()

    def observe(
        self,
        risk: str,
        *,
        confidence: float = 0.0,
        reason: str = "",
        signals: dict[str, Any] | None = None,
    ) -> HazardEntry | None:
        """Record a devil risk observation. Ignores none/empty."""
        kind = str(risk or "").strip().lower()
        if kind not in RISK_KINDS:
            return None
        entry = self._entries.get(kind)
        if entry is None:
            entry = HazardEntry(risk=kind)
            self._entries[kind] = entry
        entry.hits += 1
        conf = float(confidence or 0.0)
        sig_strength = _signal_strength(kind, signals or {})
        entry.strength = max(entry.strength, conf, sig_strength)
        if reason:
            entry.last_reason = str(reason)[:80]
        thr = max(1, int(self.config.hit_threshold))
        if entry.hits < thr:
            # Not yet fully conditioned — still track strength for memory hints.
            pass
        return entry

    def acknowledge(self, risk: str) -> None:
        kind = str(risk or "").strip().lower()
        if kind not in RISK_KINDS:
            return
        entry = self._entries.get(kind)
        if entry is None:
            entry = HazardEntry(risk=kind, hits=1)
            self._entries[kind] = entry
        entry.acknowledged = True
        self._freeze_acknowledged.add(kind)

    def is_acknowledged(self, risk: str) -> bool:
        kind = str(risk or "").strip().lower()
        entry = self._entries.get(kind)
        return bool(entry and entry.acknowledged) or kind in self._freeze_acknowledged

    def is_conditioned(self, risk: str) -> bool:
        kind = str(risk or "").strip().lower()
        entry = self._entries.get(kind)
        if entry is None:
            return False
        return entry.hits >= max(1, int(self.config.hit_threshold))

    def active_risks(self, *, include_acknowledged: bool = False) -> list[HazardEntry]:
        thr = max(1, int(self.config.hit_threshold))
        out: list[HazardEntry] = []
        for kind in RISK_KINDS:
            entry = self._entries.get(kind)
            if entry is None or entry.hits <= 0:
                continue
            if not include_acknowledged and (
                entry.acknowledged or kind in self._freeze_acknowledged
            ):
                continue
            # Surface once seen at least once during freeze; mark conditioned in text.
            if entry.hits >= 1:
                out.append(entry)
        # Prefer conditioned (hit threshold) first, then by strength.
        out.sort(
            key=lambda e: (e.hits >= thr, e.strength, e.hits),
            reverse=True,
        )
        return out

    def memory_block(self, *, include_acknowledged: bool = False) -> str:
        """Compact draft line. Acknowledged risks omitted unless include_acknowledged."""
        parts: list[str] = []
        thr = max(1, int(self.config.hit_threshold))
        for entry in self.active_risks(include_acknowledged=include_acknowledged):
            tag = "conditioned" if entry.hits >= thr else "emerging"
            reason = f' "{entry.last_reason}"' if entry.last_reason else ""
            ack = " suppressed" if entry.acknowledged else ""
            parts.append(
                f"{entry.risk}×{entry.hits} strength={entry.strength:.2f} ({tag}){reason}{ack}"
            )
        if include_acknowledged:
            for kind in sorted(self._freeze_acknowledged):
                entry = self._entries.get(kind)
                if entry is None:
                    continue
                if not entry.acknowledged:
                    continue
                # Already listed above when include_acknowledged; skip duplicates.
                if any(p.startswith(f"{kind}×") for p in parts):
                    continue
                parts.append(
                    f"{kind}×{entry.hits} strength={entry.strength:.2f} (acknowledged) suppressed"
                )
        if not parts:
            return ""
        return "CONDITIONING: " + "; ".join(parts)

    def to_signals(self) -> dict[str, Any]:
        """Signal payload keys for devil / judgement / Present."""
        out: dict[str, Any] = {}
        summary = self.memory_block(include_acknowledged=False)
        if summary:
            out["conditioning_summary"] = summary
        for entry in self.active_risks(include_acknowledged=False):
            out[f"conditioning_{entry.risk}"] = float(entry.strength)
            out[f"conditioning_{entry.risk}_hits"] = int(entry.hits)
        # Tip heuristic: strongest active conditioned risk.
        thr = max(1, int(self.config.hit_threshold))
        conditioned = [e for e in self.active_risks() if e.hits >= thr]
        if conditioned:
            top = conditioned[0]
            out["conditioning_top_risk"] = top.risk
            out["conditioning_top_strength"] = float(top.strength)
        return out

    def record_judgement(
        self,
        decision: str,
        risk: str,
        proposal_signature: tuple[Any, ...] | list[Any] | None = None,
    ) -> int:
        """Update consecutive confirmation streak. Returns current count."""
        del proposal_signature  # reserved; key is (decision, risk)
        dec = str(decision or "").strip().lower()
        kind = str(risk or "none").strip().lower() or "none"
        key = (dec, kind)
        if self._confirm_key == key:
            self._confirm_count += 1
        else:
            self._confirm_key = key
            self._confirm_count = 1
        return int(self._confirm_count)

    def confirmation_count(self) -> int:
        return int(self._confirm_count)

    def confirmation_key(self) -> tuple[str, str] | None:
        return self._confirm_key

    def confirmed(self, min_confirmations: int | None = None) -> bool:
        need = int(
            self.config.judgement_confirmations if min_confirmations is None else min_confirmations
        )
        if self._confirm_count < max(1, need):
            return False
        if self._confirm_key is None:
            return False
        return self._confirm_key[0] in EARLY_UNLOCK_DECISIONS

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": {k: v.to_dict() for k, v in self._entries.items()},
            "confirm_key": list(self._confirm_key) if self._confirm_key else None,
            "confirm_count": int(self._confirm_count),
            "freeze_acknowledged": sorted(self._freeze_acknowledged),
            "memory_block": self.memory_block(),
        }

    def load_from_dict(self, data: dict[str, Any] | None, *, merge: bool = False) -> bool:
        """Restore entries (and optional confirm/ack state) from an STM snapshot.

        Returns True if any entries were loaded.
        """
        if not isinstance(data, dict):
            return False
        raw_entries = data.get("entries")
        if not isinstance(raw_entries, dict) or not raw_entries:
            return False
        if not merge:
            self._entries.clear()
            self._freeze_acknowledged.clear()
            self.reset_confirmation()
        for kind, raw in raw_entries.items():
            risk = str(kind or "").strip().lower()
            if risk not in RISK_KINDS:
                continue
            if not isinstance(raw, dict):
                continue
            entry = HazardEntry(
                risk=risk,
                hits=int(raw.get("hits", 0) or 0),
                strength=float(raw.get("strength", 0.0) or 0.0),
                last_reason=str(raw.get("last_reason", "") or "")[:80],
                acknowledged=bool(raw.get("acknowledged", False)),
            )
            if entry.hits <= 0 and entry.strength <= 0.0:
                continue
            existing = self._entries.get(risk) if merge else None
            if existing is not None:
                existing.hits = max(existing.hits, entry.hits)
                existing.strength = max(existing.strength, entry.strength)
                if entry.last_reason:
                    existing.last_reason = entry.last_reason
                existing.acknowledged = existing.acknowledged or entry.acknowledged
            else:
                self._entries[risk] = entry
            if entry.acknowledged:
                self._freeze_acknowledged.add(risk)
        # Confirm streak is freeze-local; restore only when not merging.
        if not merge:
            key = data.get("confirm_key")
            if isinstance(key, (list, tuple)) and len(key) >= 2:
                self._confirm_key = (str(key[0]), str(key[1]))
                self._confirm_count = int(data.get("confirm_count", 0) or 0)
            ack = data.get("freeze_acknowledged")
            if isinstance(ack, (list, tuple, set)):
                for item in ack:
                    k = str(item).strip().lower()
                    if k in RISK_KINDS:
                        self._freeze_acknowledged.add(k)
                        ent = self._entries.get(k)
                        if ent is not None:
                            ent.acknowledged = True
        return bool(self._entries)

    @staticmethod
    def extract_from_signals(signals: dict[str, Any] | None) -> dict[str, Any] | None:
        """Pull a conditioning snapshot from an STM signals dict, if present."""
        if not isinstance(signals, dict):
            return None
        blob = signals.get("conditioning")
        if isinstance(blob, dict) and isinstance(blob.get("entries"), dict) and blob["entries"]:
            return blob
        return None


def conditioning_snapshot_signals(store: HazardConditioning) -> dict[str, Any]:
    """Signals payload fragment for STM rows (checkpoint or mind tick)."""
    snap = store.to_dict()
    out: dict[str, Any] = {
        "conditioning": snap,
        "conditioning_checkpoint": True,
    }
    summary = snap.get("memory_block") or store.memory_block()
    if summary:
        out["conditioning_summary"] = summary
    out.update(store.to_signals())
    return out


def load_conditioning_from_stm(
    store: HazardConditioning,
    memory: Any,
    *,
    fly_id: str,
    limit: int = 48,
) -> bool:
    """Hydrate ``store`` from the newest STM row that carries a conditioning blob."""
    try:
        recent = memory.recent(fly_id=fly_id, limit=int(limit))
    except Exception:
        return False
    for rec in recent:
        blob = HazardConditioning.extract_from_signals(getattr(rec, "signals", None))
        if blob is None:
            continue
        if store.load_from_dict(blob):
            return True
    return False


def _signal_strength(risk: str, signals: dict[str, Any]) -> float:
    if risk == "wall":
        return float(signals.get("wall_hazard", 0.0) or 0.0)
    if risk == "toxin":
        return float(signals.get("bitter", 0.0) or 0.0)
    if risk == "odor":
        left = float(signals.get("odor_left", 0.0) or 0.0)
        right = float(signals.get("odor_right", 0.0) or 0.0)
        odor = float(signals.get("odor", 0.0) or 0.0)
        raw = max(odor, left, right)
        # Heightened odor can exceed 1; normalize for strength display.
        return float(min(1.0, raw if raw <= 1.0 else raw / 10.0))
    if risk == "fall":
        return float(signals.get("survival_risk", signals.get("instability", 0.0)) or 0.0)
    if risk == "energy":
        energy = float(signals.get("energy", 1.0) or 1.0)
        return float(max(0.0, 1.0 - energy))
    return 0.0


def rewrite_present_after_judgement(
    refined_thought: str,
    *,
    signals: dict[str, Any],
    affect: dict[str, Any],
    maturity: float = 0.0,
    judgement: dict[str, Any] | None = None,
    conditioning_summary: str = "",
) -> str:
    """Replace Present: prefix with chemistry + judgement + active conditioning."""
    from flybrain.thinking.two_layer_future import format_present_line

    present = format_present_line(
        signals,
        affect,
        maturity=maturity,
        judgement=judgement,
        conditioning_summary=conditioning_summary,
    )
    text = str(refined_thought or "").strip()
    rest = text
    if rest.startswith("Present:"):
        for marker in (" Future hypothesis", " Refined after"):
            idx = rest.find(marker)
            if idx > 0:
                rest = rest[idx:].lstrip()
                break
        else:
            dot = rest.find(". ")
            rest = rest[dot + 2 :].lstrip() if dot >= 0 else ""
    while rest.startswith("Judged:"):
        dot = rest.find(". ")
        rest = rest[dot + 2 :].lstrip() if dot >= 0 else ""
    return f"{present} {rest}".strip() if rest else present
