"""Trait-weighted judgement of structured devil proposals.

Decisions: accept | reject | ignore | reconsider | acknowledge.
``acknowledge`` parks the stimulus as an interest point for low-load investigation
instead of blending motor now. Judgement rules are stable; personality traits
(read live each call) are what drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from flybrain.thinking.personality import PersonalityTraits


Decision = Literal["accept", "reject", "ignore", "reconsider", "acknowledge"]

_RISK_SEVERITY = {
    "none": 0.0,
    "energy": 0.35,
    "wall": 0.7,
    "toxin": 0.85,
    "fall": 0.8,
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


@dataclass
class DevilProposal:
    """Structured devil output that judgement can accept into the motor goal."""

    risk: str = "none"
    forward_scale: float = 1.0
    turn_delta: float = 0.0
    confidence: float = 0.0
    reason: str = ""
    ran: bool = False
    source: str = "none"  # llm | heuristic | unparsable | empty

    def signature(self) -> tuple[str, int, int]:
        turn_sign = 0 if abs(self.turn_delta) < 0.1 else (1 if self.turn_delta > 0 else -1)
        fwd_bucket = int(_clamp(self.forward_scale, 0.0, 1.0) * 4)
        return (str(self.risk), turn_sign, fwd_bucket)

    def is_noop(self) -> bool:
        return (
            abs(self.forward_scale - 1.0) < 0.05
            and abs(self.turn_delta) < 0.05
            and self.risk in ("none", "")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk": self.risk,
            "forward_scale": float(self.forward_scale),
            "turn_delta": float(self.turn_delta),
            "confidence": float(self.confidence),
            "reason": self.reason,
            "ran": bool(self.ran),
            "source": self.source,
            "signature": list(self.signature()),
        }


@dataclass
class JudgementConfig:
    enabled: bool = True
    reconsider_budget: int = 1
    reject_memory_ticks: int = 80
    base_blend_weight: float = 0.30
    max_blend_weight: float = 0.60
    accept_threshold: float = 0.15
    reject_threshold: float = 0.20
    ignore_weak_confidence: float = 0.2
    # Acknowledge (= save for later) when curious but load/wisdom prefer deferral.
    acknowledge_threshold: float = 0.18
    acknowledge_novelty_min: float = 0.15
    acknowledge_load_min: float = 0.45
    # Block acknowledge when current eyes match a stored acknowledge snapshot this closely.
    acknowledge_eye_match_max: float = 0.4

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "JudgementConfig":
        data = dict(raw or {})
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class JudgementResult:
    decision: Decision
    reason: str
    blend_weight: float
    proposal: DevilProposal
    traits_snapshot: dict[str, float]
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "blend_weight": float(self.blend_weight),
            "proposal": self.proposal.to_dict(),
            "traits": dict(self.traits_snapshot),
            "scores": dict(self.scores),
        }


@dataclass
class JudgementModule:
    config: JudgementConfig = field(default_factory=JudgementConfig)
    _rejects: dict[tuple[str, int, int], int] = field(default_factory=dict)
    _tick: int = 0

    def reset(self) -> None:
        self._rejects.clear()
        self._tick = 0

    def advance_tick(self, tick: int) -> None:
        self._tick = int(tick)
        ttl = int(self.config.reject_memory_ticks)
        dead = [sig for sig, at in self._rejects.items() if self._tick - at > ttl]
        for sig in dead:
            del self._rejects[sig]

    def record_reject(self, proposal: DevilProposal) -> None:
        self._rejects[proposal.signature()] = self._tick

    def evaluate(
        self,
        proposal: DevilProposal,
        *,
        traits: PersonalityTraits,
        signals: dict[str, Any],
        reconsider_used: int = 0,
        eye_match_score: float = 0.0,
    ) -> JudgementResult:
        cfg = self.config
        snap = traits.as_dict()
        if not cfg.enabled or not proposal.ran:
            return JudgementResult(
                decision="ignore",
                reason="disabled_or_empty",
                blend_weight=0.0,
                proposal=proposal,
                traits_snapshot=snap,
            )

        wall = float(signals.get("wall_hazard", 0.0) or 0.0)
        survival = float(signals.get("survival_risk", signals.get("survival", 0.0)) or 0.0)
        if "survival_risk" not in signals and isinstance(signals.get("affect"), dict):
            survival = float(signals["affect"].get("survival_risk", survival))
        novelty = float(
            signals.get(
                "visual_novelty",
                signals.get("novelty", 0.0),
            )
            or 0.0
        )
        if bool(signals.get("new_stimulus")):
            novelty = max(novelty, 0.55)
        # Salient devil risks count as something worth remembering even if viz habituated.
        if proposal.risk in ("wall", "toxin", "fall"):
            novelty = max(
                novelty, 0.35 * float(_RISK_SEVERITY.get(str(proposal.risk), 0.4)) + 0.2
            )

        severity = float(_RISK_SEVERITY.get(str(proposal.risk), 0.4))
        conf = _clamp(proposal.confidence, 0.0, 1.0)
        risk_appetite = float(traits.risk_appetite)
        playfulness = float(traits.playfulness)
        curiosity = float(traits.curiosity_to_test)
        intelligence = float(traits.intelligence)
        wisdom = float(traits.wisdom)
        load = max(wall, survival)

        accept = 0.35 * conf + 0.25 * curiosity + 0.15 * playfulness
        accept -= severity * (1.0 - risk_appetite)
        # Inconsistency: urging forward into a wall → intelligence wants another look.
        inconsistent = (
            proposal.forward_scale > 0.7 and wall > 0.55 and proposal.risk in ("none", "energy")
        ) or (proposal.risk == "wall" and proposal.forward_scale > 0.85)
        reconsider = 0.0
        if inconsistent:
            reconsider += 0.55 * intelligence
        if conf < 0.45:
            reconsider += 0.35 * curiosity + 0.2 * playfulness

        reject = severity * (1.0 - risk_appetite) * 0.9
        reject += 0.25 * (1.0 - conf) * (1.0 - curiosity)
        sig = proposal.signature()
        if sig in self._rejects:
            age = max(0, self._tick - self._rejects[sig])
            freshness = 1.0 - age / max(1, int(cfg.reject_memory_ticks))
            reject += 0.45 * freshness

        ignore = 0.0
        if proposal.is_noop() or conf < float(cfg.ignore_weak_confidence):
            ignore += 0.4 + 0.5 * wisdom
        if abs(proposal.forward_scale - 1.0) < 0.08 and abs(proposal.turn_delta) < 0.08:
            ignore += 0.35 * wisdom

        # Acknowledge: curious enough to care, but not safe/eager enough to accept now.
        acknowledge = 0.0
        if novelty >= float(cfg.acknowledge_novelty_min) or proposal.risk in (
            "wall",
            "toxin",
            "fall",
        ):
            if load >= float(cfg.acknowledge_load_min) and (
                reject >= accept or wisdom >= 0.45
            ):
                defer = max(load, wisdom)
                acknowledge = (
                    0.4 * curiosity
                    + 0.25 * novelty
                    + 0.2 * defer
                    + 0.15 * conf
                    + 0.1 * severity * (1.0 - risk_appetite)
                )
                # Prefer acknowledge over pure reject when curious + loaded.
                if reject > accept and curiosity >= 0.45:
                    acknowledge = max(acknowledge, reject + 0.05)

        eye_match = _clamp(float(eye_match_score), 0.0, 1.0)
        acknowledge_blocked = eye_match > float(cfg.acknowledge_eye_match_max)
        if acknowledge_blocked:
            acknowledge = 0.0

        # Survival pressure: high survival → prefer ignore/reject of playful risks.
        if survival > 0.55:
            reject += 0.25 * severity
            accept *= 0.7
            acknowledge *= 0.85

        scores = {
            "accept": float(accept),
            "reject": float(reject),
            "ignore": float(ignore),
            "reconsider": float(reconsider),
            "acknowledge": float(acknowledge),
            "severity": severity,
            "inconsistent": float(inconsistent),
            "novelty": float(novelty),
            "load": float(load),
            "eye_match": float(eye_match),
            "acknowledge_blocked": float(acknowledge_blocked),
        }

        # Pick best among available decisions.
        ranked = sorted(
            (
                ("accept", accept),
                ("reject", reject),
                ("ignore", ignore),
                ("reconsider", reconsider),
                ("acknowledge", acknowledge),
            ),
            key=lambda kv: kv[1],
            reverse=True,
        )
        decision: Decision = ranked[0][0]  # type: ignore[assignment]
        if decision == "reconsider" and reconsider_used >= int(cfg.reconsider_budget):
            decision = "ignore"
            reason = "reconsider_budget_exhausted"
        elif decision == "accept" and accept < float(cfg.accept_threshold):
            decision = "ignore"
            reason = "accept_below_threshold"
        elif decision == "acknowledge" and acknowledge_blocked:
            decision = "reject" if reject >= ignore else "ignore"
            reason = "acknowledge_eye_match"
        elif decision == "acknowledge" and acknowledge < float(cfg.acknowledge_threshold):
            decision = "reject" if reject >= ignore else "ignore"
            reason = "acknowledge_below_threshold"
        elif decision == "reject" and reject < float(cfg.reject_threshold) and ignore >= reject:
            decision = "ignore"
            reason = "weak_reject_to_ignore"
        else:
            reason = f"scores_{decision}"

        blend = 0.0
        if decision == "accept":
            blend = _clamp(
                float(cfg.base_blend_weight)
                + 0.5 * conf * (0.5 + 0.5 * curiosity),
                0.0,
                float(cfg.max_blend_weight),
            )

        return JudgementResult(
            decision=decision,
            reason=reason,
            blend_weight=blend,
            proposal=proposal,
            traits_snapshot=snap,
            scores=scores,
        )


def apply_proposal(goal: np.ndarray, proposal: DevilProposal) -> np.ndarray:
    """Apply a devil proposal onto an L1 goal vector."""
    g = np.asarray(goal, dtype=np.float32).reshape(-1).copy()
    if g.size < 5:
        padded = np.zeros(5, dtype=np.float32)
        padded[: g.size] = g
        g = padded
    g[0] = float(np.clip(float(g[0]) * float(proposal.forward_scale), -1.0, 1.0))
    g[1] = float(np.clip(float(g[1]) + float(proposal.turn_delta), -1.0, 1.0))
    return g


def blend_goals(
    l1_goal: np.ndarray, proposed: np.ndarray, weight: float
) -> np.ndarray:
    w = _clamp(weight, 0.0, 1.0)
    a = np.asarray(l1_goal, dtype=np.float32).reshape(-1)
    b = np.asarray(proposed, dtype=np.float32).reshape(-1)
    n = max(a.size, b.size, 5)
    aa = np.zeros(n, dtype=np.float32)
    bb = np.zeros(n, dtype=np.float32)
    aa[: a.size] = a
    bb[: b.size] = b
    out = (1.0 - w) * aa + w * bb
    out[0] = float(np.clip(out[0], -1.0, 1.0))
    out[1] = float(np.clip(out[1], -1.0, 1.0))
    return out.astype(np.float32)
