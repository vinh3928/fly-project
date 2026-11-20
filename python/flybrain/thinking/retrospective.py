"""Low-load retrospective: label past transitions and investigate deferred interest."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from flybrain.memory.pgvector import ShortTermMemory, StmRecord, ThoughtRecord
from flybrain.thinking.interest import (
    InterestPointStore,
    InvestigationDirective,
)
from flybrain.thinking.two_layer_future import action_name_from_vector
from flybrain.world_model.outcomes import LifeExperience, OutcomeLabel, label_transition
from flybrain.world_model.predictor import NextStateWorldModel


@dataclass
class RetrospectiveConfig:
    survival_risk_max: float = 0.45
    bitter_max: float = 0.35
    wall_hazard_max: float = 0.45
    batch_size: int = 8
    wm_train_steps: int = 1
    wm_lr: float = 1e-3
    investigate_interest: bool = True


@dataclass
class LabeledOutcome:
    stm_id: int
    action_name: str
    label: OutcomeLabel
    embedding: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        d = self.label.to_dict()
        d["stm_id"] = self.stm_id
        d["action_name"] = self.action_name
        return d


@dataclass
class RetrospectiveResult:
    ran: bool
    reason: str
    labeled: list[LabeledOutcome] = field(default_factory=list)
    thought: ThoughtRecord | None = None
    wm_loss: float | None = None
    investigations: list[InvestigationDirective] = field(default_factory=list)


def is_low_load(
    *,
    signals: dict[str, Any],
    affect: dict[str, Any],
    config: RetrospectiveConfig | None = None,
) -> bool:
    cfg = config or RetrospectiveConfig()
    survival = float(affect.get("survival_risk", 0.0))
    bitter = float(signals.get("bitter", 0.0))
    wall = float(signals.get("wall_hazard", 0.0))
    return (
        survival <= cfg.survival_risk_max
        and bitter <= cfg.bitter_max
        and wall <= cfg.wall_hazard_max
    )


def can_investigate_interest(
    *,
    signals: dict[str, Any],
    affect: dict[str, Any],
    config: RetrospectiveConfig | None = None,
) -> bool:
    """Softer than full low-load: survival/toxin calm is enough to revisit interest.

    Wall hazard alone should not block investigating a deferred wall/novelty point —
    the investigation directive itself slows and turns.
    """
    cfg = config or RetrospectiveConfig()
    survival = float(affect.get("survival_risk", 0.0))
    bitter = float(signals.get("bitter", 0.0))
    return survival <= cfg.survival_risk_max and bitter <= cfg.bitter_max


class OutcomeMemory:
    """In-process store of labeled transitions for Layer-2 retrieval."""

    def __init__(self, max_rows: int = 2000) -> None:
        self.max_rows = max_rows
        self._rows: list[dict[str, Any]] = []

    def add(self, row: dict[str, Any]) -> None:
        self._rows.append(dict(row))
        if len(self._rows) > self.max_rows:
            self._rows = self._rows[-self.max_rows :]

    def all(self) -> list[dict[str, Any]]:
        return list(self._rows)

    def similar_by_action(self, action_name: str, *, limit: int = 20) -> list[dict[str, Any]]:
        hits = [r for r in self._rows if r.get("action_name") == action_name]
        return hits[-limit:]


def run_retrospective(
    *,
    memory: ShortTermMemory,
    experience: LifeExperience,
    outcome_memory: OutcomeMemory,
    fly_id: str,
    signals: dict[str, Any],
    affect: dict[str, Any],
    recent_actions: list[dict[str, Any]] | None = None,
    world_model: NextStateWorldModel | None = None,
    config: RetrospectiveConfig | None = None,
    force: bool = False,
    tendency: Any | None = None,
    interest_store: InterestPointStore | None = None,
    tick: int = 0,
) -> RetrospectiveResult:
    """Label recent STM pairs when decision pressure is low; investigate deferred interest."""
    cfg = config or RetrospectiveConfig()
    low = force or is_low_load(signals=signals, affect=affect, config=cfg)
    can_invest = force or can_investigate_interest(signals=signals, affect=affect, config=cfg)

    investigations: list[InvestigationDirective] = []
    if (
        can_invest
        and interest_store is not None
        and cfg.investigate_interest
        and interest_store.config.enabled
    ):
        investigations = interest_store.pop_for_investigation(tick=int(tick))

    if not low and not investigations:
        return RetrospectiveResult(ran=False, reason="high_load")

    recent = memory.recent(fly_id=fly_id, limit=max(cfg.batch_size + 1, 4)) if low else []
    if low and len(recent) < 2 and not investigations:
        return RetrospectiveResult(ran=False, reason="insufficient_stm")

    # Pair consecutive STM records (older → newer).
    ordered = list(reversed(recent)) if recent else []
    # Detect order: if first has higher step than last, reverse
    if ordered and recent and ordered[0].step > ordered[-1].step:
        ordered = list(reversed(ordered))

    actions = list(recent_actions or [])
    labeled: list[LabeledOutcome] = []
    pairs = min(cfg.batch_size, max(0, len(ordered) - 1)) if low else 0
    wm_loss: float | None = None

    for i in range(len(ordered) - 1, len(ordered) - 1 - pairs, -1):
        if i <= 0:
            break
        prev: StmRecord = ordered[i - 1]
        nxt: StmRecord = ordered[i]
        # Skip if already labeled for this stm id
        if any(r.get("stm_id") == nxt.id for r in outcome_memory.all()):
            continue
        action_name = "explore"
        action_vec = None
        for row in reversed(actions):
            if int(row.get("stm_id", -1)) == prev.id or int(row.get("step", -1)) == prev.step:
                action_name = str(row.get("action_name", action_name))
                action_vec = row.get("action")
                break
        if action_vec is None and actions:
            last = actions[-1]
            action_name = str(last.get("action_name", action_name))
            action_vec = last.get("action")
        if action_vec is not None:
            action_name = action_name_from_vector(np.asarray(action_vec, dtype=np.float32))

        label = label_transition(
            prev_signals=prev.signals,
            prev_affect=prev.affect,
            next_signals=nxt.signals,
            next_affect=nxt.affect,
        )
        entry = LabeledOutcome(
            stm_id=nxt.id, action_name=action_name, label=label, embedding=nxt.embedding
        )
        labeled.append(entry)
        outcome_memory.add(entry.to_dict())
        experience.record_label(1)
        if tendency is not None and hasattr(tendency, "update_from_outcome"):
            tendency.update_from_outcome(
                action_name=action_name,
                outcome_score=float(label.score),
                kind=str(label.kind),
            )

        if world_model is not None and action_vec is not None and cfg.wm_train_steps > 0:
            loss = _wm_train_step(
                world_model,
                prev.embedding,
                np.asarray(action_vec, dtype=np.float32),
                nxt.embedding,
                lr=cfg.wm_lr,
            )
            wm_loss = float(loss) if wm_loss is None else 0.5 * (wm_loss + float(loss))

    thought: ThoughtRecord | None = None
    if investigations:
        inv = investigations[0]
        stm_ref = inv.point.stm_id
        if ordered:
            stm_ref = ordered[-1].id
        thought = memory.insert_thought(
            stm_id=stm_ref,
            reflection=(
                f"Low-load investigation of deferred {inv.point.kind} "
                f"(stm={inv.point.stm_id}, hits={inv.point.hits})."
            ),
            devil_critique="n/a",
            refined_thought=(
                f"Investigate interest#{inv.point.id}: {inv.point.reason or inv.point.kind} "
                f"bearing={inv.point.bearing:+.2f}"
            ),
            intent={
                "retrospective": True,
                "investigation": inv.to_dict(),
                "interest_point": inv.point.to_dict(),
                "maturity": experience.maturity(),
            },
        )
    elif labeled:
        summary = "; ".join(
            f"{x.action_name}:{x.label.kind}({x.label.score:.2f})" for x in labeled[:5]
        )
        thought = memory.insert_thought(
            stm_id=labeled[-1].stm_id,
            reflection=f"Retrospective labeled {len(labeled)} transitions.",
            devil_critique="n/a",
            refined_thought=f"Low-load retrospective: {summary}",
            intent={
                "retrospective": True,
                "labeled": [x.to_dict() for x in labeled],
                "maturity": experience.maturity(),
            },
        )
    elif not investigations:
        return RetrospectiveResult(ran=True, reason="nothing_new", labeled=[], wm_loss=wm_loss)

    reason = "ok"
    if investigations and labeled:
        reason = "ok_investigate_and_label"
    elif investigations:
        reason = "ok_investigate"
    return RetrospectiveResult(
        ran=True,
        reason=reason,
        labeled=labeled,
        thought=thought,
        wm_loss=wm_loss,
        investigations=investigations,
    )


def _wm_train_step(
    model: NextStateWorldModel,
    latent: np.ndarray,
    action: np.ndarray,
    next_latent: np.ndarray,
    *,
    lr: float,
) -> float:
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    z = torch.as_tensor(_pad(latent, model.latent_dim), dtype=torch.float32).unsqueeze(0)
    a = torch.as_tensor(_pad(action, model.action_dim), dtype=torch.float32).unsqueeze(0)
    zn = torch.as_tensor(_pad(next_latent, model.latent_dim), dtype=torch.float32).unsqueeze(0)
    opt.zero_grad()
    loss = model.prediction_loss(z, a, zn)
    loss.backward()
    opt.step()
    return float(loss.detach().item())


def _pad(vec: np.ndarray, dim: int) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    out = np.zeros(dim, dtype=np.float32)
    n = min(dim, v.shape[0])
    out[:n] = v[:n]
    return out
