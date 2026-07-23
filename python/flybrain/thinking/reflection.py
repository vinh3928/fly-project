"""Build reflection context from similar + recent STM records."""

from __future__ import annotations

from flybrain.memory.pgvector import ShortTermMemory, StmRecord


def build_reflection(
    memory: ShortTermMemory,
    *,
    embedding,
    fly_id: str,
    top_k: int = 5,
    recent_k: int = 8,
) -> tuple[str, list[StmRecord], list[StmRecord]]:
    similar = memory.similar(embedding, fly_id=fly_id, top_k=top_k)
    recent = memory.recent(fly_id=fly_id, limit=recent_k)

    lines = ["## Historic reflection"]
    if not similar and not recent:
        lines.append("No prior short-term memories.")
    else:
        lines.append(f"Similar memories ({len(similar)}):")
        for rec in similar:
            affect = rec.affect
            lines.append(
                f"- step={rec.step} joy={affect.get('joy', 0):.2f} "
                f"fear={affect.get('fear', 0):.2f} survival={affect.get('survival_risk', 0):.2f} "
                f"sugar={rec.signals.get('sugar', 0)}"
            )
        lines.append(f"Recent memories ({len(recent)}):")
        for rec in recent:
            lines.append(
                f"- step={rec.step} t={rec.sim_time:.3f} "
                f"anger={rec.affect.get('anger', 0):.2f} "
                f"exploration={rec.affect.get('exploration', 0):.2f}"
            )
    return "\n".join(lines), similar, recent
