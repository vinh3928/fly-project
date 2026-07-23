"""Placeholder present/future reasoning until the world-model phase."""

from __future__ import annotations

from typing import Any


def present_future_stub(
    *,
    signals: dict[str, Any],
    affect: dict[str, Any],
) -> dict[str, str]:
    energy = float(signals.get("energy", 0.0))
    bitter = float(signals.get("bitter", 0.0))
    sugar = float(signals.get("sugar", 0.0))
    survival = float(affect.get("survival_risk", 0.0))
    exploration = float(affect.get("exploration", 0.0))

    present = (
        f"Present: energy={energy:.2f}, sugar={sugar:.2f}, bitter={bitter:.2f}, "
        f"survival_risk={survival:.2f}."
    )
    if survival > 0.6:
        future = "Future hypothesis: seek energy / avoid toxins; reduce forward drive."
    elif sugar > 0.3:
        future = "Future hypothesis: approach sugar source; extend proboscis if contact likely."
    elif exploration > 0.5:
        future = "Future hypothesis: explore novel directions with mild forward motion."
    else:
        future = "Future hypothesis: maintain upright stance and sample local chemicals."
    return {"present": present, "future_hypothesis": future}
