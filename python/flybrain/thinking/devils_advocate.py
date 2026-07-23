"""Devil's-advocate critique via OpenAI-compatible LLM or heuristic mock."""

from __future__ import annotations

import json
import os
from typing import Any, Protocol


class LlmClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class HeuristicDevilClient:
    """Deterministic critique used when mock=true or no API key."""

    def complete(self, system: str, user: str) -> str:
        _ = system
        lowered = user.lower()
        critiques: list[str] = []
        if "survival" in lowered or "energy" in lowered:
            critiques.append("Counter: survival risk may be overestimated; freezing wastes foraging time.")
        if "sugar" in lowered or "joy" in lowered:
            critiques.append("Counter: sugar approach could ignore bitter nearby; sample before committing.")
        if "forward" in lowered or "explore" in lowered:
            critiques.append("Counter: forward drive without upright stability risks falling.")
        if "altruism" in lowered:
            critiques.append("Counter: altruism is premature for a solitary foraging fly.")
        if not critiques:
            critiques.append("Counter: the draft lacks a concrete risk check against toxins and falls.")
        return " ".join(critiques)


class OpenAiCompatibleClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip(
            "/"
        )
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout_seconds = timeout_seconds

    def complete(self, system: str, user: str) -> str:
        if not self.api_key:
            return HeuristicDevilClient().complete(system, user)
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install -e '.[mind]' for httpx") from exc

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.4,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        return str(data["choices"][0]["message"]["content"])


def critique_draft(
    client: LlmClient,
    *,
    draft: str,
    affect: dict[str, Any],
    signals: dict[str, Any],
) -> str:
    system = (
        "You are a devil's advocate for an embodied fly agent. "
        "Critique the draft thought for survival risk, toxin exposure, falling, "
        "and unjustified altruism. Be concise (2-4 sentences)."
    )
    user = json.dumps({"draft": draft, "affect": affect, "signals": signals}, indent=2)
    return client.complete(system, user)


def make_llm_client(*, mock: bool = True, **kwargs: Any) -> LlmClient:
    if mock or not (kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY")):
        return HeuristicDevilClient()
    return OpenAiCompatibleClient(**kwargs)
