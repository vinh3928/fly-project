"""Devil's-advocate critique via OpenAI-compatible LLM or heuristic mock.

The endpoint is OpenAI-compatible, so a local Ollama server works by pointing
``base_url`` at ``http://127.0.0.1:11434/v1``. Because a critique is requested
on every mind tick, wrap the client in :class:`BackgroundLlmClient` so the sim
loop never blocks on the model.

Use a *small text* model for the devil (e.g. ``llama3.2:3b``). The vision model
used by ``visual_thinking`` is far too heavy for this short critique.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:  # imported lazily at runtime to avoid a circular import
    from flybrain.thinking.judgement import DevilProposal


# Only these scalars go to the devil. Dumping the full live signal payload
# (curiosity, vision cues, etc.) just burns tokens and latency.
_SIGNAL_KEYS = (
    "energy",
    "hunger",
    "sugar",
    "bitter",
    "odor",
    "odor_left",
    "odor_right",
    "wall_hazard",
    "visual_novelty",
    "novelty_bearing",
    "boredom",
    "explore_bias",
    "instability",
    # Interrupt / deep-thought perspective (motor frozen; deliberate on the cue).
    "stimulus_interrupt",
    "deep_thought",
    "interrupt_remaining",
    "interrupt_cue",
    "interrupt_reason",
)
# String signals passed through verbatim (short) instead of coerced to float.
_PASSTHROUGH_STR_KEYS = {"interrupt_cue", "interrupt_reason"}
_AFFECT_KEYS = (
    "survival_risk",
    "fear",
    "novelty",
    "exploration",
    "joy",
    "anger",
    "sad",
    "surprise",
)
# Small models read the key name, not the docs: "bitter" alone reads as a taste
# preference, so name the hazard it actually represents.
_LLM_LABELS = {"bitter": "toxin_taste"}


class LlmClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


def _call(
    client: LlmClient, system: str, user: str, response_format: dict[str, Any] | None
) -> str:
    """Call a client, passing response_format only to clients that accept it."""
    if response_format is None:
        return client.complete(system, user)
    try:
        return client.complete(system, user, response_format)  # type: ignore[call-arg]
    except TypeError:
        return client.complete(system, user)


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
        max_tokens: int = 80,
        temperature: float = 0.3,
        keep_alive: str | None = "10m",
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip(
            "/"
        )
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout_seconds = timeout_seconds
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.keep_alive = keep_alive
        self._http = None

    def _client(self):
        if self._http is None:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover
                raise ImportError("pip install -e '.[mind]' for httpx") from exc
            self._http = httpx.Client(timeout=self.timeout_seconds)
        return self._http

    def complete(
        self, system: str, user: str, response_format: dict[str, Any] | None = None
    ) -> str:
        if not self.api_key:
            return HeuristicDevilClient().complete(system, user)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        # Ollama extension: keep the model in VRAM between mind ticks.
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = self._client().post(
            f"{self.base_url}/chat/completions", headers=headers, json=payload
        )
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"])

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None


class BackgroundLlmClient:
    """Never block the sim loop: return the newest critique, refresh off-thread.

    The first tick (and any tick before the first reply lands) falls back to the
    heuristic devil, so the mind always gets a usable critique. Pass
    ``executor=None`` to run inline for deterministic tests.
    """

    def __init__(
        self,
        inner: LlmClient,
        *,
        fallback: LlmClient | None = None,
        executor: ThreadPoolExecutor | None = "default",  # type: ignore[assignment]
    ) -> None:
        self.inner = inner
        self.fallback = fallback or HeuristicDevilClient()
        self._executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=1) if executor == "default" else executor
        )
        self._future: Future | None = None
        self._last: str = ""
        self.refreshes = 0
        self.errors = 0

    @property
    def is_pending(self) -> bool:
        return self._future is not None and not self._future.done()

    def complete(
        self, system: str, user: str, response_format: dict[str, Any] | None = None
    ) -> str:
        if self._future is not None and self._future.done():
            try:
                self._last = str(self._future.result())
            except Exception:  # noqa: BLE001 - degrade to the last good critique
                self.errors += 1
            self._future = None
        if self._executor is None:
            self._last = str(_call(self.inner, system, user, response_format))
            self.refreshes += 1
        elif self._future is None:
            self._future = self._executor.submit(
                _call, self.inner, system, user, response_format
            )
            self.refreshes += 1
        return self._last or self.fallback.complete(system, user)

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()


def _slim(src: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        if key not in src:
            continue
        val = src[key]
        label = _LLM_LABELS.get(key, key)
        if isinstance(val, bool):
            out[label] = val
        elif isinstance(val, (int, float)):
            out[label] = round(float(val), 3)
        elif val is None:
            continue
        elif key in _PASSTHROUGH_STR_KEYS and isinstance(val, str):
            if val:
                out[label] = val[:40]
        else:
            # Skip arrays / nested blobs — they only slow generation.
            try:
                out[label] = round(float(val), 3)
            except (TypeError, ValueError):
                continue
    return out


_PROPOSAL_SYSTEM = (
    "You are a devil's advocate for a fly flying in a small chamber. "
    "You get the fly's draft plan plus sensor signals (may exceed 1 when senses "
    "are heightened in deep thought): "
    "wall_hazard (proximity to a wall), toxin_taste (toxin nearby), "
    "odor / odor_left / odor_right (antenna plume intensity), "
    "energy (1=full), visual_novelty (something new in view). "
    "When stimulus_interrupt / deep_thought is set the body is already frozen "
    "for interrupt_remaining ticks: do not fight the freeze, instead reason about "
    "interrupt_cue (what stopped the fly, e.g. smell_novelty) and say what to do "
    "when motion resumes. Center your reason on that cue when present. "
    "Name the single biggest risk or cue the draft ignores, and say how to soften it: "
    "forward_scale is the fraction of planned speed to keep, "
    "turn_delta steers toward/away (negative left, positive right), "
    "confidence is how strongly the signals back you up, reason is under 10 words. "
    "Prefer odor when the plume is the strongest new signal. "
    "Do not invent a risk the signals do not show.\n"
    "Examples:\n"
    'in {"signals":{"energy":0.9,"toxin_taste":0.0,"wall_hazard":0.85,"visual_novelty":0.1,"odor":0.0}} '
    'out {"risk":"wall","forward_scale":0.4,"turn_delta":-0.5,"confidence":0.8,'
    '"reason":"wall close ahead"}\n'
    'in {"signals":{"energy":0.9,"toxin_taste":0.7,"wall_hazard":0.1,"visual_novelty":0.2,"odor":0.1}} '
    'out {"risk":"toxin","forward_scale":0.3,"turn_delta":0.6,"confidence":0.7,'
    '"reason":"toxin patch nearby"}\n'
    'in {"signals":{"energy":0.9,"toxin_taste":0.0,"wall_hazard":0.2,"visual_novelty":0.1,"odor":3.7}} '
    'out {"risk":"odor","forward_scale":0.5,"turn_delta":0.2,"confidence":0.8,'
    '"reason":"strong novel odor plume"}\n'
    'in {"signals":{"energy":0.15,"toxin_taste":0.0,"wall_hazard":0.1,"visual_novelty":0.2,"odor":0.0}} '
    'out {"risk":"energy","forward_scale":0.5,"turn_delta":0.0,"confidence":0.6,'
    '"reason":"low energy, conserve"}\n'
    'in {"signals":{"energy":0.8,"toxin_taste":0.0,"wall_hazard":0.05,"visual_novelty":0.8,"odor":0.0}} '
    'out {"risk":"none","forward_scale":1.0,"turn_delta":0.0,"confidence":0.3,'
    '"reason":"no hazard, proceed"}'
)

# Ollama (and OpenAI) structured output: forces valid JSON with the right enum/ranges.
PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "devil_proposal",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "risk": {
                    "type": "string",
                    "enum": ["wall", "toxin", "odor", "fall", "energy", "none"],
                },
                "forward_scale": {"type": "number", "minimum": 0, "maximum": 1},
                "turn_delta": {"type": "number", "minimum": -1, "maximum": 1},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string"},
            },
            "required": [
                "risk",
                "forward_scale",
                "turn_delta",
                "confidence",
                "reason",
            ],
            "additionalProperties": False,
        },
    },
}


def heuristic_proposal(
    *, draft: str, affect: dict[str, Any], signals: dict[str, Any]
) -> DevilProposal:
    """Deterministic structured critique used when mock=true."""
    from flybrain.thinking.judgement import DevilProposal

    wall = float(signals.get("wall_hazard", 0.0) or 0.0)
    bitter = float(signals.get("bitter", 0.0) or 0.0)
    left_odor = float(signals.get("odor_left", 0.0) or 0.0)
    right_odor = float(signals.get("odor_right", 0.0) or 0.0)
    odor = max(float(signals.get("odor", 0.0) or 0.0), left_odor, right_odor)
    survival = float(affect.get("survival_risk", 0.0) or 0.0)
    deep = bool(signals.get("deep_thought") or signals.get("stimulus_interrupt"))
    cue = str(signals.get("interrupt_cue") or "").lower()
    # Conditioning tip: prefer a conditioned active risk over raw wall spam.
    cond_top = str(signals.get("conditioning_top_risk") or "").strip().lower()
    cond_strength = float(signals.get("conditioning_top_strength", 0.0) or 0.0)
    # Draft substrings are noisy (draft always prints "odor=", "bitter=",
    # "L1=explore"), so trigger on numeric senses and the actual interrupt cue,
    # not on free-text keyword matches.
    risk = "none"
    forward_scale = 1.0
    turn_delta = 0.0
    reason = "no major counter"
    # During a smell interrupt the cue itself is the subject, even if the plume
    # reading dips between ticks.
    if cond_top in ("odor", "toxin", "fall", "energy") and cond_strength >= 0.25:
        risk = cond_top
        if cond_top == "odor":
            forward_scale = 0.5
            turn_delta = 0.25 if right_odor >= left_odor else -0.25
            reason = "conditioned odor plume, sample it"
        elif cond_top == "toxin":
            forward_scale = 0.55
            turn_delta = 0.25
            reason = "conditioned toxin risk"
        elif cond_top == "fall":
            forward_scale = 0.4
            reason = "conditioned stability risk"
        else:
            forward_scale = 0.7
            reason = "conditioned energy pressure"
    elif odor > 0.25 or (deep and "smell" in cue):
        risk = "odor"
        forward_scale = 0.5
        turn_delta = 0.25 if right_odor >= left_odor else -0.25
        reason = "novel odor plume, sample it"
    elif wall > 0.55 and not (cond_top == "odor" and cond_strength >= 0.25):
        risk = "wall"
        forward_scale = 0.45
        turn_delta = 0.35 if float(signals.get("novelty_bearing", 0.0) or 0.0) >= 0 else -0.35
        reason = "wall hazard ahead slow and turn"
    elif bitter > 0.3:
        risk = "toxin"
        forward_scale = 0.55
        turn_delta = 0.25
        reason = "toxin risk sample before commit"
    elif deep and "visual" in cue:
        risk = "none"
        forward_scale = 0.6
        reason = "inspect novel sight before moving"
    elif survival > 0.6:
        risk = "fall"
        forward_scale = 0.4
        reason = "stability risk ease forward"
    elif survival > 0.35:
        risk = "energy"
        forward_scale = 0.7
        reason = "energy pressure conserve motion"
    # Frame the counter as interrupt deliberation so the dialog owns the freeze.
    if deep:
        tag = cue or "novelty"
        reason = f"interrupt {tag}: {reason}"[:80]
    conf = 0.55
    if risk != "none" or deep:
        conf = 0.65 + 0.2 * min(1.0, max(wall, bitter, odor / 10.0 if odor > 1.0 else odor, survival))
    return DevilProposal(
        risk=risk,
        forward_scale=float(np.clip(forward_scale, 0.0, 1.0)),
        turn_delta=float(np.clip(turn_delta, -1.0, 1.0)),
        confidence=float(np.clip(conf, 0.0, 1.0)),
        reason=reason,
        ran=True,
        source="heuristic",
    )


def _repair_json(blob: str) -> str:
    """Best-effort cleanup of small-model JSON slips (unquoted keys, stray commas)."""
    repaired = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', blob)
    repaired = re.sub(r"'([^'\"]*)'", r'"\1"', repaired)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return re.sub(r",\s*$", "", repaired.strip())


def _json_candidates(blob: str) -> list[str]:
    """Progressively more aggressive repairs, cheapest first."""
    repaired = _repair_json(blob)
    candidates = [blob, repaired, repaired + "}"]
    # Truncated mid-value: drop trailing fragments until the object closes.
    trimmed = repaired
    for _ in range(4):
        cut = trimmed.rfind(",")
        if cut <= 0:
            break
        trimmed = trimmed[:cut]
        candidates.append(trimmed + "}")
    return candidates


def parse_proposal(text: str) -> DevilProposal:
    from flybrain.thinking.judgement import DevilProposal

    raw = text or ""
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    blob = match.group(0) if match else raw[raw.find("{") :] if "{" in raw else ""
    data: dict[str, Any] | None = None
    for candidate in _json_candidates(blob):
        if not candidate.strip():
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            data = parsed
            break
    if data is None:
        return DevilProposal(ran=True, source="unparsable", reason=raw[:80], confidence=0.0)
    risk = str(data.get("risk", "none")).strip().lower()
    if risk not in ("wall", "toxin", "odor", "fall", "energy", "none"):
        risk = "none"
    return DevilProposal(
        risk=risk,
        forward_scale=float(np.clip(_as_float(data.get("forward_scale"), 1.0), 0.0, 1.0)),
        turn_delta=float(np.clip(_as_float(data.get("turn_delta"), 0.0), -1.0, 1.0)),
        confidence=float(np.clip(_as_float(data.get("confidence"), 0.0), 0.0, 1.0)),
        reason=str(data.get("reason", ""))[:80],
        ran=True,
        source="llm",
    )


def _as_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return default if out != out else out  # NaN guard


def propose_critique(
    client: LlmClient,
    *,
    draft: str,
    affect: dict[str, Any],
    signals: dict[str, Any],
    prior_reason: str | None = None,
) -> DevilProposal:
    """Ask the devil for a structured motor proposal (or heuristic when mocked)."""
    if isinstance(client, HeuristicDevilClient):
        return heuristic_proposal(draft=draft, affect=affect, signals=signals)

    payload = {
        "draft": draft,
        "affect": _slim(affect, _AFFECT_KEYS),
        "signals": _slim(signals, _SIGNAL_KEYS),
    }
    if prior_reason:
        payload["reconsider"] = prior_reason
    user = json.dumps(payload, separators=(",", ":"))
    try:
        text = _call(client, _PROPOSAL_SYSTEM, user, PROPOSAL_SCHEMA)
    except Exception:  # noqa: BLE001
        return heuristic_proposal(draft=draft, affect=affect, signals=signals)
    # Background client may still return heuristic text before first LLM reply.
    if text.strip().startswith("{") or '"risk"' in text or "'risk'" in text:
        return parse_proposal(text)
    # Fallback: map free-text heuristic reply into a proposal.
    return heuristic_proposal(draft=draft + " " + text, affect=affect, signals=signals)


def critique_draft(
    client: LlmClient,
    *,
    draft: str,
    affect: dict[str, Any],
    signals: dict[str, Any],
) -> str:
    """Backward-compatible text critique; prefers structured proposal reason."""
    proposal = propose_critique(client, draft=draft, affect=affect, signals=signals)
    if proposal.reason:
        return f"Counter: {proposal.reason} (risk={proposal.risk}, fwd×{proposal.forward_scale:.2f})."
    system = (
        "You are a devil's advocate for an embodied fly. "
        "In 1-2 short sentences, counter the draft for wall collision, toxins, "
        "falling, or wasted energy. No preamble."
    )
    user = json.dumps(
        {
            "draft": draft,
            "affect": _slim(affect, _AFFECT_KEYS),
            "signals": _slim(signals, _SIGNAL_KEYS),
        },
        separators=(",", ":"),
    )
    return client.complete(system, user)


def make_llm_client(
    *, mock: bool = True, background: bool = False, **kwargs: Any
) -> LlmClient:
    if mock or not (kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY")):
        return HeuristicDevilClient()
    # Drop non-client kwargs that LiveMindConfig may forward.
    known = {
        "api_key",
        "base_url",
        "model",
        "timeout_seconds",
        "max_tokens",
        "temperature",
        "keep_alive",
    }
    client_kwargs = {k: v for k, v in kwargs.items() if k in known}
    client: LlmClient = OpenAiCompatibleClient(**client_kwargs)
    if background:
        client = BackgroundLlmClient(client)
    return client
