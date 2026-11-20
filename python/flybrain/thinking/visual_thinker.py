"""Novelty-triggered visual thinking via a local (Ollama) vision LLM.

When the eyes report high visual novelty *during low cognitive load*, the fly
"stops to think": it orients toward the stimulus and sends the current eye
frames to a vision-capable LLM (default: a local Ollama model such as
``qwen2.5vl:7b``) instead of OpenAI. The reply is turned into a short thought
plus an approach / inspect / ignore bias for the motor goal.

The LLM call is slow (seconds), so it runs off the hot path in a background
thread; the motor keeps ticking while the fly is "inspecting". Gating and
motor influence live here so they can be unit-tested without a network.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from flybrain.vision.novelty import VisualNoveltyConfig, apply_visual_explore_goal


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


@dataclass
class VisualThinkerConfig:
    """Config for the novelty-triggered vision-LLM thinking mode."""

    enabled: bool = False
    # OpenAI-compatible endpoint. Default points at a local Ollama server.
    base_url: str = "http://127.0.0.1:11434/v1"
    model: str = "qwen2.5vl:7b"
    # Ollama ignores the key, but the OpenAI-style client still sends one.
    api_key: str = "ollama"
    timeout_seconds: float = 30.0
    # When true, skip the network and infer a heuristic insight from novelty
    # cues (used by tests and when no local server is running).
    mock: bool = True

    # --- Gating: only "stop to think" when it is safe and worthwhile. ---
    min_visual_novelty: float = 0.45
    require_new_stimulus: bool = False
    max_wall_hazard: float = 0.6
    max_survival_risk: float = 0.5
    cooldown_ticks: int = 40

    # --- Frame prep for the LLM. ---
    image_size: int = 128

    # --- Motor influence. ---
    # While a thought is pending, damp forward motion and orient (stop & look).
    inspect_forward_scale: float = 0.2
    inspect_turn_gain: float = 0.6
    # How many mind ticks an "approach" insight keeps pulling the goal.
    insight_ttl_ticks: int = 40

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> VisualThinkerConfig:
        data = dict(raw or {})
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class VisualInsight:
    """Structured result of one vision-LLM inspection."""

    ran: bool
    reason: str = ""
    description: str = ""
    salience: float = 0.0
    decision: str = "ignore"  # approach | inspect | ignore
    bearing: float = 0.0
    tick: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "ran": bool(self.ran),
            "reason": self.reason,
            "description": self.description,
            "salience": float(self.salience),
            "decision": self.decision,
            "bearing": float(self.bearing),
            "tick": int(self.tick),
        }


class VisualLlmClient(Protocol):
    def inspect(self, *, system: str, prompt: str, images: list[np.ndarray]) -> str: ...


def _frame_to_png_b64(frame: np.ndarray, size: int) -> str:
    """Upscale an eye frame to ``size`` and encode as a base64 PNG data URL."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - only on real LLM path
        raise ImportError("pip install pillow for the vision-LLM path") from exc

    arr = np.asarray(frame, dtype=np.float32)
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    arr = np.clip(arr, 0.0, 1.0)
    img = Image.fromarray((arr * 255.0).astype(np.uint8))
    img = img.convert("RGB").resize((int(size), int(size)))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class OllamaVisionClient:
    """OpenAI-compatible multimodal client (Ollama /v1/chat/completions)."""

    def __init__(self, config: VisualThinkerConfig) -> None:
        self.config = config
        self.base_url = (
            config.base_url or os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
        ).rstrip("/")
        self.model = config.model or os.environ.get("OLLAMA_MODEL", "qwen2.5vl:7b")
        self.api_key = config.api_key or os.environ.get("OLLAMA_API_KEY", "ollama")

    def inspect(self, *, system: str, prompt: str, images: list[np.ndarray]) -> str:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install httpx for the vision-LLM path") from exc

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for frame in images:
            url = _frame_to_png_b64(frame, self.config.image_size)
            content.append({"type": "image_url", "image_url": {"url": url}})
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": 0.2,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.config.timeout_seconds) as client:
            response = client.post(
                f"{self.base_url}/chat/completions", headers=headers, json=payload
            )
            response.raise_for_status()
            data = response.json()
        return str(data["choices"][0]["message"]["content"])


def make_visual_client(config: VisualThinkerConfig) -> VisualLlmClient | None:
    """Return a live client, or ``None`` to use the offline heuristic."""
    if config.mock:
        return None
    return OllamaVisionClient(config)


_SYSTEM_PROMPT = (
    "You are the visual cortex of an embodied fruit fly exploring a small "
    "chamber. You are shown the left and right eye frames. A novel stimulus "
    "just entered view. Decide whether it is worth approaching. Reply with a "
    "single compact JSON object and nothing else, with keys: "
    '"description" (<=12 words), "decision" (one of "approach", "inspect", '
    '"ignore"), "salience" (0..1 how interesting), "bearing" (-1 left .. '
    "+1 right, where the stimulus is)."
)


def _parse_insight(text: str, *, fallback_bearing: float, tick: int) -> VisualInsight:
    """Extract the JSON object from an LLM reply into a VisualInsight."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return VisualInsight(
            ran=True, reason="unparsable", description=text.strip()[:120], tick=tick
        )
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return VisualInsight(
            ran=True, reason="unparsable", description=text.strip()[:120], tick=tick
        )
    decision = str(data.get("decision", "ignore")).strip().lower()
    if decision not in ("approach", "inspect", "ignore"):
        decision = "inspect"
    return VisualInsight(
        ran=True,
        reason="llm",
        description=str(data.get("description", ""))[:200],
        salience=_clamp(float(data.get("salience", 0.0) or 0.0), 0.0, 1.0),
        decision=decision,
        bearing=_clamp(float(data.get("bearing", fallback_bearing) or 0.0), -1.0, 1.0),
        tick=tick,
    )


@dataclass
class _Job:
    tick: int
    novelty: float
    bearing: float
    left: np.ndarray
    right: np.ndarray


class VisualThinker:
    """Gate → (background) vision-LLM inspection → motor goal bias.

    Pass ``executor=None`` to run inspections inline (deterministic tests).
    """

    def __init__(
        self,
        config: VisualThinkerConfig | None = None,
        *,
        client: VisualLlmClient | None = None,
        executor: ThreadPoolExecutor | None = "default",  # type: ignore[assignment]
        novelty_config: VisualNoveltyConfig | None = None,
    ) -> None:
        self.config = config or VisualThinkerConfig()
        self.client = client if client is not None else make_visual_client(self.config)
        if executor == "default":
            self._executor: ThreadPoolExecutor | None = (
                ThreadPoolExecutor(max_workers=1) if self.config.enabled else None
            )
        else:
            self._executor = executor
        self._novelty_config = novelty_config or VisualNoveltyConfig()
        self._future: Future | None = None
        self._pending_bearing: float = 0.0
        self._last_insight: VisualInsight | None = None
        self._last_tick: int = -(10**9)
        self.inspections = 0

    def reset(self) -> None:
        self._future = None
        self._pending_bearing = 0.0
        self._last_insight = None
        self._last_tick = -(10**9)
        self.inspections = 0

    @property
    def is_pending(self) -> bool:
        return self._future is not None and not self._future.done()

    @property
    def last_insight(self) -> VisualInsight | None:
        return self._last_insight

    def _gate(
        self,
        *,
        visual_novelty: float,
        new_stimulus: bool,
        wall_hazard: float,
        survival_risk: float,
        tick: int,
    ) -> str:
        cfg = self.config
        if not cfg.enabled:
            return "disabled"
        if self.is_pending:
            return "pending"
        if tick - self._last_tick < int(cfg.cooldown_ticks):
            return "cooldown"
        if visual_novelty < float(cfg.min_visual_novelty):
            return "low_novelty"
        if cfg.require_new_stimulus and not new_stimulus:
            return "not_new"
        if wall_hazard > float(cfg.max_wall_hazard):
            return "wall"
        if survival_risk > float(cfg.max_survival_risk):
            return "danger"
        return "go"

    def _infer(self, job: _Job) -> VisualInsight:
        if self.client is None:
            # Offline heuristic: approach a salient, well-localized stimulus.
            decision = "approach" if job.novelty >= 0.6 else "inspect"
            return VisualInsight(
                ran=True,
                reason="heuristic",
                description="novel stimulus in view",
                salience=_clamp(job.novelty, 0.0, 1.0),
                decision=decision,
                bearing=_clamp(job.bearing, -1.0, 1.0),
                tick=job.tick,
            )
        prompt = (
            "A novel stimulus appeared roughly at bearing "
            f"{job.bearing:+.2f} (-1 left .. +1 right). Should the fly approach it?"
        )
        try:
            text = self.client.inspect(
                system=_SYSTEM_PROMPT, prompt=prompt, images=[job.left, job.right]
            )
        except Exception as exc:  # noqa: BLE001 - degrade gracefully off the hot path
            return VisualInsight(
                ran=True, reason=f"error:{type(exc).__name__}", tick=job.tick
            )
        return _parse_insight(text, fallback_bearing=job.bearing, tick=job.tick)

    def maybe_think(
        self,
        *,
        tick: int,
        visual_novelty: float,
        new_stimulus: bool,
        wall_hazard: float,
        survival_risk: float,
        left_eye: np.ndarray | None,
        right_eye: np.ndarray | None,
        bearing: float,
    ) -> str:
        """Submit an inspection if the gate opens. Returns the gate decision."""
        if left_eye is None or right_eye is None:
            return "no_eyes"
        gate = self._gate(
            visual_novelty=visual_novelty,
            new_stimulus=new_stimulus,
            wall_hazard=wall_hazard,
            survival_risk=survival_risk,
            tick=tick,
        )
        if gate != "go":
            return gate
        job = _Job(
            tick=tick,
            novelty=float(visual_novelty),
            bearing=float(bearing),
            left=np.asarray(left_eye, dtype=np.float32).copy(),
            right=np.asarray(right_eye, dtype=np.float32).copy(),
        )
        self._last_tick = tick
        self._pending_bearing = float(bearing)
        self.inspections += 1
        if self._executor is None:
            self._last_insight = self._infer(job)
            self._future = None
        else:
            self._future = self._executor.submit(self._infer, job)
        return "go"

    def poll(self) -> VisualInsight | None:
        """Collect a finished background inspection, if any."""
        if self._future is not None and self._future.done():
            try:
                self._last_insight = self._future.result()
            except Exception as exc:  # noqa: BLE001
                self._last_insight = VisualInsight(ran=True, reason=f"error:{type(exc).__name__}")
            self._future = None
            return self._last_insight
        return None

    def bias_goal(
        self, goal: np.ndarray, *, tick: int, wall_hazard: float = 0.0
    ) -> np.ndarray:
        """Apply the current thinking state to the motor goal.

        While inspecting: damp forward motion and orient toward the stimulus.
        With a fresh "approach"/"inspect" insight: pull the goal toward it.
        """
        g = np.asarray(goal, dtype=np.float32).reshape(-1).copy()
        if g.size < 5:
            padded = np.zeros(5, dtype=np.float32)
            padded[: g.size] = g
            g = padded
        cfg = self.config
        if not cfg.enabled:
            return g
        if self.is_pending:
            g[0] = float(g[0]) * float(cfg.inspect_forward_scale)
            g[1] = _clamp(
                0.5 * float(g[1]) + 0.5 * self._pending_bearing * float(cfg.inspect_turn_gain),
                -1.0,
                1.0,
            )
            return g
        insight = self._last_insight
        if (
            insight is not None
            and insight.ran
            and insight.decision in ("approach", "inspect")
            and insight.tick >= 0
            and tick - insight.tick <= int(cfg.insight_ttl_ticks)
        ):
            forward_scale = 1.0 if insight.decision == "approach" else 0.5
            cues = {
                "visual_novelty": max(insight.salience, float(cfg.min_visual_novelty)),
                "novelty_bearing": insight.bearing,
                "novelty_confidence": insight.salience,
            }
            g = apply_visual_explore_goal(
                g, cues, config=self._novelty_config, wall_hazard=wall_hazard
            )
            g[0] = float(g[0]) * forward_scale
        return g

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
