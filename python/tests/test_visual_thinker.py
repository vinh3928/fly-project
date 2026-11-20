"""Novelty-triggered vision-LLM thinking mode."""

from __future__ import annotations

from concurrent.futures import Future

import numpy as np

from flybrain.thinking.visual_thinker import (
    OllamaVisionClient,
    VisualInsight,
    VisualThinker,
    VisualThinkerConfig,
    _parse_insight,
    make_visual_client,
)


def _eyes(size: int = 16) -> tuple[np.ndarray, np.ndarray]:
    left = np.full((size, size), 0.3, dtype=np.float32)
    right = np.full((size, size), 0.3, dtype=np.float32)
    return left, right


class _FakeClient:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0
        self.last_images: list[np.ndarray] = []

    def inspect(self, *, system: str, prompt: str, images: list[np.ndarray]) -> str:
        self.calls += 1
        self.last_images = images
        return self.reply


class _BoomClient:
    def inspect(self, *, system: str, prompt: str, images: list[np.ndarray]) -> str:
        raise RuntimeError("network down")


def _thinker(config: VisualThinkerConfig, client=None) -> VisualThinker:
    # executor=None → run inspections inline for deterministic tests.
    return VisualThinker(config, client=client, executor=None)


def test_disabled_never_thinks() -> None:
    thinker = _thinker(VisualThinkerConfig(enabled=False))
    left, right = _eyes()
    gate = thinker.maybe_think(
        tick=0,
        visual_novelty=0.9,
        new_stimulus=True,
        wall_hazard=0.0,
        survival_risk=0.0,
        left_eye=left,
        right_eye=right,
        bearing=0.5,
    )
    assert gate == "disabled"
    assert thinker.last_insight is None


def test_gate_blocks_low_novelty_wall_and_danger() -> None:
    left, right = _eyes()
    base = dict(left_eye=left, right_eye=right, new_stimulus=True, bearing=0.4)
    low = _thinker(VisualThinkerConfig(enabled=True, mock=True, min_visual_novelty=0.5))
    assert (
        low.maybe_think(tick=0, visual_novelty=0.2, wall_hazard=0.0, survival_risk=0.0, **base)
        == "low_novelty"
    )

    wall = _thinker(VisualThinkerConfig(enabled=True, mock=True, max_wall_hazard=0.5))
    assert (
        wall.maybe_think(tick=0, visual_novelty=0.9, wall_hazard=0.8, survival_risk=0.0, **base)
        == "wall"
    )

    danger = _thinker(VisualThinkerConfig(enabled=True, mock=True, max_survival_risk=0.5))
    assert (
        danger.maybe_think(tick=0, visual_novelty=0.9, wall_hazard=0.0, survival_risk=0.9, **base)
        == "danger"
    )


def test_cooldown_prevents_back_to_back_inspections() -> None:
    thinker = _thinker(
        VisualThinkerConfig(enabled=True, mock=True, cooldown_ticks=20, min_visual_novelty=0.4)
    )
    left, right = _eyes()
    kwargs = dict(
        visual_novelty=0.9,
        new_stimulus=True,
        wall_hazard=0.0,
        survival_risk=0.0,
        left_eye=left,
        right_eye=right,
        bearing=0.3,
    )
    assert thinker.maybe_think(tick=0, **kwargs) == "go"
    assert thinker.maybe_think(tick=5, **kwargs) == "cooldown"
    assert thinker.maybe_think(tick=25, **kwargs) == "go"
    assert thinker.inspections == 2


def test_offline_heuristic_approaches_high_novelty() -> None:
    thinker = _thinker(VisualThinkerConfig(enabled=True, mock=True, min_visual_novelty=0.4))
    left, right = _eyes()
    gate = thinker.maybe_think(
        tick=0,
        visual_novelty=0.8,
        new_stimulus=True,
        wall_hazard=0.0,
        survival_risk=0.0,
        left_eye=left,
        right_eye=right,
        bearing=-0.6,
    )
    assert gate == "go"
    insight = thinker.last_insight
    assert insight is not None and insight.ran
    assert insight.decision == "approach"
    assert insight.bearing < 0.0


def test_llm_client_reply_parsed_and_biases_goal() -> None:
    reply = 'Here you go: {"description": "a dark pillar", "decision": "approach", "salience": 0.8, "bearing": -0.7}'
    client = _FakeClient(reply)
    thinker = _thinker(
        VisualThinkerConfig(enabled=True, mock=False, min_visual_novelty=0.4), client=client
    )
    left, right = _eyes()
    gate = thinker.maybe_think(
        tick=3,
        visual_novelty=0.7,
        new_stimulus=True,
        wall_hazard=0.1,
        survival_risk=0.0,
        left_eye=left,
        right_eye=right,
        bearing=-0.7,
    )
    assert gate == "go"
    assert client.calls == 1
    assert len(client.last_images) == 2
    insight = thinker.last_insight
    assert insight is not None and insight.decision == "approach"

    goal = np.asarray([0.2, 0.0, 0.5, 0.5, 0.0], dtype=np.float32)
    biased = thinker.bias_goal(goal, tick=4, wall_hazard=0.1)
    # Turn toward the negative (left) bearing reported by the LLM.
    assert float(biased[1]) < -0.1


def test_client_error_degrades_gracefully() -> None:
    thinker = _thinker(
        VisualThinkerConfig(enabled=True, mock=False, min_visual_novelty=0.4), client=_BoomClient()
    )
    left, right = _eyes()
    gate = thinker.maybe_think(
        tick=0,
        visual_novelty=0.9,
        new_stimulus=True,
        wall_hazard=0.0,
        survival_risk=0.0,
        left_eye=left,
        right_eye=right,
        bearing=0.2,
    )
    assert gate == "go"
    insight = thinker.last_insight
    assert insight is not None and insight.reason.startswith("error")
    assert insight.decision == "ignore"


def test_pending_state_damps_forward_and_orients() -> None:
    thinker = _thinker(VisualThinkerConfig(enabled=True, mock=True, inspect_forward_scale=0.1))
    # Simulate an in-flight background inspection toward the right.
    thinker._future = Future()
    thinker._pending_bearing = 0.8
    goal = np.asarray([1.0, 0.0, 0.5, 0.5, 0.0], dtype=np.float32)
    biased = thinker.bias_goal(goal, tick=1, wall_hazard=0.0)
    assert float(biased[0]) < 0.2  # forward damped while thinking
    assert float(biased[1]) > 0.1  # oriented toward the pending bearing


def test_insight_expires_after_ttl() -> None:
    thinker = _thinker(VisualThinkerConfig(enabled=True, mock=True, insight_ttl_ticks=5))
    thinker._last_insight = VisualInsight(
        ran=True, reason="llm", decision="approach", salience=0.9, bearing=-0.6, tick=0
    )
    goal = np.asarray([0.2, 0.0, 0.5, 0.5, 0.0], dtype=np.float32)
    fresh = thinker.bias_goal(goal, tick=3, wall_hazard=0.0)
    assert float(fresh[1]) < -0.05
    stale = thinker.bias_goal(goal, tick=100, wall_hazard=0.0)
    assert float(stale[1]) == 0.0  # expired → no bias


def test_parse_insight_handles_noise_and_bad_json() -> None:
    good = _parse_insight(
        'noise {"decision":"inspect","salience":0.5,"bearing":0.3} tail',
        fallback_bearing=0.0,
        tick=1,
    )
    assert good.decision == "inspect" and abs(good.bearing - 0.3) < 1e-6
    bad = _parse_insight("no json here", fallback_bearing=0.25, tick=2)
    assert bad.reason == "unparsable" and bad.ran


def test_make_visual_client_respects_mock_flag() -> None:
    assert make_visual_client(VisualThinkerConfig(mock=True)) is None
    live = make_visual_client(VisualThinkerConfig(mock=False))
    assert isinstance(live, OllamaVisionClient)


def test_config_from_dict_ignores_unknown_keys() -> None:
    cfg = VisualThinkerConfig.from_dict({"enabled": True, "model": "moondream", "bogus": 1})
    assert cfg.enabled is True and cfg.model == "moondream"
