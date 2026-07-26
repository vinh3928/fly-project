"""Devil's advocate wired to a local Ollama (OpenAI-compatible) endpoint."""

from __future__ import annotations

import json
from concurrent.futures import wait

from flybrain.pipeline.live_mind import LiveMindConfig
from flybrain.thinking.devils_advocate import (
    BackgroundLlmClient,
    HeuristicDevilClient,
    OpenAiCompatibleClient,
    critique_draft,
    make_llm_client,
)


class _FakeClient:
    def __init__(self, reply: str = "Counter: the plan ignores the wall ahead.") -> None:
        self.reply = reply
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return self.reply


class _BoomClient:
    def complete(self, system: str, user: str) -> str:
        raise RuntimeError("ollama down")


def test_mock_still_returns_heuristic() -> None:
    client = make_llm_client(mock=True, base_url="http://127.0.0.1:11434/v1")
    assert isinstance(client, HeuristicDevilClient)


def test_ollama_config_builds_openai_compatible_client() -> None:
    client = make_llm_client(
        mock=False,
        base_url="http://127.0.0.1:11434/v1",
        model="qwen2.5vl:7b",
        api_key="ollama",
    )
    assert isinstance(client, OpenAiCompatibleClient)
    assert client.base_url == "http://127.0.0.1:11434/v1"
    assert client.model == "qwen2.5vl:7b"


def test_background_flag_wraps_client() -> None:
    client = make_llm_client(
        mock=False, api_key="ollama", model="qwen2.5vl:7b", background=True
    )
    assert isinstance(client, BackgroundLlmClient)
    client.close()


def test_background_client_inline_returns_inner_reply() -> None:
    inner = _FakeClient()
    client = BackgroundLlmClient(inner, executor=None)
    out = client.complete("sys", "user")
    assert out == inner.reply
    assert inner.calls == 1


def test_background_client_first_tick_falls_back_then_uses_llm() -> None:
    inner = _FakeClient()
    client = BackgroundLlmClient(inner)
    try:
        first = client.complete("sys", "user")
        # Nothing has landed yet, so the heuristic keeps the mind running.
        assert "Counter" in first
        if client._future is not None:
            wait([client._future])
        second = client.complete("sys", "user")
        assert second == inner.reply
    finally:
        client.close()


def test_background_client_survives_llm_errors() -> None:
    client = BackgroundLlmClient(_BoomClient())
    try:
        client.complete("sys", "user")
        if client._future is not None:
            wait([client._future])
        out = client.complete("sys", "user")
        assert "Counter" in out  # fell back to the heuristic devil
        assert client.errors >= 1
    finally:
        client.close()


def test_critique_draft_sends_slim_payload() -> None:
    captured: dict[str, str] = {}

    class Capture:
        def complete(self, system: str, user: str) -> str:
            captured["system"] = system
            captured["user"] = user
            return "Counter: wall ahead."

    critique_draft(
        Capture(),
        draft="Plan explore",
        affect={"survival_risk": 0.2, "fear": 0.4, "extra_blob": {"x": 1}},
        signals={
            "energy": 0.9,
            "wall_hazard": 0.8,
            "bitter": 0.6,
            "odor": 3.5,
            "odor_left": 3.4,
            "odor_right": 3.5,
            "body": [0.0] * 85,
            "curiosity": 0.01,
            "visual_novelty": 0.0,
        },
    )
    payload = json.loads(captured["user"])
    assert "body" not in payload["signals"]
    assert "extra_blob" not in payload["affect"]
    assert payload["signals"]["wall_hazard"] == 0.8
    # "bitter" alone reads as a taste preference to a small model.
    assert "bitter" not in payload["signals"]
    assert payload["signals"]["toxin_taste"] == 0.6
    assert payload["signals"]["odor"] == 3.5
    assert payload["signals"]["odor_left"] == 3.4
    assert "odor" in captured["system"]
    assert "forward_scale" in captured["system"]
    assert "toxin_taste" in captured["system"]
    assert "body" not in captured["user"]


def test_live_mind_config_forwards_llm_settings() -> None:
    cfg = LiveMindConfig.from_yaml(
        {
            "llm": {
                "base_url": "http://127.0.0.1:11434/v1",
                "model": "qwen2.5vl:7b",
                "api_key": "ollama",
                "timeout_seconds": 15,
                "mock": False,
                "background": True,
            }
        }
    )
    assert cfg.llm_mock is False
    assert cfg.llm_kwargs == {
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen2.5vl:7b",
        "api_key": "ollama",
        "timeout_seconds": 15,
        "background": True,
    }


def test_live_mind_config_omits_absent_llm_keys() -> None:
    cfg = LiveMindConfig.from_yaml({"llm": {"mock": True}})
    assert cfg.llm_mock is True
    assert cfg.llm_kwargs == {}
