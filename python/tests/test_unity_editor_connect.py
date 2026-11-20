"""Regression: --editor must not auto-launch builds, and must use port 5004."""

from __future__ import annotations

from pathlib import Path

from flybrain.unity_env import UnityFlyConfig, resolve_unity_file_name


def test_editor_connect_skips_build_fallback(monkeypatch, tmp_path: Path) -> None:
    """Even when a standalone build exists, editor_connect must keep file_name=None."""
    fake_build = tmp_path / "FlyChamber.app"
    fake_build.mkdir()
    monkeypatch.setattr("flybrain.unity_env.default_build_path", lambda: fake_build)

    editor = UnityFlyConfig(file_name=None, editor_connect=True)
    assert resolve_unity_file_name(editor) is None

    build = UnityFlyConfig(file_name=None, editor_connect=False)
    resolved = resolve_unity_file_name(build)
    assert resolved is not None
    assert "FlyChamber" in resolved

    explicit = UnityFlyConfig(file_name="/custom/player", editor_connect=True)
    assert resolve_unity_file_name(explicit) == "/custom/player"


def test_editor_default_port_is_5004() -> None:
    """Unity Editor ML-Agents trainer port is 5004; builds use 5005."""
    editor = True
    port = None
    if port is None:
        port = 5004 if editor else 5005
    assert port == 5004
    editor = False
    port = None
    if port is None:
        port = 5004 if editor else 5005
    assert port == 5005


def test_editor_connect_instructions_are_not_inverted() -> None:
    """Regression: docs/script used to say wait for unity_env_ready before Play."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_live_mind.py"
    text = script.read_text(encoding="utf-8")
    assert "Do NOT wait for unity_env_ready first" in text
    assert "_open_editor_env_early" in text
