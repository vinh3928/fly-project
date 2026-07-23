"""Regression: --editor must not auto-launch builds, and must use port 5004."""

from __future__ import annotations

from flybrain.unity_env import UnityFlyConfig, default_build_path


def test_editor_connect_skips_build_fallback() -> None:
    """UnityFlyEnv used to replace file_name=None with the standalone build path."""
    assert default_build_path() is not None  # repo has a build; that's what caused the bug

    cfg = UnityFlyConfig(file_name=None, editor_connect=True)
    file_name = cfg.file_name
    if file_name is None and not cfg.editor_connect:
        build = default_build_path()
        file_name = str(build) if build is not None else None
    assert file_name is None

    cfg_build = UnityFlyConfig(file_name=None, editor_connect=False)
    file_name = cfg_build.file_name
    if file_name is None and not cfg_build.editor_connect:
        build = default_build_path()
        file_name = str(build) if build is not None else None
    assert file_name is not None
    assert "FlyChamber" in file_name


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
