"""Fly profile registry tests."""

from __future__ import annotations

from pathlib import Path

from flybrain.fly_profiles import (
    FlyProfileStore,
    fly_id_from_number,
    place_cache_dir_for_fly,
)


def test_fly_id_from_number() -> None:
    assert fly_id_from_number(1) == "fly_001"
    assert fly_id_from_number(12) == "fly_012"


def test_store_create_and_select(tmp_path: Path) -> None:
    path = tmp_path / "fly_profiles.json"
    store = FlyProfileStore.load(path)
    assert store.active().number == 1
    assert store.active().fly_id == "fly_001"

    fly2 = store.ensure(2, name="Explorer")
    assert fly2.fly_id == "fly_002"
    store.set_active(fly2.fly_id)
    assert store.active_fly_id == "fly_002"

    reloaded = FlyProfileStore.load(path)
    assert reloaded.active_fly_id == "fly_002"
    assert reloaded.get_by_number(2).name == "Explorer"


def test_resolve_number_creates(tmp_path: Path) -> None:
    store = FlyProfileStore.load(tmp_path / "p.json")
    profile = store.resolve(fly=3)
    assert profile.fly_id == "fly_003"
    assert store.active_fly_id == "fly_003"


def test_interactive_select_new(tmp_path: Path) -> None:
    store = FlyProfileStore.load(tmp_path / "p.json")
    answers = iter(["n", "Scout"])
    profile = store.interactive_select(
        input_fn=lambda _prompt: next(answers), print_fn=lambda *_: None
    )
    assert profile.number == 2
    assert profile.name == "Scout"


def test_place_cache_dir_for_fly() -> None:
    assert place_cache_dir_for_fly("fly_002").endswith("fly_002")


def test_session_ready_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "fly_profiles.json"
    store = FlyProfileStore.load(path)
    store.ensure(3, name="Fly 3")
    store.set_active("fly_003", session_ready=False)
    assert store.session_ready is False
    assert store.active_fly_id == "fly_003"

    reloaded = FlyProfileStore.load(path)
    assert reloaded.session_ready is False
    assert reloaded.active_fly_id == "fly_003"
    assert reloaded.get("fly_003") is not None
    assert reloaded.get("fly_003").name == "Fly 3"

    reloaded.set_session_ready(True)
    ready = FlyProfileStore.load(path)
    assert ready.session_ready is True
    assert ready.active_fly_id == "fly_003"


def test_wait_until_session_ready(tmp_path: Path) -> None:
    path = tmp_path / "fly_profiles.json"
    store = FlyProfileStore.load(path)
    store.ensure(3, name="Fly 3")
    store.set_active("fly_003", session_ready=False)

    ticks = {"n": 0}

    def pump() -> None:
        ticks["n"] += 1
        if ticks["n"] >= 2:
            other = FlyProfileStore.load(path)
            other.set_active("fly_003", touch=True, session_ready=True)

    profile = store.wait_until_session_ready(timeout_s=5.0, poll_s=0.01, pump=pump, print_fn=None)
    assert profile.fly_id == "fly_003"
    assert store.session_ready is True
