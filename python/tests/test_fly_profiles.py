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
