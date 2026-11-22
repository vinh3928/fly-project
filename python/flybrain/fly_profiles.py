"""Fly profile registry: choose Fly 1 / 2 / 3 … before a live run.

Profiles are stored in ``data/fly_profiles.json`` (repo root) so Unity and
Python share the same roster and the active selection.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_PROFILES_PATH = Path(__file__).resolve().parents[2] / "data" / "fly_profiles.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def fly_id_from_number(n: int) -> str:
    return f"fly_{max(1, int(n)):03d}"


def parse_fly_number(fly_id: str) -> int | None:
    match = re.fullmatch(r"fly_(\d+)", str(fly_id or "").strip().lower())
    if not match:
        return None
    return int(match.group(1))


@dataclass
class FlyProfile:
    fly_id: str
    name: str
    number: int
    created_at: str = field(default_factory=_utc_now)
    last_played_at: str | None = None
    notes: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FlyProfile:
        fly_id = str(raw.get("fly_id") or "fly_001")
        number = int(raw.get("number") or parse_fly_number(fly_id) or 1)
        name = str(raw.get("name") or f"Fly {number}")
        return cls(
            fly_id=fly_id,
            name=name,
            number=number,
            created_at=str(raw.get("created_at") or _utc_now()),
            last_played_at=(str(raw["last_played_at"]) if raw.get("last_played_at") else None),
            notes=str(raw.get("notes") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FlyProfileStore:
    """JSON-backed roster of flies + which one is active."""

    path: Path = field(default_factory=lambda: DEFAULT_PROFILES_PATH)
    profiles: list[FlyProfile] = field(default_factory=list)
    active_fly_id: str = "fly_001"

    @classmethod
    def load(cls, path: Path | None = None) -> FlyProfileStore:
        store = cls(path=path or DEFAULT_PROFILES_PATH)
        store.reload()
        return store

    def reload(self) -> None:
        path = self.path
        if not path.exists():
            self.profiles = [self._default_profile(1)]
            self.active_fly_id = self.profiles[0].fly_id
            self.save()
            return
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = raw.get("profiles") if isinstance(raw, dict) else raw
        profiles: list[FlyProfile] = []
        if isinstance(rows, list):
            for item in rows:
                if isinstance(item, dict):
                    profiles.append(FlyProfile.from_dict(item))
        if not profiles:
            profiles = [self._default_profile(1)]
        profiles.sort(key=lambda p: p.number)
        self.profiles = profiles
        active = ""
        if isinstance(raw, dict):
            active = str(raw.get("active_fly_id") or "")
        if not active or self.get(active) is None:
            active = self.profiles[0].fly_id
        self.active_fly_id = active

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "active_fly_id": self.active_fly_id,
            "profiles": [p.to_dict() for p in sorted(self.profiles, key=lambda x: x.number)],
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # Mirror for Unity StreamingAssets when present.
        unity_mirror = (
            Path(__file__).resolve().parents[2]
            / "unity"
            / "Assets"
            / "StreamingAssets"
            / "fly_profiles.json"
        )
        try:
            unity_mirror.parent.mkdir(parents=True, exist_ok=True)
            unity_mirror.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        except OSError:
            pass

    @staticmethod
    def _default_profile(number: int, name: str | None = None) -> FlyProfile:
        n = max(1, int(number))
        return FlyProfile(
            fly_id=fly_id_from_number(n),
            name=name or f"Fly {n}",
            number=n,
        )

    def get(self, fly_id: str) -> FlyProfile | None:
        key = str(fly_id or "").strip()
        for profile in self.profiles:
            if profile.fly_id == key:
                return profile
        return None

    def get_by_number(self, number: int) -> FlyProfile | None:
        n = int(number)
        for profile in self.profiles:
            if profile.number == n:
                return profile
        return None

    def active(self) -> FlyProfile:
        profile = self.get(self.active_fly_id)
        if profile is not None:
            return profile
        return self.profiles[0]

    def next_number(self) -> int:
        if not self.profiles:
            return 1
        return max(p.number for p in self.profiles) + 1

    def ensure(self, number: int, *, name: str | None = None) -> FlyProfile:
        existing = self.get_by_number(number)
        if existing is not None:
            if name:
                existing.name = name
            return existing
        profile = self._default_profile(number, name=name)
        self.profiles.append(profile)
        self.profiles.sort(key=lambda p: p.number)
        return profile

    def create(self, *, name: str | None = None, notes: str = "") -> FlyProfile:
        profile = self._default_profile(self.next_number(), name=name)
        profile.notes = notes
        self.profiles.append(profile)
        self.profiles.sort(key=lambda p: p.number)
        return profile

    def set_active(self, fly_id: str, *, touch: bool = True) -> FlyProfile:
        profile = self.get(fly_id)
        if profile is None:
            raise KeyError(f"unknown fly_id: {fly_id}")
        self.active_fly_id = profile.fly_id
        if touch:
            profile.last_played_at = _utc_now()
        self.save()
        return profile

    def set_active_number(self, number: int, *, touch: bool = True) -> FlyProfile:
        profile = self.ensure(number)
        return self.set_active(profile.fly_id, touch=touch)

    def resolve(
        self,
        *,
        fly: str | int | None = None,
        fly_id: str | None = None,
        create_if_missing: bool = True,
    ) -> FlyProfile:
        """Resolve CLI / config selection into a profile and mark it active."""
        if fly_id:
            profile = self.get(str(fly_id))
            if profile is None and create_if_missing:
                num = parse_fly_number(str(fly_id)) or self.next_number()
                profile = self.ensure(num)
                # Preserve explicit id if non-standard.
                if str(fly_id) != profile.fly_id:
                    profile.fly_id = str(fly_id)
            if profile is None:
                raise KeyError(f"unknown fly_id: {fly_id}")
            return self.set_active(profile.fly_id)
        if fly is not None and str(fly).strip() != "":
            text = str(fly).strip()
            if text.isdigit():
                num = int(text)
                if create_if_missing:
                    self.ensure(num)
                profile = self.get_by_number(num)
                if profile is None:
                    raise KeyError(f"unknown fly number: {num}")
                return self.set_active(profile.fly_id)
            # Treat as fly_id string.
            return self.resolve(fly_id=text, create_if_missing=create_if_missing)
        return self.set_active(self.active().fly_id)

    def format_menu(self) -> str:
        lines = ["Fly profiles:"]
        active = self.active_fly_id
        for profile in self.profiles:
            mark = "*" if profile.fly_id == active else " "
            played = profile.last_played_at or "-"
            lines.append(
                f"  {mark} [{profile.number}] {profile.name}  ({profile.fly_id})  last={played}"
            )
        lines.append("  [n] create new fly")
        return "\n".join(lines)

    def interactive_select(self, *, input_fn=input, print_fn=print) -> FlyProfile:
        """TTY menu: pick a number or create a new fly."""
        while True:
            print_fn(self.format_menu())
            print_fn(f"Active: {self.active().name} ({self.active_fly_id})")
            choice = str(input_fn("Select fly number (Enter=active, n=new): ")).strip().lower()
            if choice in ("", "a", "active"):
                return self.set_active(self.active_fly_id)
            if choice in ("n", "new", "+"):
                name = str(input_fn("Name for new fly (Enter=Fly N): ")).strip()
                profile = self.create(name=name or None)
                return self.set_active(profile.fly_id)
            if choice.isdigit():
                return self.set_active_number(int(choice))
            print_fn(f"Unrecognized choice: {choice!r}")


def place_cache_dir_for_fly(fly_id: str, base: str = "logs/place_cache") -> str:
    """Per-fly place-cache directory so maps do not collide."""
    safe = re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(fly_id or "fly_001"))
    return str(Path(base) / safe)
