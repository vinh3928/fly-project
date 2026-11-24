"""Skill cortex: discovered motor programs on a separate Neo4j partition.

Stores timed 5-d goal sequences (groom/touch, dance, …) as candidates that
Father can promote into the reusable skill library. Place-graph Neo4j stays
on a different cluster; place_id on Context is a soft string reference only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Protocol
from uuid import uuid4

import numpy as np

from flybrain.phase9 import MOTOR_GOAL_DIM

SkillStatus = Literal["candidate", "promoted", "retired"]
SkillSource = Literal["babble", "father", "seed"]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp01(value: float, default: float = 0.0) -> float:
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except (TypeError, ValueError):
        return float(default)


def _clip_goal(goal: np.ndarray) -> np.ndarray:
    """Local 5-d clip (avoid importing motor — breaks circular import)."""
    g = np.asarray(goal, dtype=np.float32).reshape(-1)
    out = np.zeros(MOTOR_GOAL_DIM, dtype=np.float32)
    n = min(MOTOR_GOAL_DIM, g.size)
    out[:n] = g[:n]
    out[0] = float(np.clip(out[0], -1.0, 1.0))
    out[1] = float(np.clip(out[1], -1.0, 1.0))
    out[2] = float(np.clip(out[2], 0.0, 1.0))
    out[3] = float(np.clip(out[3], 0.0, 1.0))
    out[4] = float(np.clip(out[4], 0.0, 1.0))
    return out


@dataclass
class SkillBabbleConfig:
    enabled: bool = True
    min_novelty: float = 0.35
    max_survival_risk: float = 0.45
    max_hunger: float = 0.55
    cooldown_steps: int = 200
    max_steps: int = 8
    min_steps: int = 3
    duration_range: tuple[int, int] = (8, 24)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> SkillBabbleConfig:
        data = dict(raw or {})
        dur = data.get("duration_range", [8, 24])
        if isinstance(dur, (list, tuple)) and len(dur) >= 2:
            lo, hi = int(dur[0]), int(dur[1])
        else:
            lo, hi = 8, 24
        if hi < lo:
            lo, hi = hi, lo
        return cls(
            enabled=bool(data.get("enabled", True)),
            min_novelty=float(data.get("min_novelty", 0.35)),
            max_survival_risk=float(data.get("max_survival_risk", 0.45)),
            max_hunger=float(data.get("max_hunger", 0.55)),
            cooldown_steps=max(0, int(data.get("cooldown_steps", 200))),
            max_steps=max(1, int(data.get("max_steps", 8))),
            min_steps=max(1, int(data.get("min_steps", 3))),
            duration_range=(lo, hi),
        )


@dataclass
class SkillCortexConfig:
    enabled: bool = False
    backend: str = "in_memory"
    uri: str = "bolt://127.0.0.1:7688"
    user: str = "neo4j"
    password: str | None = None
    database: str = "neo4j"
    promote_min_score: float = 0.4
    # When Display 4 is not focused, auto Ask-LLM. 0 = immediate, >0 = grace
    # seconds, <0 = disable Python fallback (Unity may still auto-send when away).
    auto_llm_away_seconds: float = 0.0
    babble: SkillBabbleConfig = field(default_factory=SkillBabbleConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> SkillCortexConfig:
        data = dict(raw or {})
        return cls(
            enabled=bool(data.get("enabled", False)),
            backend=str(data.get("backend", "in_memory")),
            uri=str(data.get("uri", "bolt://127.0.0.1:7688")),
            user=str(data.get("user", "neo4j")),
            password=data.get("password"),
            database=str(data.get("database", "neo4j")),
            promote_min_score=float(data.get("promote_min_score", 0.4)),
            auto_llm_away_seconds=float(data.get("auto_llm_away_seconds", 0.0)),
            babble=SkillBabbleConfig.from_dict(
                data.get("babble") if isinstance(data.get("babble"), dict) else {}
            ),
        )


@dataclass
class SkillStep:
    goal: np.ndarray
    duration_steps: int = 12

    def __post_init__(self) -> None:
        self.goal = _clip_goal(np.asarray(self.goal, dtype=np.float32))
        self.duration_steps = max(1, int(self.duration_steps))

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": [float(x) for x in np.asarray(self.goal, dtype=np.float32).tolist()],
            "duration_steps": int(self.duration_steps),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> SkillStep:
        data = dict(raw or {})
        goal = data.get("goal") or [0.0, 0.0, 0.5, 0.3, 0.0]
        return cls(
            goal=np.asarray(goal, dtype=np.float32),
            duration_steps=int(data.get("duration_steps", 12) or 12),
        )


@dataclass
class SkillContext:
    cue: str = ""
    place_id: str | None = None
    step: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cue": str(self.cue or ""),
            "place_id": self.place_id,
            "step": int(self.step),
        }


@dataclass
class SkillProgram:
    skill_id: str
    name: str
    status: SkillStatus
    steps: list[SkillStep]
    tags: list[str] = field(default_factory=list)
    source: str = "babble"
    fly_id: str = "fly_001"
    score: float = 0.0
    created_at: str = ""
    promoted_at: str | None = None
    context: SkillContext | None = None

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = _utcnow()
        self.score = _clamp01(self.score)
        self.steps = list(self.steps or [])
        self.tags = [str(t) for t in (self.tags or [])]

    @property
    def first_goal(self) -> np.ndarray:
        if not self.steps:
            return _clip_goal(np.zeros(MOTOR_GOAL_DIM, dtype=np.float32))
        return np.asarray(self.steps[0].goal, dtype=np.float32).copy()

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "status": self.status,
            "steps": [s.to_dict() for s in self.steps],
            "tags": list(self.tags),
            "source": self.source,
            "fly_id": self.fly_id,
            "score": float(self.score),
            "created_at": self.created_at,
            "promoted_at": self.promoted_at,
            "context": self.context.to_dict() if self.context else None,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> SkillProgram:
        data = dict(raw or {})
        steps_raw = data.get("steps") or []
        steps = [
            SkillStep.from_dict(s) if isinstance(s, dict) else SkillStep(goal=[0, 0, 0.5, 0.3, 0])
            for s in steps_raw
        ]
        ctx_raw = data.get("context")
        ctx = None
        if isinstance(ctx_raw, dict):
            ctx = SkillContext(
                cue=str(ctx_raw.get("cue") or ""),
                place_id=(
                    str(ctx_raw["place_id"]) if ctx_raw.get("place_id") is not None else None
                ),
                step=int(ctx_raw.get("step") or 0),
            )
        status = str(data.get("status") or "candidate")
        if status not in ("candidate", "promoted", "retired"):
            status = "candidate"
        return cls(
            skill_id=str(data.get("skill_id") or uuid4()),
            name=str(data.get("name") or "unnamed"),
            status=status,  # type: ignore[arg-type]
            steps=steps,
            tags=[str(t) for t in (data.get("tags") or [])],
            source=str(data.get("source") or "babble"),
            fly_id=str(data.get("fly_id") or "fly_001"),
            score=float(data.get("score") or 0.0),
            created_at=str(data.get("created_at") or ""),
            promoted_at=str(data["promoted_at"]) if data.get("promoted_at") else None,
            context=ctx,
        )


class SkillCortexStore(Protocol):
    def upsert_skill(self, program: SkillProgram) -> SkillProgram: ...

    def get_skill(self, skill_id: str) -> SkillProgram | None: ...

    def get_by_name(
        self,
        name: str,
        *,
        fly_id: str = "fly_001",
        status: SkillStatus | None = "promoted",
    ) -> SkillProgram | None: ...

    def list_skills(
        self,
        *,
        fly_id: str = "fly_001",
        status: SkillStatus | None = None,
        limit: int = 64,
    ) -> list[SkillProgram]: ...

    def promote_skill(
        self,
        skill_id: str,
        *,
        name: str | None = None,
        tags: list[str] | None = None,
    ) -> SkillProgram | None: ...

    def retire_skill(self, skill_id: str) -> SkillProgram | None: ...

    def set_score(self, skill_id: str, score: float) -> SkillProgram | None: ...

    def close(self) -> None: ...


@dataclass
class InMemorySkillCortex:
    """RAM skill cortex for tests / offline."""

    _skills: dict[str, SkillProgram] = field(default_factory=dict)

    def upsert_skill(self, program: SkillProgram) -> SkillProgram:
        row = SkillProgram.from_dict(program.to_dict())
        if not row.skill_id:
            row.skill_id = str(uuid4())
        self._skills[row.skill_id] = row
        return SkillProgram.from_dict(row.to_dict())

    def get_skill(self, skill_id: str) -> SkillProgram | None:
        row = self._skills.get(str(skill_id))
        return SkillProgram.from_dict(row.to_dict()) if row else None

    def get_by_name(
        self,
        name: str,
        *,
        fly_id: str = "fly_001",
        status: SkillStatus | None = "promoted",
    ) -> SkillProgram | None:
        key = str(name or "").strip().lower()
        if not key:
            return None
        for row in reversed(list(self._skills.values())):
            if str(row.fly_id) != str(fly_id):
                continue
            if str(row.name).strip().lower() != key:
                continue
            if status is not None and row.status != status:
                continue
            return SkillProgram.from_dict(row.to_dict())
        return None

    def list_skills(
        self,
        *,
        fly_id: str = "fly_001",
        status: SkillStatus | None = None,
        limit: int = 64,
    ) -> list[SkillProgram]:
        rows = [
            SkillProgram.from_dict(s.to_dict())
            for s in self._skills.values()
            if str(s.fly_id) == str(fly_id) and (status is None or s.status == status)
        ]
        rows.sort(key=lambda s: (s.promoted_at or "", s.created_at), reverse=True)
        return rows[: max(1, int(limit))]

    def promote_skill(
        self,
        skill_id: str,
        *,
        name: str | None = None,
        tags: list[str] | None = None,
    ) -> SkillProgram | None:
        row = self._skills.get(str(skill_id))
        if row is None:
            return None
        row.status = "promoted"
        row.promoted_at = _utcnow()
        if name is not None and str(name).strip():
            row.name = str(name).strip()
        if tags:
            merged = list(dict.fromkeys([*row.tags, *[str(t) for t in tags]]))
            row.tags = merged
        return SkillProgram.from_dict(row.to_dict())

    def retire_skill(self, skill_id: str) -> SkillProgram | None:
        row = self._skills.get(str(skill_id))
        if row is None:
            return None
        row.status = "retired"
        return SkillProgram.from_dict(row.to_dict())

    def set_score(self, skill_id: str, score: float) -> SkillProgram | None:
        row = self._skills.get(str(skill_id))
        if row is None:
            return None
        row.score = _clamp01(score)
        return SkillProgram.from_dict(row.to_dict())

    def close(self) -> None:
        return None


@dataclass
class Neo4jSkillCortex:
    """Neo4j skill cortex on the dedicated skill cluster."""

    uri: str = "bolt://localhost:7688"
    user: str = "neo4j"
    password: str = "flymindskill"
    database: str = "neo4j"
    _driver: Any = field(default=None, init=False, repr=False)

    SKILL_ID_CONSTRAINT = "skill_id_unique"
    STEP_ID_CONSTRAINT = "skill_step_id_unique"

    def __post_init__(self) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install mind extras: pip install -e '.[mind]'") from exc
        self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self._ensure_schema()

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def _query(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[Any]:
        if self._driver is None:
            raise RuntimeError("Neo4j skill cortex is closed")
        records, _summary, _keys = self._driver.execute_query(
            cypher,
            parameters_=parameters or {},
            database_=self.database,
        )
        return list(records)

    def _ensure_schema(self) -> None:
        self._query(f"""
            CREATE CONSTRAINT {self.SKILL_ID_CONSTRAINT} IF NOT EXISTS
            FOR (s:Skill) REQUIRE s.skill_id IS UNIQUE
            """)
        self._query(f"""
            CREATE CONSTRAINT {self.STEP_ID_CONSTRAINT} IF NOT EXISTS
            FOR (t:SkillStep) REQUIRE t.step_id IS UNIQUE
            """)
        self._query("""
            CREATE INDEX skill_fly_status IF NOT EXISTS
            FOR (s:Skill) ON (s.fly_id, s.status, s.updated_at)
            """)

    def upsert_skill(self, program: SkillProgram) -> SkillProgram:
        row = SkillProgram.from_dict(program.to_dict())
        if not row.skill_id:
            row.skill_id = str(uuid4())
        now = _utcnow()
        if not row.created_at:
            row.created_at = now
        tags_json = json.dumps(list(row.tags))
        self._query(
            """
            MERGE (s:Skill {skill_id: $skill_id})
            SET s.name = $name,
                s.status = $status,
                s.tags_json = $tags_json,
                s.source = $source,
                s.fly_id = $fly_id,
                s.score = $score,
                s.created_at = coalesce(s.created_at, $created_at),
                s.promoted_at = $promoted_at,
                s.updated_at = $updated_at
            WITH s
            OPTIONAL MATCH (s)-[r:HAS_STEP]->(old:SkillStep)
            DELETE r, old
            """,
            {
                "skill_id": row.skill_id,
                "name": row.name,
                "status": row.status,
                "tags_json": tags_json,
                "source": row.source,
                "fly_id": row.fly_id,
                "score": float(row.score),
                "created_at": row.created_at,
                "promoted_at": row.promoted_at,
                "updated_at": now,
            },
        )
        for ord_i, step in enumerate(row.steps):
            step_id = f"{row.skill_id}:{ord_i}"
            self._query(
                """
                MATCH (s:Skill {skill_id: $skill_id})
                MERGE (t:SkillStep {step_id: $step_id})
                SET t.goal_json = $goal_json,
                    t.duration_steps = $duration_steps,
                    t.ord = $ord
                MERGE (s)-[r:HAS_STEP {ord: $ord}]->(t)
                """,
                {
                    "skill_id": row.skill_id,
                    "step_id": step_id,
                    "goal_json": json.dumps(
                        [float(x) for x in np.asarray(step.goal, dtype=np.float32).tolist()]
                    ),
                    "duration_steps": int(step.duration_steps),
                    "ord": int(ord_i),
                },
            )
        if row.context is not None:
            ctx = row.context
            self._query(
                """
                MATCH (s:Skill {skill_id: $skill_id})
                MERGE (c:Context {skill_id: $skill_id})
                SET c.cue = $cue,
                    c.place_id = $place_id,
                    c.step = $step
                MERGE (s)-[:DISCOVERED_AT]->(c)
                """,
                {
                    "skill_id": row.skill_id,
                    "cue": str(ctx.cue or ""),
                    "place_id": ctx.place_id,
                    "step": int(ctx.step),
                },
            )
        return row

    def _record_to_program(self, record: Any) -> SkillProgram:
        tags_raw = record.get("tags_json") or "[]"
        try:
            tags = json.loads(tags_raw) if isinstance(tags_raw, str) else list(tags_raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            tags = []
        steps_raw = record.get("steps") or []
        steps: list[SkillStep] = []
        for raw in steps_raw:
            if not isinstance(raw, dict):
                raw = dict(raw)
            goal_raw = raw.get("goal_json") or "[]"
            try:
                goal = json.loads(goal_raw) if isinstance(goal_raw, str) else list(goal_raw)
            except (TypeError, ValueError, json.JSONDecodeError):
                goal = [0.0, 0.0, 0.5, 0.3, 0.0]
            steps.append(
                SkillStep(
                    goal=np.asarray(goal, dtype=np.float32),
                    duration_steps=int(raw.get("duration_steps") or 12),
                )
            )
        ctx = None
        if record.get("ctx_cue") is not None or record.get("ctx_place_id") is not None:
            ctx = SkillContext(
                cue=str(record.get("ctx_cue") or ""),
                place_id=(
                    str(record["ctx_place_id"]) if record.get("ctx_place_id") is not None else None
                ),
                step=int(record.get("ctx_step") or 0),
            )
        status = str(record.get("status") or "candidate")
        if status not in ("candidate", "promoted", "retired"):
            status = "candidate"
        return SkillProgram(
            skill_id=str(record.get("skill_id")),
            name=str(record.get("name") or "unnamed"),
            status=status,  # type: ignore[arg-type]
            steps=steps,
            tags=[str(t) for t in tags],
            source=str(record.get("source") or "babble"),
            fly_id=str(record.get("fly_id") or "fly_001"),
            score=float(record.get("score") or 0.0),
            created_at=str(record.get("created_at") or ""),
            promoted_at=str(record["promoted_at"]) if record.get("promoted_at") else None,
            context=ctx,
        )

    _RETURN = """
        s.skill_id AS skill_id,
        s.name AS name,
        s.status AS status,
        s.tags_json AS tags_json,
        s.source AS source,
        s.fly_id AS fly_id,
        s.score AS score,
        s.created_at AS created_at,
        s.promoted_at AS promoted_at,
        [(s)-[r:HAS_STEP]->(t:SkillStep) | {
            ord: r.ord,
            goal_json: t.goal_json,
            duration_steps: t.duration_steps
        }] AS steps,
        head([(s)-[:DISCOVERED_AT]->(c:Context) | c.cue]) AS ctx_cue,
        head([(s)-[:DISCOVERED_AT]->(c:Context) | c.place_id]) AS ctx_place_id,
        head([(s)-[:DISCOVERED_AT]->(c:Context) | c.step]) AS ctx_step
    """

    def get_skill(self, skill_id: str) -> SkillProgram | None:
        rows = self._query(
            f"""
            MATCH (s:Skill {{skill_id: $skill_id}})
            RETURN {self._RETURN}
            """,
            {"skill_id": str(skill_id)},
        )
        if not rows:
            return None
        prog = self._record_to_program(rows[0])
        raw_steps = rows[0].get("steps") or []
        ordered = sorted(
            [dict(x) if not isinstance(x, dict) else x for x in raw_steps],
            key=lambda d: int(d.get("ord") or 0),
        )
        prog.steps = [
            SkillStep(
                goal=np.asarray(
                    (
                        json.loads(d["goal_json"])
                        if isinstance(d.get("goal_json"), str)
                        else (d.get("goal_json") or [0, 0, 0.5, 0.3, 0])
                    ),
                    dtype=np.float32,
                ),
                duration_steps=int(d.get("duration_steps") or 12),
            )
            for d in ordered
        ]
        return prog

    def get_by_name(
        self,
        name: str,
        *,
        fly_id: str = "fly_001",
        status: SkillStatus | None = "promoted",
    ) -> SkillProgram | None:
        key = str(name or "").strip().lower()
        if not key:
            return None
        if status is None:
            rows = self._query(
                f"""
                MATCH (s:Skill {{fly_id: $fly_id}})
                WHERE toLower(s.name) = $name
                RETURN {self._RETURN}
                ORDER BY coalesce(s.promoted_at, s.created_at) DESC
                LIMIT 1
                """,
                {"fly_id": str(fly_id), "name": key},
            )
        else:
            rows = self._query(
                f"""
                MATCH (s:Skill {{fly_id: $fly_id, status: $status}})
                WHERE toLower(s.name) = $name
                RETURN {self._RETURN}
                ORDER BY coalesce(s.promoted_at, s.created_at) DESC
                LIMIT 1
                """,
                {"fly_id": str(fly_id), "name": key, "status": status},
            )
        if not rows:
            return None
        return self.get_skill(str(rows[0].get("skill_id")))

    def list_skills(
        self,
        *,
        fly_id: str = "fly_001",
        status: SkillStatus | None = None,
        limit: int = 64,
    ) -> list[SkillProgram]:
        lim = max(1, int(limit))
        if status is None:
            rows = self._query(
                f"""
                MATCH (s:Skill {{fly_id: $fly_id}})
                RETURN {self._RETURN}
                ORDER BY coalesce(s.promoted_at, s.created_at) DESC
                LIMIT $limit
                """,
                {"fly_id": str(fly_id), "limit": lim},
            )
        else:
            rows = self._query(
                f"""
                MATCH (s:Skill {{fly_id: $fly_id, status: $status}})
                RETURN {self._RETURN}
                ORDER BY coalesce(s.promoted_at, s.created_at) DESC
                LIMIT $limit
                """,
                {"fly_id": str(fly_id), "status": status, "limit": lim},
            )
        out: list[SkillProgram] = []
        for row in rows:
            sid = str(row.get("skill_id") or "")
            got = self.get_skill(sid) if sid else None
            if got is not None:
                out.append(got)
        return out

    def promote_skill(
        self,
        skill_id: str,
        *,
        name: str | None = None,
        tags: list[str] | None = None,
    ) -> SkillProgram | None:
        existing = self.get_skill(str(skill_id))
        if existing is None:
            return None
        existing.status = "promoted"
        existing.promoted_at = _utcnow()
        if name is not None and str(name).strip():
            existing.name = str(name).strip()
        if tags:
            existing.tags = list(dict.fromkeys([*existing.tags, *[str(t) for t in tags]]))
        return self.upsert_skill(existing)

    def retire_skill(self, skill_id: str) -> SkillProgram | None:
        existing = self.get_skill(str(skill_id))
        if existing is None:
            return None
        existing.status = "retired"
        return self.upsert_skill(existing)

    def set_score(self, skill_id: str, score: float) -> SkillProgram | None:
        existing = self.get_skill(str(skill_id))
        if existing is None:
            return None
        existing.score = _clamp01(score)
        return self.upsert_skill(existing)


def open_skill_cortex(
    *,
    backend: str = "in_memory",
    uri: str = "bolt://127.0.0.1:7688",
    user: str = "neo4j",
    password: str | None = None,
    database: str = "neo4j",
) -> SkillCortexStore:
    kind = str(backend or "in_memory").strip().lower()
    if kind == "in_memory":
        return InMemorySkillCortex()
    if kind != "neo4j":
        raise ValueError(f"unsupported skill cortex backend: {backend!r}")
    return Neo4jSkillCortex(
        uri=uri,
        user=user,
        password=password or os.getenv("NEO4J_SKILL_PASSWORD", "flymindskill"),
        database=database,
    )


def new_skill_id() -> str:
    return str(uuid4())
