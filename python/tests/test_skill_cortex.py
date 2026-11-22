"""Skill cortex store: in-memory + Neo4j Cypher shape."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from flybrain.memory.skill_cortex import (
    InMemorySkillCortex,
    Neo4jSkillCortex,
    SkillCortexConfig,
    SkillProgram,
    SkillStep,
    open_skill_cortex,
)


def _program(**kwargs) -> SkillProgram:
    base = dict(
        skill_id="skill-1",
        name="candidate_dance_abc",
        status="candidate",
        steps=[
            SkillStep(goal=np.asarray([0.2, 0.8, 0.5, 0.7, 0.0], dtype=np.float32), duration_steps=10),
            SkillStep(goal=np.asarray([0.2, -0.8, 0.5, 0.7, 0.0], dtype=np.float32), duration_steps=10),
        ],
        tags=["dance"],
        source="babble",
        fly_id="fly_a",
        score=0.0,
    )
    base.update(kwargs)
    return SkillProgram(**base)


def test_in_memory_upsert_list_promote() -> None:
    store = InMemorySkillCortex()
    row = store.upsert_skill(_program())
    assert row.skill_id == "skill-1"
    assert len(row.steps) == 2
    listed = store.list_skills(fly_id="fly_a", status="candidate")
    assert len(listed) == 1
    assert store.get_by_name("candidate_dance_abc", fly_id="fly_a", status="candidate") is not None
    promoted = store.promote_skill("skill-1", name="dance_spin", tags=["father"])
    assert promoted is not None
    assert promoted.status == "promoted"
    assert promoted.name == "dance_spin"
    assert "father" in promoted.tags
    assert store.get_by_name("dance_spin", fly_id="fly_a", status="promoted") is not None
    scored = store.set_score("skill-1", 0.77)
    assert scored is not None
    assert scored.score == 0.77


def test_skill_cortex_config_from_dict() -> None:
    cfg = SkillCortexConfig.from_dict(
        {
            "enabled": True,
            "backend": "neo4j",
            "uri": "bolt://127.0.0.1:7688",
            "babble": {"min_steps": 4, "duration_range": [5, 12]},
            "promote_min_score": 0.5,
        }
    )
    assert cfg.enabled is True
    assert cfg.uri.endswith(":7688")
    assert cfg.babble.min_steps == 4
    assert cfg.babble.duration_range == (5, 12)
    assert cfg.promote_min_score == 0.5


def test_open_skill_cortex_in_memory() -> None:
    store = open_skill_cortex(backend="in_memory")
    assert isinstance(store, InMemorySkillCortex)
    store.close()


def test_neo4j_upsert_skill_writes_has_step(monkeypatch) -> None:
    class _FakeDriver:
        def __init__(self) -> None:
            self.closed = False

        def execute_query(self, cypher, parameters_=None, database_=None):
            return [], None, None

        def close(self) -> None:
            self.closed = True

    driver = _FakeDriver()
    graph_database = SimpleNamespace(driver=lambda *args, **kwargs: driver)
    monkeypatch.setitem(sys.modules, "neo4j", SimpleNamespace(GraphDatabase=graph_database))
    cortex = Neo4jSkillCortex(password="test-password")
    captured: list[tuple[str, dict]] = []

    def _query(cypher, params=None):
        captured.append((str(cypher), dict(params or {})))
        return []

    cortex._query = _query  # type: ignore[method-assign]
    row = cortex.upsert_skill(_program(skill_id="s-42", name="touch_self"))
    assert row.skill_id == "s-42"
    assert any(":Skill" in q for q, _ in captured)
    assert any("HAS_STEP" in q for q, _ in captured)
    assert any(p.get("name") == "touch_self" for _, p in captured if "name" in p)
    cortex.close()
