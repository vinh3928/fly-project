"""Short-term memory stores: in-memory and Postgres pgvector."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

import numpy as np

from flybrain.phase4 import TOTAL_EMBEDDING_DIM


@dataclass
class StmRecord:
    id: int
    episode_id: str
    step: int
    sim_time: float
    fly_id: str
    embedding: np.ndarray
    signals: dict[str, Any]
    affect: dict[str, Any]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ThoughtRecord:
    id: int
    stm_id: int | None
    reflection: str
    devil_critique: str
    refined_thought: str
    intent: dict[str, Any]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ShortTermMemory(Protocol):
    def insert_embedding(
        self,
        *,
        embedding: np.ndarray,
        signals: dict[str, Any],
        affect: dict[str, Any],
        episode_id: str = "live",
        step: int = 0,
        sim_time: float = 0.0,
        fly_id: str = "fly_001",
    ) -> StmRecord: ...

    def similar(self, embedding: np.ndarray, *, fly_id: str, top_k: int = 5) -> list[StmRecord]: ...

    def recent(self, *, fly_id: str, limit: int = 8) -> list[StmRecord]: ...

    def insert_thought(
        self,
        *,
        stm_id: int | None,
        reflection: str,
        devil_critique: str,
        refined_thought: str,
        intent: dict[str, Any],
    ) -> ThoughtRecord: ...

    def close(self) -> None: ...


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


class InMemoryShortTermMemory:
    """Process-local STM used for tests and when Postgres is unavailable."""

    def __init__(self, dim: int = TOTAL_EMBEDDING_DIM) -> None:
        self.dim = dim
        self._stm: list[StmRecord] = []
        self._thoughts: list[ThoughtRecord] = []
        self._next_stm = 1
        self._next_thought = 1

    def insert_embedding(
        self,
        *,
        embedding: np.ndarray,
        signals: dict[str, Any],
        affect: dict[str, Any],
        episode_id: str = "live",
        step: int = 0,
        sim_time: float = 0.0,
        fly_id: str = "fly_001",
    ) -> StmRecord:
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.dim:
            padded = np.zeros(self.dim, dtype=np.float32)
            n = min(self.dim, vec.shape[0])
            padded[:n] = vec[:n]
            vec = padded
        rec = StmRecord(
            id=self._next_stm,
            episode_id=episode_id,
            step=step,
            sim_time=sim_time,
            fly_id=fly_id,
            embedding=vec.copy(),
            signals=dict(signals),
            affect=dict(affect),
        )
        self._next_stm += 1
        self._stm.append(rec)
        return rec

    def similar(self, embedding: np.ndarray, *, fly_id: str, top_k: int = 5) -> list[StmRecord]:
        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        scored = [
            (_cosine_distance(query, r.embedding), r)
            for r in self._stm
            if r.fly_id == fly_id
        ]
        scored.sort(key=lambda t: t[0])
        return [r for _, r in scored[:top_k]]

    def recent(self, *, fly_id: str, limit: int = 8) -> list[StmRecord]:
        rows = [r for r in self._stm if r.fly_id == fly_id]
        return list(reversed(rows[-limit:]))

    def insert_thought(
        self,
        *,
        stm_id: int | None,
        reflection: str,
        devil_critique: str,
        refined_thought: str,
        intent: dict[str, Any],
    ) -> ThoughtRecord:
        rec = ThoughtRecord(
            id=self._next_thought,
            stm_id=stm_id,
            reflection=reflection,
            devil_critique=devil_critique,
            refined_thought=refined_thought,
            intent=dict(intent),
        )
        self._next_thought += 1
        self._thoughts.append(rec)
        return rec

    def close(self) -> None:
        return None


class PgVectorShortTermMemory:
    """Postgres + pgvector STM client."""

    def __init__(self, dsn: str, dim: int = TOTAL_EMBEDDING_DIM, schema: str = "fly_mind") -> None:
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Install mind extras: pip install -e '.[mind]'"
            ) from exc

        self.dim = dim
        self.schema = schema
        self._conn = psycopg.connect(dsn)
        register_vector(self._conn)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.stm_embeddings (
                    id BIGSERIAL PRIMARY KEY,
                    episode_id TEXT NOT NULL DEFAULT 'live',
                    step BIGINT NOT NULL DEFAULT 0,
                    sim_time DOUBLE PRECISION NOT NULL DEFAULT 0,
                    fly_id TEXT NOT NULL DEFAULT 'fly_001',
                    embedding vector({self.dim}) NOT NULL,
                    signals JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    affect JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.thoughts (
                    id BIGSERIAL PRIMARY KEY,
                    stm_id BIGINT,
                    reflection TEXT NOT NULL DEFAULT '',
                    devil_critique TEXT NOT NULL DEFAULT '',
                    refined_thought TEXT NOT NULL DEFAULT '',
                    intent JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
        self._conn.commit()

    def insert_embedding(
        self,
        *,
        embedding: np.ndarray,
        signals: dict[str, Any],
        affect: dict[str, Any],
        episode_id: str = "live",
        step: int = 0,
        sim_time: float = 0.0,
        fly_id: str = "fly_001",
    ) -> StmRecord:
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.dim:
            padded = np.zeros(self.dim, dtype=np.float32)
            n = min(self.dim, vec.shape[0])
            padded[:n] = vec[:n]
            vec = padded
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self.schema}.stm_embeddings
                    (episode_id, step, sim_time, fly_id, embedding, signals, affect)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                RETURNING id, created_at
                """,
                (
                    episode_id,
                    step,
                    sim_time,
                    fly_id,
                    vec.tolist(),
                    json.dumps(signals),
                    json.dumps(affect),
                ),
            )
            row = cur.fetchone()
        self._conn.commit()
        assert row is not None
        return StmRecord(
            id=int(row[0]),
            episode_id=episode_id,
            step=step,
            sim_time=sim_time,
            fly_id=fly_id,
            embedding=vec.copy(),
            signals=dict(signals),
            affect=dict(affect),
            created_at=row[1],
        )

    def similar(self, embedding: np.ndarray, *, fly_id: str, top_k: int = 5) -> list[StmRecord]:
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, episode_id, step, sim_time, fly_id, embedding, signals, affect, created_at
                FROM {self.schema}.stm_embeddings
                WHERE fly_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (fly_id, vec.tolist(), top_k),
            )
            rows = cur.fetchall()
        return [self._row_to_stm(r) for r in rows]

    def recent(self, *, fly_id: str, limit: int = 8) -> list[StmRecord]:
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, episode_id, step, sim_time, fly_id, embedding, signals, affect, created_at
                FROM {self.schema}.stm_embeddings
                WHERE fly_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (fly_id, limit),
            )
            rows = cur.fetchall()
        return [self._row_to_stm(r) for r in rows]

    def insert_thought(
        self,
        *,
        stm_id: int | None,
        reflection: str,
        devil_critique: str,
        refined_thought: str,
        intent: dict[str, Any],
    ) -> ThoughtRecord:
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self.schema}.thoughts
                    (stm_id, reflection, devil_critique, refined_thought, intent)
                VALUES (%s, %s, %s, %s, %s::jsonb)
                RETURNING id, created_at
                """,
                (
                    stm_id,
                    reflection,
                    devil_critique,
                    refined_thought,
                    json.dumps(intent),
                ),
            )
            row = cur.fetchone()
        self._conn.commit()
        assert row is not None
        return ThoughtRecord(
            id=int(row[0]),
            stm_id=stm_id,
            reflection=reflection,
            devil_critique=devil_critique,
            refined_thought=refined_thought,
            intent=dict(intent),
            created_at=row[1],
        )

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_stm(row: tuple[Any, ...]) -> StmRecord:
        emb = row[5]
        if hasattr(emb, "tolist"):
            emb = np.asarray(emb, dtype=np.float32)
        else:
            emb = np.asarray(emb, dtype=np.float32)
        signals = row[6] if isinstance(row[6], dict) else json.loads(row[6])
        affect = row[7] if isinstance(row[7], dict) else json.loads(row[7])
        return StmRecord(
            id=int(row[0]),
            episode_id=str(row[1]),
            step=int(row[2]),
            sim_time=float(row[3]),
            fly_id=str(row[4]),
            embedding=emb,
            signals=signals,
            affect=affect,
            created_at=row[8],
        )


def open_short_term_memory(
    *,
    backend: str = "in_memory",
    dsn: str | None = None,
    dim: int = TOTAL_EMBEDDING_DIM,
    schema: str = "fly_mind",
) -> ShortTermMemory:
    backend = (backend or "in_memory").lower()
    if backend == "pgvector":
        resolved = dsn or os.environ.get(
            "FLY_MIND_DSN", "postgresql://fly:fly@127.0.0.1:5433/fly_mind"
        )
        return PgVectorShortTermMemory(resolved, dim=dim, schema=schema)
    return InMemoryShortTermMemory(dim=dim)
