"""Short-term memory stores: in-memory and Postgres pgvector."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Protocol

import numpy as np

from flybrain.phase4 import (
    AFFECT_VEC_DIM,
    EYE_VEC_DIM,
    INTERRUPT_JOINT_VEC_DIM,
    PATCH_VEC_DIM,
    SENSORS_VEC_DIM,
    SMELL_VEC_DIM,
    STATUS_VEC_DIM,
    STM_MODALITIES,
    TOTAL_EMBEDDING_DIM,
)

Modality = Literal[
    "embedding",
    "left_eye",
    "right_eye",
    "sensors",
    "status",
    "affect",
    "patches",
    "smell",
    "joint",
]

_MODALITY_ATTR = {
    "embedding": "embedding",
    "left_eye": "left_eye_vec",
    "right_eye": "right_eye_vec",
    "sensors": "sensors_vec",
    "status": "status_vec",
    "affect": "affect_vec",
    "patches": "patch_vec",
    "smell": "smell_vec",
    "joint": "joint_vec",
}


def _json_default(obj: Any) -> Any:
    """Serialize numpy scalars/arrays for jsonb columns."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, default=_json_default)

_MODALITY_DIM = {
    "embedding": TOTAL_EMBEDDING_DIM,
    "left_eye": EYE_VEC_DIM,
    "right_eye": EYE_VEC_DIM,
    "sensors": SENSORS_VEC_DIM,
    "status": STATUS_VEC_DIM,
    "affect": AFFECT_VEC_DIM,
    "patches": PATCH_VEC_DIM,
    "smell": SMELL_VEC_DIM,
    "joint": INTERRUPT_JOINT_VEC_DIM,
}

# Modalities whose zero-norm rows are skipped during neighbor retrieval.
_SKIP_ZERO_NORM_MODALITIES = frozenset({"patches", "smell", "joint"})


def _pad_vec(vec: Any | None, dim: int) -> np.ndarray:
    """Coerce a numpy array / list / pgvector.Vector into a fixed-length float32 vector."""
    if vec is None:
        return np.zeros(dim, dtype=np.float32)
    # pgvector.Vector: to_numpy / to_list (not ndarray.tolist) — np.asarray(Vector) raises.
    if hasattr(vec, "to_numpy"):
        arr = np.asarray(vec.to_numpy(), dtype=np.float32).reshape(-1)
    elif hasattr(vec, "to_list"):
        arr = np.asarray(vec.to_list(), dtype=np.float32).reshape(-1)
    elif isinstance(vec, np.ndarray):
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    elif hasattr(vec, "tolist") and not isinstance(vec, (str, bytes)):
        arr = np.asarray(vec.tolist(), dtype=np.float32).reshape(-1)
    else:
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.shape[0] == dim:
        return arr.copy()
    out = np.zeros(dim, dtype=np.float32)
    n = min(dim, arr.shape[0])
    out[:n] = arr[:n]
    return out


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
    left_eye_vec: np.ndarray = field(default_factory=lambda: np.zeros(EYE_VEC_DIM, dtype=np.float32))
    right_eye_vec: np.ndarray = field(default_factory=lambda: np.zeros(EYE_VEC_DIM, dtype=np.float32))
    sensors_vec: np.ndarray = field(default_factory=lambda: np.zeros(SENSORS_VEC_DIM, dtype=np.float32))
    status_vec: np.ndarray = field(default_factory=lambda: np.zeros(STATUS_VEC_DIM, dtype=np.float32))
    affect_vec: np.ndarray = field(default_factory=lambda: np.zeros(AFFECT_VEC_DIM, dtype=np.float32))
    patch_vec: np.ndarray = field(default_factory=lambda: np.zeros(PATCH_VEC_DIM, dtype=np.float32))
    smell_vec: np.ndarray = field(default_factory=lambda: np.zeros(SMELL_VEC_DIM, dtype=np.float32))
    joint_vec: np.ndarray = field(
        default_factory=lambda: np.zeros(INTERRUPT_JOINT_VEC_DIM, dtype=np.float32)
    )
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
        left_eye_vec: np.ndarray | None = None,
        right_eye_vec: np.ndarray | None = None,
        sensors_vec: np.ndarray | None = None,
        status_vec: np.ndarray | None = None,
        affect_vec: np.ndarray | None = None,
        patch_vec: np.ndarray | None = None,
        smell_vec: np.ndarray | None = None,
        joint_vec: np.ndarray | None = None,
    ) -> StmRecord: ...

    def similar(self, embedding: np.ndarray, *, fly_id: str, top_k: int = 5) -> list[StmRecord]: ...

    def similar_by(
        self,
        query: np.ndarray,
        *,
        modality: Modality,
        fly_id: str,
        top_k: int = 5,
    ) -> list[StmRecord]: ...

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


def _modality_vector(rec: StmRecord, modality: Modality) -> np.ndarray:
    attr = _MODALITY_ATTR[modality]
    return np.asarray(getattr(rec, attr), dtype=np.float32).reshape(-1)


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
        left_eye_vec: np.ndarray | None = None,
        right_eye_vec: np.ndarray | None = None,
        sensors_vec: np.ndarray | None = None,
        status_vec: np.ndarray | None = None,
        affect_vec: np.ndarray | None = None,
        patch_vec: np.ndarray | None = None,
        smell_vec: np.ndarray | None = None,
        joint_vec: np.ndarray | None = None,
    ) -> StmRecord:
        rec = StmRecord(
            id=self._next_stm,
            episode_id=episode_id,
            step=step,
            sim_time=sim_time,
            fly_id=fly_id,
            embedding=_pad_vec(embedding, self.dim),
            signals=dict(signals),
            affect=dict(affect),
            left_eye_vec=_pad_vec(left_eye_vec, EYE_VEC_DIM),
            right_eye_vec=_pad_vec(right_eye_vec, EYE_VEC_DIM),
            sensors_vec=_pad_vec(sensors_vec, SENSORS_VEC_DIM),
            status_vec=_pad_vec(status_vec, STATUS_VEC_DIM),
            affect_vec=_pad_vec(affect_vec, AFFECT_VEC_DIM),
            patch_vec=_pad_vec(patch_vec, PATCH_VEC_DIM),
            smell_vec=_pad_vec(smell_vec, SMELL_VEC_DIM),
            joint_vec=_pad_vec(joint_vec, INTERRUPT_JOINT_VEC_DIM),
        )
        self._next_stm += 1
        self._stm.append(rec)
        return rec

    def similar(self, embedding: np.ndarray, *, fly_id: str, top_k: int = 5) -> list[StmRecord]:
        return self.similar_by(embedding, modality="embedding", fly_id=fly_id, top_k=top_k)

    def similar_by(
        self,
        query: np.ndarray,
        *,
        modality: Modality,
        fly_id: str,
        top_k: int = 5,
    ) -> list[StmRecord]:
        if modality not in STM_MODALITIES:
            raise ValueError(f"Unknown modality {modality!r}; expected one of {STM_MODALITIES}")
        q = _pad_vec(query, _MODALITY_DIM[modality])
        scored = []
        for r in self._stm:
            if r.fly_id != fly_id:
                continue
            vec = _modality_vector(r, modality)
            if modality in _SKIP_ZERO_NORM_MODALITIES and float(np.linalg.norm(vec)) < 1e-6:
                continue
            scored.append((_cosine_distance(q, vec), r))
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
    """Postgres + pgvector STM client writing fly_mind.short_term_memory."""

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
        self._table = f"{schema}.short_term_memory"
        self._conn = psycopg.connect(dsn)
        register_vector(self._conn)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id BIGSERIAL PRIMARY KEY,
                    episode_id TEXT NOT NULL DEFAULT 'live',
                    step BIGINT NOT NULL DEFAULT 0,
                    sim_time DOUBLE PRECISION NOT NULL DEFAULT 0,
                    fly_id TEXT NOT NULL DEFAULT 'fly_001',
                    left_eye_vec vector({EYE_VEC_DIM}) NOT NULL,
                    right_eye_vec vector({EYE_VEC_DIM}) NOT NULL,
                    sensors_vec vector({SENSORS_VEC_DIM}) NOT NULL,
                    status_vec vector({STATUS_VEC_DIM}) NOT NULL,
                    affect_vec vector({AFFECT_VEC_DIM}) NOT NULL,
                    patch_vec vector({PATCH_VEC_DIM}),
                    smell_vec vector({SMELL_VEC_DIM}),
                    joint_vec vector({INTERRUPT_JOINT_VEC_DIM}),
                    embedding vector({self.dim}) NOT NULL,
                    signals JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    affect JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            # Migrate older tables that predate patch_vec / smell_vec / joint_vec.
            cur.execute(
                f"""
                ALTER TABLE {self._table}
                ADD COLUMN IF NOT EXISTS patch_vec vector({PATCH_VEC_DIM})
                """
            )
            cur.execute(
                f"""
                ALTER TABLE {self._table}
                ADD COLUMN IF NOT EXISTS smell_vec vector({SMELL_VEC_DIM})
                """
            )
            cur.execute(
                f"""
                ALTER TABLE {self._table}
                ADD COLUMN IF NOT EXISTS joint_vec vector({INTERRUPT_JOINT_VEC_DIM})
                """
            )
            # Cosine indexes for fused recall + eye revisit + interrupt modalities.
            for col in (
                "embedding",
                "left_eye_vec",
                "right_eye_vec",
                "patch_vec",
                "smell_vec",
                "joint_vec",
            ):
                idx = f"stm_{col}_hnsw"
                cur.execute(
                    f"""
                    DO $$ BEGIN
                        CREATE INDEX IF NOT EXISTS {idx}
                        ON {self._table}
                        USING hnsw ({col} vector_cosine_ops);
                    EXCEPTION WHEN OTHERS THEN
                        NULL;  -- older pgvector without hnsw: skip
                    END $$;
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
        left_eye_vec: np.ndarray | None = None,
        right_eye_vec: np.ndarray | None = None,
        sensors_vec: np.ndarray | None = None,
        status_vec: np.ndarray | None = None,
        affect_vec: np.ndarray | None = None,
        patch_vec: np.ndarray | None = None,
        smell_vec: np.ndarray | None = None,
        joint_vec: np.ndarray | None = None,
    ) -> StmRecord:
        emb = _pad_vec(embedding, self.dim)
        left = _pad_vec(left_eye_vec, EYE_VEC_DIM)
        right = _pad_vec(right_eye_vec, EYE_VEC_DIM)
        sensors = _pad_vec(sensors_vec, SENSORS_VEC_DIM)
        status = _pad_vec(status_vec, STATUS_VEC_DIM)
        aff_v = _pad_vec(affect_vec, AFFECT_VEC_DIM)
        patches = _pad_vec(patch_vec, PATCH_VEC_DIM)
        smell = _pad_vec(smell_vec, SMELL_VEC_DIM)
        joint = _pad_vec(joint_vec, INTERRUPT_JOINT_VEC_DIM)
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self._table}
                    (episode_id, step, sim_time, fly_id,
                     left_eye_vec, right_eye_vec, sensors_vec, status_vec, affect_vec,
                     patch_vec, smell_vec, joint_vec, embedding, signals, affect)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                RETURNING id, created_at
                """,
                (
                    episode_id,
                    step,
                    sim_time,
                    fly_id,
                    left.tolist(),
                    right.tolist(),
                    sensors.tolist(),
                    status.tolist(),
                    aff_v.tolist(),
                    patches.tolist(),
                    smell.tolist(),
                    joint.tolist(),
                    emb.tolist(),
                    _json_dumps(signals),
                    _json_dumps(affect),
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
            embedding=emb,
            signals=dict(signals),
            affect=dict(affect),
            left_eye_vec=left,
            right_eye_vec=right,
            sensors_vec=sensors,
            status_vec=status,
            affect_vec=aff_v,
            patch_vec=patches,
            smell_vec=smell,
            joint_vec=joint,
            created_at=row[1],
        )

    def similar(self, embedding: np.ndarray, *, fly_id: str, top_k: int = 5) -> list[StmRecord]:
        return self.similar_by(embedding, modality="embedding", fly_id=fly_id, top_k=top_k)

    def similar_by(
        self,
        query: np.ndarray,
        *,
        modality: Modality,
        fly_id: str,
        top_k: int = 5,
    ) -> list[StmRecord]:
        if modality not in _MODALITY_ATTR:
            raise ValueError(f"Unknown modality {modality!r}; expected one of {STM_MODALITIES}")
        col = _MODALITY_ATTR[modality]
        vec = _pad_vec(query, _MODALITY_DIM[modality])
        select_cols = (
            "id, episode_id, step, sim_time, fly_id, "
            "left_eye_vec, right_eye_vec, sensors_vec, status_vec, affect_vec, "
            "patch_vec, smell_vec, joint_vec, embedding, signals, affect, created_at"
        )
        with self._conn.cursor() as cur:
            if modality in _SKIP_ZERO_NORM_MODALITIES:
                cur.execute(
                    f"""
                    SELECT {select_cols}
                    FROM {self._table}
                    WHERE fly_id = %s
                      AND {col} IS NOT NULL
                    ORDER BY {col} <=> %s::vector
                    LIMIT %s
                    """,
                    (fly_id, vec.tolist(), top_k * 4),
                )
                rows = cur.fetchall()
                kept = []
                dim = _MODALITY_DIM[modality]
                # Column index in select_cols: patch=10, smell=11, joint=12
                col_idx = {"patches": 10, "smell": 11, "joint": 12}[modality]
                for r in rows:
                    pv = self._as_f32(r[col_idx], dim)
                    if float(np.linalg.norm(pv)) < 1e-6:
                        continue
                    kept.append(r)
                    if len(kept) >= top_k:
                        break
                return [self._row_to_stm(r) for r in kept]
            cur.execute(
                f"""
                SELECT {select_cols}
                FROM {self._table}
                WHERE fly_id = %s
                ORDER BY {col} <=> %s::vector
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
                SELECT id, episode_id, step, sim_time, fly_id,
                       left_eye_vec, right_eye_vec, sensors_vec, status_vec, affect_vec,
                       patch_vec, smell_vec, joint_vec, embedding, signals, affect, created_at
                FROM {self._table}
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
                    _json_dumps(intent),
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
    def _as_f32(val: Any, dim: int) -> np.ndarray:
        return _pad_vec(val, dim)

    @classmethod
    def _row_to_stm(cls, row: tuple[Any, ...]) -> StmRecord:
        # 17 cols: … patch, smell, joint, embedding, signals, affect, created_at
        # 16 cols: … patch, smell, embedding, signals, affect, created_at (pre-joint)
        # 15 cols: … patch, embedding, signals, affect, created_at (pre-smell)
        # 14 cols: … embedding, signals, affect, created_at (pre-patch)
        if len(row) >= 17:
            signals = row[14] if isinstance(row[14], dict) else json.loads(row[14])
            affect = row[15] if isinstance(row[15], dict) else json.loads(row[15])
            return StmRecord(
                id=int(row[0]),
                episode_id=str(row[1]),
                step=int(row[2]),
                sim_time=float(row[3]),
                fly_id=str(row[4]),
                left_eye_vec=cls._as_f32(row[5], EYE_VEC_DIM),
                right_eye_vec=cls._as_f32(row[6], EYE_VEC_DIM),
                sensors_vec=cls._as_f32(row[7], SENSORS_VEC_DIM),
                status_vec=cls._as_f32(row[8], STATUS_VEC_DIM),
                affect_vec=cls._as_f32(row[9], AFFECT_VEC_DIM),
                patch_vec=cls._as_f32(row[10], PATCH_VEC_DIM),
                smell_vec=cls._as_f32(row[11], SMELL_VEC_DIM),
                joint_vec=cls._as_f32(row[12], INTERRUPT_JOINT_VEC_DIM),
                embedding=cls._as_f32(row[13], TOTAL_EMBEDDING_DIM),
                signals=signals,
                affect=affect,
                created_at=row[16],
            )
        if len(row) >= 16:
            signals = row[13] if isinstance(row[13], dict) else json.loads(row[13])
            affect = row[14] if isinstance(row[14], dict) else json.loads(row[14])
            return StmRecord(
                id=int(row[0]),
                episode_id=str(row[1]),
                step=int(row[2]),
                sim_time=float(row[3]),
                fly_id=str(row[4]),
                left_eye_vec=cls._as_f32(row[5], EYE_VEC_DIM),
                right_eye_vec=cls._as_f32(row[6], EYE_VEC_DIM),
                sensors_vec=cls._as_f32(row[7], SENSORS_VEC_DIM),
                status_vec=cls._as_f32(row[8], STATUS_VEC_DIM),
                affect_vec=cls._as_f32(row[9], AFFECT_VEC_DIM),
                patch_vec=cls._as_f32(row[10], PATCH_VEC_DIM),
                smell_vec=cls._as_f32(row[11], SMELL_VEC_DIM),
                embedding=cls._as_f32(row[12], TOTAL_EMBEDDING_DIM),
                signals=signals,
                affect=affect,
                created_at=row[15],
            )
        if len(row) >= 15:
            signals = row[12] if isinstance(row[12], dict) else json.loads(row[12])
            affect = row[13] if isinstance(row[13], dict) else json.loads(row[13])
            return StmRecord(
                id=int(row[0]),
                episode_id=str(row[1]),
                step=int(row[2]),
                sim_time=float(row[3]),
                fly_id=str(row[4]),
                left_eye_vec=cls._as_f32(row[5], EYE_VEC_DIM),
                right_eye_vec=cls._as_f32(row[6], EYE_VEC_DIM),
                sensors_vec=cls._as_f32(row[7], SENSORS_VEC_DIM),
                status_vec=cls._as_f32(row[8], STATUS_VEC_DIM),
                affect_vec=cls._as_f32(row[9], AFFECT_VEC_DIM),
                patch_vec=cls._as_f32(row[10], PATCH_VEC_DIM),
                embedding=cls._as_f32(row[11], TOTAL_EMBEDDING_DIM),
                signals=signals,
                affect=affect,
                created_at=row[14],
            )
        signals = row[11] if isinstance(row[11], dict) else json.loads(row[11])
        affect = row[12] if isinstance(row[12], dict) else json.loads(row[12])
        return StmRecord(
            id=int(row[0]),
            episode_id=str(row[1]),
            step=int(row[2]),
            sim_time=float(row[3]),
            fly_id=str(row[4]),
            left_eye_vec=cls._as_f32(row[5], EYE_VEC_DIM),
            right_eye_vec=cls._as_f32(row[6], EYE_VEC_DIM),
            sensors_vec=cls._as_f32(row[7], SENSORS_VEC_DIM),
            status_vec=cls._as_f32(row[8], STATUS_VEC_DIM),
            affect_vec=cls._as_f32(row[9], AFFECT_VEC_DIM),
            embedding=cls._as_f32(row[10], TOTAL_EMBEDDING_DIM),
            signals=signals,
            affect=affect,
            created_at=row[13],
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
