"""Grounded phrases on the skill Neo4j cluster (Bolt 7688).

Maps simultaneous eye embedding + heard spectrogram + fly alphabet to an
LLM-assigned single/multi-word meaning, stored for later image+sound recall.
Motor ``:Skill`` nodes stay separate; this uses ``:GroundedPhrase`` in the
same skill database.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import uuid4

import numpy as np


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp01(value: float, default: float = 0.0) -> float:
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except (TypeError, ValueError):
        return float(default)


def cosine_sim(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """Cosine similarity in [0, 1] (negative → 0)."""
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(x.size, y.size)
    if n <= 0:
        return 0.0
    x = x[:n]
    y = y[:n]
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx < 1e-9 or ny < 1e-9:
        return 0.0
    return float(np.clip(float(np.dot(x, y) / (nx * ny)), 0.0, 1.0))


def flatten_spectrogram(frames: list[list[float]] | None) -> list[float]:
    out: list[float] = []
    for row in frames or []:
        out.extend(float(x) for x in row)
    return out


@dataclass
class GroundedPhraseConfig:
    enabled: bool = False
    # Same Neo4j as skill cortex by default.
    backend: str = "in_memory"
    uri: str = "bolt://127.0.0.1:7688"
    user: str = "neo4j"
    password: str | None = None
    database: str = "neo4j"
    mock: bool | None = None  # None → inherit visual_thinking.mock
    use_vision: bool = True
    min_confidence: float = 0.35
    match_patch_min: float = 0.82
    match_audio_min: float = 0.55
    match_combined_min: float = 0.7
    cooldown_steps: int = 40
    label_timeout_seconds: float = 30.0
    reinforce_speech_words: bool = True
    word_weight_delta: float = 0.08
    # Sentence compose: top phrases + drives → LLM short sentence → :GroundedSentence
    sentence_compose: bool = True
    sentence_min_phrases: int = 2
    sentence_top_k: int = 4
    sentence_match_combined_min: float = 0.55
    sentence_cooldown_steps: int = 80
    sentence_min_confidence: float = 0.35

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> GroundedPhraseConfig:
        data = dict(raw or {})
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class GroundedPhrase:
    phrase_id: str
    phrase: str
    alphabet: str = ""
    fly_id: str = "fly_001"
    source: str = "hear"  # hear | babble
    patch_vec: list[float] = field(default_factory=list)
    spectrogram: list[list[float]] = field(default_factory=list)
    rms: float = 0.0
    voiced: float = 0.0
    confidence: float = 0.0
    score: float = 0.0
    hits: int = 0
    utterance_id: str | None = None
    place_id: str | None = None
    step: int = 0
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        if not self.phrase_id:
            self.phrase_id = str(uuid4())
        self.phrase = " ".join(str(self.phrase or "").lower().split())
        self.alphabet = str(self.alphabet or "")
        self.confidence = _clamp01(self.confidence)
        self.score = _clamp01(self.score)
        self.hits = max(0, int(self.hits))
        self.rms = _clamp01(self.rms)
        self.voiced = _clamp01(self.voiced)
        if not self.created_at:
            self.created_at = _utcnow()
        if not self.updated_at:
            self.updated_at = self.created_at

    def audio_vec(self) -> list[float]:
        return flatten_spectrogram(self.spectrogram)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phrase_id": self.phrase_id,
            "phrase": self.phrase,
            "alphabet": self.alphabet,
            "fly_id": self.fly_id,
            "source": self.source,
            "patch_vec": [float(x) for x in self.patch_vec],
            "spectrogram": [list(r) for r in self.spectrogram],
            "rms": float(self.rms),
            "voiced": float(self.voiced),
            "confidence": float(self.confidence),
            "score": float(self.score),
            "hits": int(self.hits),
            "utterance_id": self.utterance_id,
            "place_id": self.place_id,
            "step": int(self.step),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> GroundedPhrase:
        data = dict(raw or {})
        return cls(
            phrase_id=str(data.get("phrase_id") or uuid4()),
            phrase=str(data.get("phrase") or ""),
            alphabet=str(data.get("alphabet") or ""),
            fly_id=str(data.get("fly_id") or "fly_001"),
            source=str(data.get("source") or "hear"),
            patch_vec=[float(x) for x in (data.get("patch_vec") or [])],
            spectrogram=[list(r) for r in (data.get("spectrogram") or [])],
            rms=float(data.get("rms") or 0.0),
            voiced=float(data.get("voiced") or 0.0),
            confidence=float(data.get("confidence") or 0.0),
            score=float(data.get("score") or 0.0),
            hits=int(data.get("hits") or 0),
            utterance_id=(
                str(data["utterance_id"]) if data.get("utterance_id") is not None else None
            ),
            place_id=(str(data["place_id"]) if data.get("place_id") is not None else None),
            step=int(data.get("step") or 0),
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
        )


@dataclass
class PhraseMatch:
    phrase: GroundedPhrase
    patch_sim: float
    audio_sim: float
    combined: float


@dataclass
class GroundedSentence:
    """LLM-composed sentence from recalled phrases + drives (skill Neo4j)."""

    sentence_id: str
    sentence: str
    phrases: list[str] = field(default_factory=list)
    phrase_ids: list[str] = field(default_factory=list)
    fly_id: str = "fly_001"
    drive: dict[str, float] = field(default_factory=dict)
    patch_vec: list[float] = field(default_factory=list)
    spectrogram: list[list[float]] = field(default_factory=list)
    confidence: float = 0.0
    score: float = 0.0
    hits: int = 0
    place_id: str | None = None
    step: int = 0
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        if not self.sentence_id:
            self.sentence_id = str(uuid4())
        self.sentence = " ".join(str(self.sentence or "").split())
        self.phrases = [" ".join(str(p).lower().split()) for p in self.phrases if str(p).strip()]
        self.phrase_ids = [str(x) for x in self.phrase_ids if str(x).strip()]
        self.confidence = _clamp01(self.confidence)
        self.score = _clamp01(self.score)
        self.hits = max(0, int(self.hits))
        cleaned: dict[str, float] = {}
        for key, val in dict(self.drive or {}).items():
            try:
                cleaned[str(key)] = _clamp01(float(val))
            except (TypeError, ValueError):
                continue
        self.drive = cleaned
        if not self.created_at:
            self.created_at = _utcnow()
        if not self.updated_at:
            self.updated_at = self.created_at

    def audio_vec(self) -> list[float]:
        return flatten_spectrogram(self.spectrogram)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentence_id": self.sentence_id,
            "sentence": self.sentence,
            "phrases": list(self.phrases),
            "phrase_ids": list(self.phrase_ids),
            "fly_id": self.fly_id,
            "drive": {k: float(v) for k, v in self.drive.items()},
            "patch_vec": [float(x) for x in self.patch_vec],
            "spectrogram": [list(r) for r in self.spectrogram],
            "confidence": float(self.confidence),
            "score": float(self.score),
            "hits": int(self.hits),
            "place_id": self.place_id,
            "step": int(self.step),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> GroundedSentence:
        data = dict(raw or {})
        drive_raw = data.get("drive") or {}
        if not isinstance(drive_raw, dict):
            drive_raw = {}
        return cls(
            sentence_id=str(data.get("sentence_id") or uuid4()),
            sentence=str(data.get("sentence") or ""),
            phrases=[str(x) for x in (data.get("phrases") or [])],
            phrase_ids=[str(x) for x in (data.get("phrase_ids") or [])],
            fly_id=str(data.get("fly_id") or "fly_001"),
            drive={str(k): float(v) for k, v in drive_raw.items()},
            patch_vec=[float(x) for x in (data.get("patch_vec") or [])],
            spectrogram=[list(r) for r in (data.get("spectrogram") or [])],
            confidence=float(data.get("confidence") or 0.0),
            score=float(data.get("score") or 0.0),
            hits=int(data.get("hits") or 0),
            place_id=(str(data["place_id"]) if data.get("place_id") is not None else None),
            step=int(data.get("step") or 0),
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
        )


@dataclass
class SentenceMatch:
    sentence: GroundedSentence
    patch_sim: float
    audio_sim: float
    combined: float


class GroundedPhraseStore(Protocol):
    def upsert_phrase(self, phrase: GroundedPhrase) -> GroundedPhrase: ...

    def get_phrase(self, phrase_id: str) -> GroundedPhrase | None: ...

    def list_phrases(
        self, *, fly_id: str = "fly_001", limit: int = 128
    ) -> list[GroundedPhrase]: ...

    def bump_hit(self, phrase_id: str, *, delta_score: float = 0.05) -> GroundedPhrase | None: ...

    def upsert_sentence(self, sentence: GroundedSentence) -> GroundedSentence: ...

    def get_sentence(self, sentence_id: str) -> GroundedSentence | None: ...

    def list_sentences(
        self, *, fly_id: str = "fly_001", limit: int = 64
    ) -> list[GroundedSentence]: ...

    def bump_sentence_hit(
        self, sentence_id: str, *, delta_score: float = 0.05
    ) -> GroundedSentence | None: ...

    def close(self) -> None: ...


def _combined_sim(
    *,
    patch_vec: list[float] | np.ndarray,
    audio: list[float],
    row_patch: list[float],
    row_audio: list[float],
) -> tuple[float, float, float]:
    p_sim = cosine_sim(patch_vec, row_patch)
    a_sim = cosine_sim(audio, row_audio) if audio and row_audio else 0.0
    if audio and row_audio:
        combined = 0.6 * p_sim + 0.4 * a_sim
    else:
        combined = p_sim
    return p_sim, a_sim, combined


def match_top_grounded_phrases(
    store: GroundedPhraseStore,
    *,
    fly_id: str,
    patch_vec: list[float] | np.ndarray,
    spectrogram: list[list[float]] | None,
    config: GroundedPhraseConfig,
    limit: int = 64,
    top_k: int | None = None,
    combined_min: float | None = None,
) -> list[PhraseMatch]:
    """Rank stored phrases by image+audio similarity; return top-k above floor."""
    audio = flatten_spectrogram(spectrogram)
    floor = float(config.sentence_match_combined_min if combined_min is None else combined_min)
    k = max(1, int(config.sentence_top_k if top_k is None else top_k))
    scored: list[PhraseMatch] = []
    for row in store.list_phrases(fly_id=fly_id, limit=limit):
        if not row.phrase:
            continue
        p_sim, a_sim, combined = _combined_sim(
            patch_vec=patch_vec,
            audio=audio,
            row_patch=row.patch_vec,
            row_audio=row.audio_vec(),
        )
        if combined < floor:
            continue
        scored.append(PhraseMatch(phrase=row, patch_sim=p_sim, audio_sim=a_sim, combined=combined))
    scored.sort(key=lambda m: m.combined, reverse=True)
    return scored[:k]


def match_grounded_phrase(
    store: GroundedPhraseStore,
    *,
    fly_id: str,
    patch_vec: list[float] | np.ndarray,
    spectrogram: list[list[float]] | None,
    config: GroundedPhraseConfig,
    limit: int = 64,
) -> PhraseMatch | None:
    """Nearest stored phrase by combined image + audio cosine similarity."""
    audio = flatten_spectrogram(spectrogram)
    best: PhraseMatch | None = None
    for row in store.list_phrases(fly_id=fly_id, limit=limit):
        if not row.phrase:
            continue
        p_sim, a_sim, combined = _combined_sim(
            patch_vec=patch_vec,
            audio=audio,
            row_patch=row.patch_vec,
            row_audio=row.audio_vec(),
        )
        if p_sim < float(config.match_patch_min) and a_sim < float(config.match_audio_min):
            continue
        if combined < float(config.match_combined_min):
            continue
        if best is None or combined > best.combined:
            best = PhraseMatch(phrase=row, patch_sim=p_sim, audio_sim=a_sim, combined=combined)
    return best


def match_grounded_sentence(
    store: GroundedPhraseStore,
    *,
    fly_id: str,
    patch_vec: list[float] | np.ndarray,
    spectrogram: list[list[float]] | None,
    config: GroundedPhraseConfig,
    limit: int = 64,
) -> SentenceMatch | None:
    """Nearest stored sentence by combined image + audio cosine similarity."""
    audio = flatten_spectrogram(spectrogram)
    best: SentenceMatch | None = None
    for row in store.list_sentences(fly_id=fly_id, limit=limit):
        if not row.sentence:
            continue
        p_sim, a_sim, combined = _combined_sim(
            patch_vec=patch_vec,
            audio=audio,
            row_patch=row.patch_vec,
            row_audio=row.audio_vec(),
        )
        if p_sim < float(config.match_patch_min) and a_sim < float(config.match_audio_min):
            continue
        if combined < float(config.match_combined_min):
            continue
        if best is None or combined > best.combined:
            best = SentenceMatch(sentence=row, patch_sim=p_sim, audio_sim=a_sim, combined=combined)
    return best


@dataclass
class InMemoryGroundedPhraseStore:
    _rows: dict[str, GroundedPhrase] = field(default_factory=dict)
    _sentences: dict[str, GroundedSentence] = field(default_factory=dict)

    def upsert_phrase(self, phrase: GroundedPhrase) -> GroundedPhrase:
        row = GroundedPhrase.from_dict(phrase.to_dict())
        if not row.phrase_id:
            row.phrase_id = str(uuid4())
        row.updated_at = _utcnow()
        self._rows[row.phrase_id] = row
        return GroundedPhrase.from_dict(row.to_dict())

    def get_phrase(self, phrase_id: str) -> GroundedPhrase | None:
        row = self._rows.get(str(phrase_id))
        return GroundedPhrase.from_dict(row.to_dict()) if row else None

    def list_phrases(self, *, fly_id: str = "fly_001", limit: int = 128) -> list[GroundedPhrase]:
        rows = [
            GroundedPhrase.from_dict(r.to_dict())
            for r in self._rows.values()
            if str(r.fly_id) == str(fly_id) and r.phrase
        ]
        rows.sort(key=lambda r: (r.score, r.hits, r.updated_at), reverse=True)
        return rows[: max(1, int(limit))]

    def bump_hit(self, phrase_id: str, *, delta_score: float = 0.05) -> GroundedPhrase | None:
        row = self._rows.get(str(phrase_id))
        if row is None:
            return None
        row.hits = int(row.hits) + 1
        row.score = _clamp01(float(row.score) + float(delta_score))
        row.updated_at = _utcnow()
        return GroundedPhrase.from_dict(row.to_dict())

    def upsert_sentence(self, sentence: GroundedSentence) -> GroundedSentence:
        row = GroundedSentence.from_dict(sentence.to_dict())
        if not row.sentence_id:
            row.sentence_id = str(uuid4())
        row.updated_at = _utcnow()
        self._sentences[row.sentence_id] = row
        return GroundedSentence.from_dict(row.to_dict())

    def get_sentence(self, sentence_id: str) -> GroundedSentence | None:
        row = self._sentences.get(str(sentence_id))
        return GroundedSentence.from_dict(row.to_dict()) if row else None

    def list_sentences(self, *, fly_id: str = "fly_001", limit: int = 64) -> list[GroundedSentence]:
        rows = [
            GroundedSentence.from_dict(r.to_dict())
            for r in self._sentences.values()
            if str(r.fly_id) == str(fly_id) and r.sentence
        ]
        rows.sort(key=lambda r: (r.score, r.hits, r.updated_at), reverse=True)
        return rows[: max(1, int(limit))]

    def bump_sentence_hit(
        self, sentence_id: str, *, delta_score: float = 0.05
    ) -> GroundedSentence | None:
        row = self._sentences.get(str(sentence_id))
        if row is None:
            return None
        row.hits = int(row.hits) + 1
        row.score = _clamp01(float(row.score) + float(delta_score))
        row.updated_at = _utcnow()
        return GroundedSentence.from_dict(row.to_dict())

    def close(self) -> None:
        return None


@dataclass
class Neo4jGroundedPhraseStore:
    """``:GroundedPhrase`` nodes on the skill Neo4j partition."""

    uri: str = "bolt://localhost:7688"
    user: str = "neo4j"
    password: str = "flymindskill"
    database: str = "neo4j"
    _driver: Any = field(default=None, init=False, repr=False)

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

    def _query(self, cypher: str, parameters: dict[str, Any] | None = None) -> list[Any]:
        if self._driver is None:
            raise RuntimeError("Neo4j grounded-phrase store is closed")
        records, _summary, _keys = self._driver.execute_query(
            cypher,
            parameters_=parameters or {},
            database_=self.database,
        )
        return list(records)

    def _ensure_schema(self) -> None:
        self._query("""
            CREATE CONSTRAINT grounded_phrase_id_unique IF NOT EXISTS
            FOR (p:GroundedPhrase) REQUIRE p.phrase_id IS UNIQUE
            """)
        self._query("""
            CREATE INDEX grounded_phrase_fly IF NOT EXISTS
            FOR (p:GroundedPhrase) ON (p.fly_id, p.updated_at)
            """)
        self._query("""
            CREATE CONSTRAINT grounded_sentence_id_unique IF NOT EXISTS
            FOR (s:GroundedSentence) REQUIRE s.sentence_id IS UNIQUE
            """)
        self._query("""
            CREATE INDEX grounded_sentence_fly IF NOT EXISTS
            FOR (s:GroundedSentence) ON (s.fly_id, s.updated_at)
            """)

    def upsert_phrase(self, phrase: GroundedPhrase) -> GroundedPhrase:
        row = GroundedPhrase.from_dict(phrase.to_dict())
        if not row.phrase_id:
            row.phrase_id = str(uuid4())
        now = _utcnow()
        if not row.created_at:
            row.created_at = now
        row.updated_at = now
        self._query(
            """
            MERGE (p:GroundedPhrase {phrase_id: $phrase_id})
            SET p.phrase = $phrase,
                p.alphabet = $alphabet,
                p.fly_id = $fly_id,
                p.source = $source,
                p.patch_vec_json = $patch_vec_json,
                p.spectrogram_json = $spectrogram_json,
                p.rms = $rms,
                p.voiced = $voiced,
                p.confidence = $confidence,
                p.score = $score,
                p.hits = $hits,
                p.utterance_id = $utterance_id,
                p.place_id = $place_id,
                p.step = $step,
                p.created_at = coalesce(p.created_at, $created_at),
                p.updated_at = $updated_at
            """,
            {
                "phrase_id": row.phrase_id,
                "phrase": row.phrase,
                "alphabet": row.alphabet,
                "fly_id": row.fly_id,
                "source": row.source,
                "patch_vec_json": json.dumps(list(row.patch_vec)),
                "spectrogram_json": json.dumps(row.spectrogram),
                "rms": float(row.rms),
                "voiced": float(row.voiced),
                "confidence": float(row.confidence),
                "score": float(row.score),
                "hits": int(row.hits),
                "utterance_id": row.utterance_id,
                "place_id": row.place_id,
                "step": int(row.step),
                "created_at": row.created_at,
                "updated_at": row.updated_at,
            },
        )
        return row

    def get_phrase(self, phrase_id: str) -> GroundedPhrase | None:
        rows = self._query(
            "MATCH (p:GroundedPhrase {phrase_id: $phrase_id}) RETURN p",
            {"phrase_id": str(phrase_id)},
        )
        if not rows:
            return None
        return self._node_to_phrase(rows[0]["p"])

    def list_phrases(self, *, fly_id: str = "fly_001", limit: int = 128) -> list[GroundedPhrase]:
        rows = self._query(
            """
            MATCH (p:GroundedPhrase {fly_id: $fly_id})
            RETURN p
            ORDER BY p.score DESC, p.hits DESC, coalesce(p.updated_at, '') DESC
            LIMIT $limit
            """,
            {"fly_id": str(fly_id), "limit": int(limit)},
        )
        return [self._node_to_phrase(r["p"]) for r in rows]

    def bump_hit(self, phrase_id: str, *, delta_score: float = 0.05) -> GroundedPhrase | None:
        rows = self._query(
            """
            MATCH (p:GroundedPhrase {phrase_id: $phrase_id})
            SET p.hits = coalesce(p.hits, 0) + 1,
                p.score = CASE
                    WHEN coalesce(p.score, 0.0) + $delta > 1.0 THEN 1.0
                    WHEN coalesce(p.score, 0.0) + $delta < 0.0 THEN 0.0
                    ELSE coalesce(p.score, 0.0) + $delta
                END,
                p.updated_at = $now
            RETURN p
            """,
            {
                "phrase_id": str(phrase_id),
                "delta": float(delta_score),
                "now": _utcnow(),
            },
        )
        if not rows:
            return None
        return self._node_to_phrase(rows[0]["p"])

    def upsert_sentence(self, sentence: GroundedSentence) -> GroundedSentence:
        row = GroundedSentence.from_dict(sentence.to_dict())
        if not row.sentence_id:
            row.sentence_id = str(uuid4())
        now = _utcnow()
        if not row.created_at:
            row.created_at = now
        row.updated_at = now
        self._query(
            """
            MERGE (s:GroundedSentence {sentence_id: $sentence_id})
            SET s.sentence = $sentence,
                s.phrases_json = $phrases_json,
                s.phrase_ids_json = $phrase_ids_json,
                s.fly_id = $fly_id,
                s.drive_json = $drive_json,
                s.patch_vec_json = $patch_vec_json,
                s.spectrogram_json = $spectrogram_json,
                s.confidence = $confidence,
                s.score = $score,
                s.hits = $hits,
                s.place_id = $place_id,
                s.step = $step,
                s.created_at = coalesce(s.created_at, $created_at),
                s.updated_at = $updated_at
            """,
            {
                "sentence_id": row.sentence_id,
                "sentence": row.sentence,
                "phrases_json": json.dumps(list(row.phrases)),
                "phrase_ids_json": json.dumps(list(row.phrase_ids)),
                "fly_id": row.fly_id,
                "drive_json": json.dumps(row.drive),
                "patch_vec_json": json.dumps(list(row.patch_vec)),
                "spectrogram_json": json.dumps(row.spectrogram),
                "confidence": float(row.confidence),
                "score": float(row.score),
                "hits": int(row.hits),
                "place_id": row.place_id,
                "step": int(row.step),
                "created_at": row.created_at,
                "updated_at": row.updated_at,
            },
        )
        return row

    def get_sentence(self, sentence_id: str) -> GroundedSentence | None:
        rows = self._query(
            "MATCH (s:GroundedSentence {sentence_id: $sentence_id}) RETURN s",
            {"sentence_id": str(sentence_id)},
        )
        if not rows:
            return None
        return self._node_to_sentence(rows[0]["s"])

    def list_sentences(self, *, fly_id: str = "fly_001", limit: int = 64) -> list[GroundedSentence]:
        rows = self._query(
            """
            MATCH (s:GroundedSentence {fly_id: $fly_id})
            RETURN s
            ORDER BY s.score DESC, s.hits DESC, coalesce(s.updated_at, '') DESC
            LIMIT $limit
            """,
            {"fly_id": str(fly_id), "limit": int(limit)},
        )
        return [self._node_to_sentence(r["s"]) for r in rows]

    def bump_sentence_hit(
        self, sentence_id: str, *, delta_score: float = 0.05
    ) -> GroundedSentence | None:
        rows = self._query(
            """
            MATCH (s:GroundedSentence {sentence_id: $sentence_id})
            SET s.hits = coalesce(s.hits, 0) + 1,
                s.score = CASE
                    WHEN coalesce(s.score, 0.0) + $delta > 1.0 THEN 1.0
                    WHEN coalesce(s.score, 0.0) + $delta < 0.0 THEN 0.0
                    ELSE coalesce(s.score, 0.0) + $delta
                END,
                s.updated_at = $now
            RETURN s
            """,
            {
                "sentence_id": str(sentence_id),
                "delta": float(delta_score),
                "now": _utcnow(),
            },
        )
        if not rows:
            return None
        return self._node_to_sentence(rows[0]["s"])

    def _node_to_phrase(self, node: Any) -> GroundedPhrase:
        try:
            patch = json.loads(node.get("patch_vec_json") or "[]")
        except json.JSONDecodeError:
            patch = []
        try:
            spec = json.loads(node.get("spectrogram_json") or "[]")
        except json.JSONDecodeError:
            spec = []
        return GroundedPhrase(
            phrase_id=str(node.get("phrase_id") or uuid4()),
            phrase=str(node.get("phrase") or ""),
            alphabet=str(node.get("alphabet") or ""),
            fly_id=str(node.get("fly_id") or "fly_001"),
            source=str(node.get("source") or "hear"),
            patch_vec=[float(x) for x in patch],
            spectrogram=spec,
            rms=float(node.get("rms") or 0.0),
            voiced=float(node.get("voiced") or 0.0),
            confidence=float(node.get("confidence") or 0.0),
            score=float(node.get("score") or 0.0),
            hits=int(node.get("hits") or 0),
            utterance_id=node.get("utterance_id"),
            place_id=node.get("place_id"),
            step=int(node.get("step") or 0),
            created_at=str(node.get("created_at") or ""),
            updated_at=str(node.get("updated_at") or ""),
        )

    def _node_to_sentence(self, node: Any) -> GroundedSentence:
        try:
            patch = json.loads(node.get("patch_vec_json") or "[]")
        except json.JSONDecodeError:
            patch = []
        try:
            spec = json.loads(node.get("spectrogram_json") or "[]")
        except json.JSONDecodeError:
            spec = []
        try:
            phrases = json.loads(node.get("phrases_json") or "[]")
        except json.JSONDecodeError:
            phrases = []
        try:
            phrase_ids = json.loads(node.get("phrase_ids_json") or "[]")
        except json.JSONDecodeError:
            phrase_ids = []
        try:
            drive = json.loads(node.get("drive_json") or "{}")
        except json.JSONDecodeError:
            drive = {}
        if not isinstance(drive, dict):
            drive = {}
        return GroundedSentence(
            sentence_id=str(node.get("sentence_id") or uuid4()),
            sentence=str(node.get("sentence") or ""),
            phrases=[str(x) for x in phrases],
            phrase_ids=[str(x) for x in phrase_ids],
            fly_id=str(node.get("fly_id") or "fly_001"),
            drive={str(k): float(v) for k, v in drive.items()},
            patch_vec=[float(x) for x in patch],
            spectrogram=spec,
            confidence=float(node.get("confidence") or 0.0),
            score=float(node.get("score") or 0.0),
            hits=int(node.get("hits") or 0),
            place_id=node.get("place_id"),
            step=int(node.get("step") or 0),
            created_at=str(node.get("created_at") or ""),
            updated_at=str(node.get("updated_at") or ""),
        )


def open_grounded_phrase_store(
    *,
    backend: str = "in_memory",
    uri: str = "bolt://127.0.0.1:7688",
    user: str = "neo4j",
    password: str | None = None,
    database: str = "neo4j",
) -> GroundedPhraseStore:
    kind = str(backend or "in_memory").strip().lower()
    if kind in ("", "in_memory", "memory", "ram"):
        return InMemoryGroundedPhraseStore()
    pwd = (
        password if password is not None else os.environ.get("NEO4J_SKILL_PASSWORD", "flymindskill")
    )
    return Neo4jGroundedPhraseStore(uri=uri, user=user, password=str(pwd), database=database)
