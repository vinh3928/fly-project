"""Speech cortex: utterances + alphabet + eye images on a separate Neo4j partition.

Place graph stays on 7687; skill cortex on 7688; speech on 7689.
No ML speech model — alphabet comes from the articulatory/spectral table.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Protocol
from uuid import uuid4

import numpy as np

from flybrain.speech.articulatory import (
    ARTICULATORY_DIM,
    AUDIO_BAND_DIM,
    articulatory_to_letters,
    articulatory_to_phonemes,
    clip_articulatory,
    clip_audio_bands,
    phonemes_to_letters,
    spectral_to_letters,
    spectral_to_phonemes,
)
from flybrain.speech.babble import SpeechBabbleConfig

UtteranceStatus = Literal["candidate", "promoted", "retired"]
UtteranceSource = Literal["babble", "hear", "seed"]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp01(value: float, default: float = 0.0) -> float:
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except (TypeError, ValueError):
        return float(default)


@dataclass
class SpeechCortexConfig:
    enabled: bool = False
    backend: str = "in_memory"
    uri: str = "bolt://127.0.0.1:7689"
    user: str = "neo4j"
    password: str | None = None
    database: str = "neo4j"
    promote_min_score: float = 0.4
    hear_rms_min: float = 0.08
    hear_cooldown_steps: int = 30
    # Parallel speech loop cadence (hear / start-babble). Mouth playback still
    # advances every motor tick so articulatory sequences play smoothly.
    speech_interval: int = 4
    image_dir: str = "data/speech_images"
    # When babble alphabet looks word-like, bump Word.weight + utterance.score.
    word_min_len: int = 2
    word_weight_delta: float = 0.1
    babble: SpeechBabbleConfig = field(default_factory=SpeechBabbleConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> SpeechCortexConfig:
        data = dict(raw or {})
        return cls(
            enabled=bool(data.get("enabled", False)),
            backend=str(data.get("backend", "in_memory")),
            uri=str(data.get("uri", "bolt://127.0.0.1:7689")),
            user=str(data.get("user", "neo4j")),
            password=data.get("password"),
            database=str(data.get("database", "neo4j")),
            promote_min_score=float(data.get("promote_min_score", 0.4)),
            hear_rms_min=float(data.get("hear_rms_min", 0.08)),
            hear_cooldown_steps=max(0, int(data.get("hear_cooldown_steps", 30))),
            speech_interval=max(1, int(data.get("speech_interval", 4))),
            image_dir=str(data.get("image_dir", "data/speech_images")),
            word_min_len=max(1, int(data.get("word_min_len", 2))),
            word_weight_delta=_clamp01(float(data.get("word_weight_delta", 0.1)), default=0.1),
            babble=SpeechBabbleConfig.from_dict(
                data.get("babble") if isinstance(data.get("babble"), dict) else {}
            ),
        )


_VOWELS = frozenset("aeiou")
_CONSONANTS = frozenset("bcdfghjklmnpqrstvwxyz")


def normalize_speech_word(alphabet: str | None) -> str:
    """Lowercase a-z only; drop silence markers."""
    raw = str(alphabet or "").lower()
    return "".join(ch for ch in raw if ch.isalpha())


def looks_like_word(alphabet: str | None, *, min_len: int = 2) -> bool:
    """True when babble alphabet looks like a pronounceable word token.

    Requires length, mixed vowel+consonant, and not a single repeated letter.
    """
    word = normalize_speech_word(alphabet)
    if len(word) < max(1, int(min_len)):
        return False
    if len(set(word)) < 2:
        return False
    chars = set(word)
    return bool(chars & _VOWELS) and bool(chars & _CONSONANTS)


@dataclass
class SpeechWord:
    """Per-fly lexical memory entry reinforced by word-like babble."""

    word: str
    fly_id: str = "fly_001"
    weight: float = 0.0
    hits: int = 0
    updated_at: str = ""

    def __post_init__(self) -> None:
        self.word = normalize_speech_word(self.word)
        self.fly_id = str(self.fly_id or "fly_001")
        self.weight = _clamp01(self.weight)
        self.hits = max(0, int(self.hits))
        if not self.updated_at:
            self.updated_at = _utcnow()

    def to_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "fly_id": self.fly_id,
            "weight": float(self.weight),
            "hits": int(self.hits),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> SpeechWord:
        data = dict(raw or {})
        return cls(
            word=str(data.get("word") or ""),
            fly_id=str(data.get("fly_id") or "fly_001"),
            weight=float(data.get("weight") or 0.0),
            hits=int(data.get("hits") or 0),
            updated_at=str(data.get("updated_at") or ""),
        )


@dataclass
class SpeechImage:
    image_id: str
    path_or_b64_thumb: str = ""
    place_id: str | None = None
    step: int = 0
    patch_vec: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "path_or_b64_thumb": self.path_or_b64_thumb,
            "place_id": self.place_id,
            "step": int(self.step),
            "patch_vec": [float(x) for x in self.patch_vec],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> SpeechImage:
        data = dict(raw or {})
        return cls(
            image_id=str(data.get("image_id") or uuid4()),
            path_or_b64_thumb=str(data.get("path_or_b64_thumb") or ""),
            place_id=(str(data["place_id"]) if data.get("place_id") is not None else None),
            step=int(data.get("step") or 0),
            patch_vec=[float(x) for x in (data.get("patch_vec") or [])],
        )


@dataclass
class SpeechContext:
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
class UtteranceProgram:
    utterance_id: str
    alphabet: str
    status: UtteranceStatus
    phonemes: list[str] = field(default_factory=list)
    articulatory: list[list[float]] = field(default_factory=list)
    spectrogram: list[list[float]] = field(default_factory=list)
    source: str = "babble"
    fly_id: str = "fly_001"
    score: float = 0.0
    created_at: str = ""
    promoted_at: str | None = None
    image: SpeechImage | None = None
    context: SpeechContext | None = None

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = _utcnow()
        self.score = _clamp01(self.score)
        self.phonemes = [str(p) for p in (self.phonemes or [])]
        self.alphabet = str(self.alphabet or "#")
        self.articulatory = [
            [float(x) for x in clip_articulatory(row).tolist()] for row in (self.articulatory or [])
        ]
        self.spectrogram = [
            [float(x) for x in clip_audio_bands(row).tolist()] for row in (self.spectrogram or [])
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "utterance_id": self.utterance_id,
            "alphabet": self.alphabet,
            "status": self.status,
            "phonemes": list(self.phonemes),
            "articulatory": [list(r) for r in self.articulatory],
            "spectrogram": [list(r) for r in self.spectrogram],
            "source": self.source,
            "fly_id": self.fly_id,
            "score": float(self.score),
            "created_at": self.created_at,
            "promoted_at": self.promoted_at,
            "image": self.image.to_dict() if self.image else None,
            "context": self.context.to_dict() if self.context else None,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> UtteranceProgram:
        data = dict(raw or {})
        img_raw = data.get("image")
        img = SpeechImage.from_dict(img_raw) if isinstance(img_raw, dict) else None
        ctx_raw = data.get("context")
        ctx = None
        if isinstance(ctx_raw, dict):
            ctx = SpeechContext(
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
            utterance_id=str(data.get("utterance_id") or uuid4()),
            alphabet=str(data.get("alphabet") or "#"),
            status=status,  # type: ignore[arg-type]
            phonemes=[str(p) for p in (data.get("phonemes") or [])],
            articulatory=[list(r) for r in (data.get("articulatory") or [])],
            spectrogram=[list(r) for r in (data.get("spectrogram") or [])],
            source=str(data.get("source") or "babble"),
            fly_id=str(data.get("fly_id") or "fly_001"),
            score=float(data.get("score") or 0.0),
            created_at=str(data.get("created_at") or ""),
            promoted_at=str(data["promoted_at"]) if data.get("promoted_at") else None,
            image=img,
            context=ctx,
        )


def build_babble_utterance(
    *,
    fly_id: str,
    articulatory: list[np.ndarray],
    image: SpeechImage | None = None,
    context: SpeechContext | None = None,
    spectrogram: list[np.ndarray] | None = None,
) -> UtteranceProgram:
    seq = [clip_articulatory(a) for a in articulatory]
    phonemes = articulatory_to_phonemes(seq)
    alphabet = articulatory_to_letters(seq)
    spec = [clip_audio_bands(s) for s in (spectrogram or [])]
    return UtteranceProgram(
        utterance_id=str(uuid4()),
        alphabet=alphabet,
        status="candidate",
        phonemes=phonemes,
        articulatory=[a.tolist() for a in seq],
        spectrogram=[s.tolist() for s in spec],
        source="babble",
        fly_id=str(fly_id),
        image=image,
        context=context,
    )


def build_hear_utterance(
    *,
    fly_id: str,
    spectrogram: list[np.ndarray],
    rms: float = 0.0,
    voiced: float = 0.0,
    image: SpeechImage | None = None,
    context: SpeechContext | None = None,
) -> UtteranceProgram:
    frames = [clip_audio_bands(s) for s in spectrogram]
    phonemes = spectral_to_phonemes(frames, rms=rms, voiced=voiced)
    alphabet = phonemes_to_letters(phonemes) if phonemes else spectral_to_letters(frames, rms=rms)
    return UtteranceProgram(
        utterance_id=str(uuid4()),
        alphabet=alphabet or "#",
        status="candidate",
        phonemes=phonemes,
        articulatory=[],
        spectrogram=[f.tolist() for f in frames],
        source="hear",
        fly_id=str(fly_id),
        image=image,
        context=context,
    )


class SpeechCortexStore(Protocol):
    def upsert_utterance(self, program: UtteranceProgram) -> UtteranceProgram: ...

    def get_utterance(self, utterance_id: str) -> UtteranceProgram | None: ...

    def list_utterances(
        self,
        *,
        fly_id: str = "fly_001",
        status: UtteranceStatus | None = None,
        limit: int = 64,
    ) -> list[UtteranceProgram]: ...

    def promote_utterance(
        self, utterance_id: str, *, alphabet: str | None = None
    ) -> UtteranceProgram | None: ...

    def retire_utterance(self, utterance_id: str) -> UtteranceProgram | None: ...

    def reinforce_word(
        self,
        *,
        fly_id: str,
        word: str,
        delta: float = 0.1,
        utterance_id: str | None = None,
    ) -> SpeechWord: ...

    def get_word(self, *, fly_id: str, word: str) -> SpeechWord | None: ...

    def list_words(
        self, *, fly_id: str = "fly_001", limit: int = 64
    ) -> list[SpeechWord]: ...

    def close(self) -> None: ...


def reinforce_wordlike_babble(
    store: SpeechCortexStore,
    program: UtteranceProgram,
    *,
    min_len: int = 2,
    delta: float = 0.1,
) -> tuple[UtteranceProgram, SpeechWord | None]:
    """If babble alphabet looks like a word, bump Word.weight and utterance.score."""
    row = UtteranceProgram.from_dict(program.to_dict())
    if str(row.source) != "babble":
        return row, None
    if not looks_like_word(row.alphabet, min_len=min_len):
        return row, None
    token = normalize_speech_word(row.alphabet)
    if not token:
        return row, None
    bumped = store.reinforce_word(
        fly_id=row.fly_id,
        word=token,
        delta=delta,
        utterance_id=row.utterance_id,
    )
    row.score = _clamp01(float(row.score) + float(delta))
    row = store.upsert_utterance(row)
    return row, bumped


@dataclass
class InMemorySpeechCortex:
    _rows: dict[str, UtteranceProgram] = field(default_factory=dict)
    _words: dict[tuple[str, str], SpeechWord] = field(default_factory=dict)

    def upsert_utterance(self, program: UtteranceProgram) -> UtteranceProgram:
        row = UtteranceProgram.from_dict(program.to_dict())
        if not row.utterance_id:
            row.utterance_id = str(uuid4())
        self._rows[row.utterance_id] = row
        return UtteranceProgram.from_dict(row.to_dict())

    def get_utterance(self, utterance_id: str) -> UtteranceProgram | None:
        row = self._rows.get(str(utterance_id))
        return UtteranceProgram.from_dict(row.to_dict()) if row else None

    def list_utterances(
        self,
        *,
        fly_id: str = "fly_001",
        status: UtteranceStatus | None = None,
        limit: int = 64,
    ) -> list[UtteranceProgram]:
        rows = [
            UtteranceProgram.from_dict(s.to_dict())
            for s in self._rows.values()
            if str(s.fly_id) == str(fly_id) and (status is None or s.status == status)
        ]
        rows.sort(key=lambda s: (s.promoted_at or "", s.created_at), reverse=True)
        return rows[: max(1, int(limit))]

    def promote_utterance(
        self, utterance_id: str, *, alphabet: str | None = None
    ) -> UtteranceProgram | None:
        row = self._rows.get(str(utterance_id))
        if row is None:
            return None
        row.status = "promoted"
        row.promoted_at = _utcnow()
        if alphabet is not None and str(alphabet).strip():
            row.alphabet = str(alphabet).strip()
        return UtteranceProgram.from_dict(row.to_dict())

    def retire_utterance(self, utterance_id: str) -> UtteranceProgram | None:
        row = self._rows.get(str(utterance_id))
        if row is None:
            return None
        row.status = "retired"
        return UtteranceProgram.from_dict(row.to_dict())

    def reinforce_word(
        self,
        *,
        fly_id: str,
        word: str,
        delta: float = 0.1,
        utterance_id: str | None = None,
    ) -> SpeechWord:
        _ = utterance_id
        token = normalize_speech_word(word)
        key = (str(fly_id), token)
        existing = self._words.get(key)
        if existing is None:
            existing = SpeechWord(word=token, fly_id=str(fly_id), weight=0.0, hits=0)
        existing.weight = _clamp01(float(existing.weight) + float(delta))
        existing.hits = int(existing.hits) + 1
        existing.updated_at = _utcnow()
        self._words[key] = existing
        return SpeechWord.from_dict(existing.to_dict())

    def get_word(self, *, fly_id: str, word: str) -> SpeechWord | None:
        row = self._words.get((str(fly_id), normalize_speech_word(word)))
        return SpeechWord.from_dict(row.to_dict()) if row else None

    def list_words(
        self, *, fly_id: str = "fly_001", limit: int = 64
    ) -> list[SpeechWord]:
        rows = [
            SpeechWord.from_dict(w.to_dict())
            for (fid, _), w in self._words.items()
            if str(fid) == str(fly_id) and w.word
        ]
        rows.sort(key=lambda w: (w.weight, w.hits, w.updated_at), reverse=True)
        return rows[: max(1, int(limit))]

    def close(self) -> None:
        return None


@dataclass
class Neo4jSpeechCortex:
    uri: str = "bolt://localhost:7689"
    user: str = "neo4j"
    password: str = "flymindspeech"
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
            raise RuntimeError("Neo4j speech cortex is closed")
        records, _summary, _keys = self._driver.execute_query(
            cypher,
            parameters_=parameters or {},
            database_=self.database,
        )
        return list(records)

    def _ensure_schema(self) -> None:
        self._query("""
            CREATE CONSTRAINT utterance_id_unique IF NOT EXISTS
            FOR (u:Utterance) REQUIRE u.utterance_id IS UNIQUE
            """)
        self._query("""
            CREATE CONSTRAINT letter_char_unique IF NOT EXISTS
            FOR (l:Letter) REQUIRE l.char IS UNIQUE
            """)
        self._query("""
            CREATE CONSTRAINT speech_image_id_unique IF NOT EXISTS
            FOR (i:SpeechImage) REQUIRE i.image_id IS UNIQUE
            """)
        self._query("""
            CREATE INDEX utterance_fly_status IF NOT EXISTS
            FOR (u:Utterance) ON (u.fly_id, u.status, u.updated_at)
            """)
        self._query("""
            CREATE CONSTRAINT speech_word_unique IF NOT EXISTS
            FOR (w:Word) REQUIRE (w.fly_id, w.word) IS UNIQUE
            """)
        self._query("""
            CREATE INDEX speech_word_weight IF NOT EXISTS
            FOR (w:Word) ON (w.fly_id, w.weight)
            """)
        # Seed alphabet nodes.
        for ch in list("abcdefghijklmnopqrstuvwxyz") + ["#"]:
            self._query(
                "MERGE (l:Letter {char: $char}) SET l.char = $char",
                {"char": ch},
            )

    def upsert_utterance(self, program: UtteranceProgram) -> UtteranceProgram:
        row = UtteranceProgram.from_dict(program.to_dict())
        if not row.utterance_id:
            row.utterance_id = str(uuid4())
        now = _utcnow()
        if not row.created_at:
            row.created_at = now
        self._query(
            """
            MERGE (u:Utterance {utterance_id: $utterance_id})
            SET u.alphabet = $alphabet,
                u.status = $status,
                u.phonemes_json = $phonemes_json,
                u.articulatory_json = $articulatory_json,
                u.spectrogram_json = $spectrogram_json,
                u.source = $source,
                u.fly_id = $fly_id,
                u.score = $score,
                u.created_at = coalesce(u.created_at, $created_at),
                u.promoted_at = $promoted_at,
                u.updated_at = $updated_at
            """,
            {
                "utterance_id": row.utterance_id,
                "alphabet": row.alphabet,
                "status": row.status,
                "phonemes_json": json.dumps(list(row.phonemes)),
                "articulatory_json": json.dumps(row.articulatory),
                "spectrogram_json": json.dumps(row.spectrogram),
                "source": row.source,
                "fly_id": row.fly_id,
                "score": float(row.score),
                "created_at": row.created_at,
                "promoted_at": row.promoted_at,
                "updated_at": now,
            },
        )
        # Replace letter edges (one ordered edge per alphabet character).
        self._query(
            """
            MATCH (u:Utterance {utterance_id: $utterance_id})-[r:HAS_LETTER]->(:Letter)
            DELETE r
            """,
            {"utterance_id": row.utterance_id},
        )
        alphabet = row.alphabet or "#"
        for ord_i, ch in enumerate(alphabet):
            ph = (
                row.phonemes[min(ord_i, len(row.phonemes) - 1)]
                if row.phonemes
                else "#"
            )
            self._query(
                """
                MATCH (u:Utterance {utterance_id: $utterance_id})
                MERGE (l:Letter {char: $char})
                MERGE (u)-[r:HAS_LETTER {ord: $ord}]->(l)
                SET r.phoneme = $phoneme
                """,
                {
                    "utterance_id": row.utterance_id,
                    "char": ch if (ch.isalpha() or ch == "#") else "#",
                    "ord": int(ord_i),
                    "phoneme": str(ph),
                },
            )
        if row.image is not None:
            img = row.image
            self._query(
                """
                MATCH (u:Utterance {utterance_id: $utterance_id})
                OPTIONAL MATCH (u)-[old:SAW]->(:SpeechImage)
                DELETE old
                MERGE (i:SpeechImage {image_id: $image_id})
                SET i.path_or_b64_thumb = $path,
                    i.place_id = $place_id,
                    i.step = $step,
                    i.patch_vec_json = $patch_vec_json
                MERGE (u)-[:SAW]->(i)
                """,
                {
                    "utterance_id": row.utterance_id,
                    "image_id": img.image_id,
                    "path": img.path_or_b64_thumb,
                    "place_id": img.place_id,
                    "step": int(img.step),
                    "patch_vec_json": json.dumps(list(img.patch_vec)),
                },
            )
        if row.context is not None:
            ctx = row.context
            self._query(
                """
                MATCH (u:Utterance {utterance_id: $utterance_id})
                MERGE (c:SpeechContext {utterance_id: $utterance_id})
                SET c.cue = $cue, c.place_id = $place_id, c.step = $step
                MERGE (u)-[:HEARD_AT]->(c)
                """,
                {
                    "utterance_id": row.utterance_id,
                    "cue": ctx.cue,
                    "place_id": ctx.place_id,
                    "step": int(ctx.step),
                },
            )
        return row

    def get_utterance(self, utterance_id: str) -> UtteranceProgram | None:
        rows = self._query(
            """
            MATCH (u:Utterance {utterance_id: $utterance_id})
            OPTIONAL MATCH (u)-[:SAW]->(i:SpeechImage)
            OPTIONAL MATCH (u)-[:HEARD_AT]->(c:SpeechContext)
            RETURN u, i, c
            """,
            {"utterance_id": str(utterance_id)},
        )
        if not rows:
            return None
        return self._row_to_program(rows[0])

    def list_utterances(
        self,
        *,
        fly_id: str = "fly_001",
        status: UtteranceStatus | None = None,
        limit: int = 64,
    ) -> list[UtteranceProgram]:
        if status is None:
            rows = self._query(
                """
                MATCH (u:Utterance {fly_id: $fly_id})
                OPTIONAL MATCH (u)-[:SAW]->(i:SpeechImage)
                OPTIONAL MATCH (u)-[:HEARD_AT]->(c:SpeechContext)
                RETURN u, i, c
                ORDER BY coalesce(u.promoted_at, '') DESC, u.created_at DESC
                LIMIT $limit
                """,
                {"fly_id": fly_id, "limit": int(limit)},
            )
        else:
            rows = self._query(
                """
                MATCH (u:Utterance {fly_id: $fly_id, status: $status})
                OPTIONAL MATCH (u)-[:SAW]->(i:SpeechImage)
                OPTIONAL MATCH (u)-[:HEARD_AT]->(c:SpeechContext)
                RETURN u, i, c
                ORDER BY coalesce(u.promoted_at, '') DESC, u.created_at DESC
                LIMIT $limit
                """,
                {"fly_id": fly_id, "status": status, "limit": int(limit)},
            )
        return [self._row_to_program(r) for r in rows]

    def promote_utterance(
        self, utterance_id: str, *, alphabet: str | None = None
    ) -> UtteranceProgram | None:
        row = self.get_utterance(utterance_id)
        if row is None:
            return None
        row.status = "promoted"
        row.promoted_at = _utcnow()
        if alphabet is not None and str(alphabet).strip():
            row.alphabet = str(alphabet).strip()
        return self.upsert_utterance(row)

    def retire_utterance(self, utterance_id: str) -> UtteranceProgram | None:
        row = self.get_utterance(utterance_id)
        if row is None:
            return None
        row.status = "retired"
        return self.upsert_utterance(row)

    def reinforce_word(
        self,
        *,
        fly_id: str,
        word: str,
        delta: float = 0.1,
        utterance_id: str | None = None,
    ) -> SpeechWord:
        token = normalize_speech_word(word)
        now = _utcnow()
        rows = self._query(
            """
            MERGE (w:Word {fly_id: $fly_id, word: $word})
            ON CREATE SET w.weight = 0.0, w.hits = 0, w.created_at = $now
            SET w.weight = CASE
                    WHEN coalesce(w.weight, 0.0) + $delta > 1.0 THEN 1.0
                    WHEN coalesce(w.weight, 0.0) + $delta < 0.0 THEN 0.0
                    ELSE coalesce(w.weight, 0.0) + $delta
                END,
                w.hits = coalesce(w.hits, 0) + 1,
                w.updated_at = $now
            RETURN w
            """,
            {
                "fly_id": str(fly_id),
                "word": token,
                "delta": float(delta),
                "now": now,
            },
        )
        if utterance_id:
            self._query(
                """
                MATCH (u:Utterance {utterance_id: $utterance_id})
                MATCH (w:Word {fly_id: $fly_id, word: $word})
                MERGE (u)-[:UTTERS]->(w)
                """,
                {
                    "utterance_id": str(utterance_id),
                    "fly_id": str(fly_id),
                    "word": token,
                },
            )
        node = rows[0]["w"] if rows else {"word": token, "fly_id": fly_id}
        return SpeechWord(
            word=str(node.get("word") or token),
            fly_id=str(node.get("fly_id") or fly_id),
            weight=float(node.get("weight") or 0.0),
            hits=int(node.get("hits") or 0),
            updated_at=str(node.get("updated_at") or now),
        )

    def get_word(self, *, fly_id: str, word: str) -> SpeechWord | None:
        rows = self._query(
            """
            MATCH (w:Word {fly_id: $fly_id, word: $word})
            RETURN w
            """,
            {"fly_id": str(fly_id), "word": normalize_speech_word(word)},
        )
        if not rows:
            return None
        node = rows[0]["w"]
        return SpeechWord(
            word=str(node.get("word") or ""),
            fly_id=str(node.get("fly_id") or fly_id),
            weight=float(node.get("weight") or 0.0),
            hits=int(node.get("hits") or 0),
            updated_at=str(node.get("updated_at") or ""),
        )

    def list_words(
        self, *, fly_id: str = "fly_001", limit: int = 64
    ) -> list[SpeechWord]:
        rows = self._query(
            """
            MATCH (w:Word {fly_id: $fly_id})
            RETURN w
            ORDER BY w.weight DESC, w.hits DESC, coalesce(w.updated_at, '') DESC
            LIMIT $limit
            """,
            {"fly_id": str(fly_id), "limit": int(limit)},
        )
        out: list[SpeechWord] = []
        for record in rows:
            node = record["w"]
            out.append(
                SpeechWord(
                    word=str(node.get("word") or ""),
                    fly_id=str(node.get("fly_id") or fly_id),
                    weight=float(node.get("weight") or 0.0),
                    hits=int(node.get("hits") or 0),
                    updated_at=str(node.get("updated_at") or ""),
                )
            )
        return out

    def _row_to_program(self, record: Any) -> UtteranceProgram:
        u = record["u"]
        img_node = record.get("i")
        ctx_node = record.get("c")
        image = None
        if img_node is not None:
            patch = []
            try:
                patch = json.loads(img_node.get("patch_vec_json") or "[]")
            except json.JSONDecodeError:
                patch = []
            image = SpeechImage(
                image_id=str(img_node.get("image_id") or uuid4()),
                path_or_b64_thumb=str(img_node.get("path_or_b64_thumb") or ""),
                place_id=img_node.get("place_id"),
                step=int(img_node.get("step") or 0),
                patch_vec=[float(x) for x in patch],
            )
        context = None
        if ctx_node is not None:
            context = SpeechContext(
                cue=str(ctx_node.get("cue") or ""),
                place_id=ctx_node.get("place_id"),
                step=int(ctx_node.get("step") or 0),
            )
        try:
            phonemes = json.loads(u.get("phonemes_json") or "[]")
        except json.JSONDecodeError:
            phonemes = []
        try:
            articulatory = json.loads(u.get("articulatory_json") or "[]")
        except json.JSONDecodeError:
            articulatory = []
        try:
            spectrogram = json.loads(u.get("spectrogram_json") or "[]")
        except json.JSONDecodeError:
            spectrogram = []
        status = str(u.get("status") or "candidate")
        if status not in ("candidate", "promoted", "retired"):
            status = "candidate"
        return UtteranceProgram(
            utterance_id=str(u.get("utterance_id")),
            alphabet=str(u.get("alphabet") or "#"),
            status=status,  # type: ignore[arg-type]
            phonemes=[str(p) for p in phonemes],
            articulatory=articulatory,
            spectrogram=spectrogram,
            source=str(u.get("source") or "babble"),
            fly_id=str(u.get("fly_id") or "fly_001"),
            score=float(u.get("score") or 0.0),
            created_at=str(u.get("created_at") or ""),
            promoted_at=str(u["promoted_at"]) if u.get("promoted_at") else None,
            image=image,
            context=context,
        )


def open_speech_cortex(
    *,
    backend: str = "in_memory",
    uri: str = "bolt://127.0.0.1:7689",
    user: str = "neo4j",
    password: str | None = None,
    database: str = "neo4j",
) -> SpeechCortexStore:
    kind = str(backend or "in_memory").strip().lower()
    if kind in ("", "in_memory", "memory", "ram"):
        return InMemorySpeechCortex()
    pwd = password if password is not None else os.environ.get("NEO4J_SPEECH_PASSWORD", "flymindspeech")
    return Neo4jSpeechCortex(uri=uri, user=user, password=str(pwd), database=database)


# Avoid circular import at module load: babble config lives in speech.babble
# but SpeechCortexConfig needs it — imported at top from babble after we write babble.
# Fix: speech_cortex imports SpeechBabbleConfig from babble; babble must not import speech_cortex at top for config.
_ = (ARTICULATORY_DIM, AUDIO_BAND_DIM)
