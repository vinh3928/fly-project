CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS fly_mind;

-- Legacy single-vector STM (left for historical rows; writers use short_term_memory).
CREATE TABLE IF NOT EXISTS fly_mind.stm_embeddings (
    id              BIGSERIAL PRIMARY KEY,
    episode_id      TEXT NOT NULL DEFAULT 'live',
    step            BIGINT NOT NULL DEFAULT 0,
    sim_time        DOUBLE PRECISION NOT NULL DEFAULT 0,
    fly_id          TEXT NOT NULL DEFAULT 'fly_001',
    embedding       vector(272) NOT NULL,
    signals         JSONB NOT NULL DEFAULT '{}'::jsonb,
    affect          JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS stm_embeddings_embedding_hnsw
    ON fly_mind.stm_embeddings
    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS stm_embeddings_fly_created
    ON fly_mind.stm_embeddings (fly_id, created_at DESC);

-- Multi-vector short-term memory (primary STM for live mind).
CREATE TABLE IF NOT EXISTS fly_mind.short_term_memory (
    id              BIGSERIAL PRIMARY KEY,
    episode_id      TEXT NOT NULL DEFAULT 'live',
    step            BIGINT NOT NULL DEFAULT 0,
    sim_time        DOUBLE PRECISION NOT NULL DEFAULT 0,
    fly_id          TEXT NOT NULL DEFAULT 'fly_001',
    left_eye_vec    vector(128) NOT NULL,
    right_eye_vec   vector(128) NOT NULL,
    sensors_vec     vector(192) NOT NULL,
    status_vec      vector(32) NOT NULL,
    affect_vec      vector(16) NOT NULL,
    embedding       vector(272) NOT NULL,
    signals         JSONB NOT NULL DEFAULT '{}'::jsonb,
    affect          JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS stm_embedding_hnsw
    ON fly_mind.short_term_memory
    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS stm_left_eye_hnsw
    ON fly_mind.short_term_memory
    USING hnsw (left_eye_vec vector_cosine_ops);

CREATE INDEX IF NOT EXISTS stm_right_eye_hnsw
    ON fly_mind.short_term_memory
    USING hnsw (right_eye_vec vector_cosine_ops);

CREATE INDEX IF NOT EXISTS stm_fly_created
    ON fly_mind.short_term_memory (fly_id, created_at DESC);

CREATE TABLE IF NOT EXISTS fly_mind.thoughts (
    id               BIGSERIAL PRIMARY KEY,
    stm_id           BIGINT REFERENCES fly_mind.short_term_memory(id) ON DELETE SET NULL,
    reflection       TEXT NOT NULL DEFAULT '',
    devil_critique   TEXT NOT NULL DEFAULT '',
    refined_thought  TEXT NOT NULL DEFAULT '',
    intent           JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
