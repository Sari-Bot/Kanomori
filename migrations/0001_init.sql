-- 0001_init.sql — core schema for the transcript vertical slice.
--
-- One datastore: PostgreSQL + pgvector (HNSW) + tsvector (GIN). Vectors, lexical search, and
-- metadata all live here so a single query can do filtered-vector + lexical + metadata joins
-- with no cross-store sync. Visual entities (frames/ocr/scene) arrive in 0002.
--
-- Embedding dims are pinned to the chosen models and are authoritative here:
--   transcript_segments.embedding  vector(1024)  -- BGE-M3 dense
--
-- Japanese FTS: stock Postgres cannot segment Japanese (no spaces => one token). We tokenize
-- application-side (fugashi/MeCab) and store space-joined tokens, indexing with the 'simple'
-- config. The query path tokenizes identically, so lexical matching is symmetric. Dense
-- BGE-M3 cushions lexical gaps. Upgrade path (later): pgroonga / pg_bigm n-gram FTS.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per ingested stream. content_hash (sha256 of the media) is the idempotency key:
-- re-ingesting the same bytes is a no-op. We store a source_url (link, never redistributed
-- video) and a derived media_path on the local filesystem.
CREATE TABLE IF NOT EXISTS videos (
    id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    content_hash    text NOT NULL UNIQUE,
    source_platform text,
    source_url      text,
    title           text,
    streamed_at     timestamptz,
    duration_sec    double precision,
    media_path      text,
    stream_type     text,                 -- inferred scene/stream type (nullable until classify)
    created_at      timestamptz NOT NULL DEFAULT now()
);

-- Ingestion job + per-stage status. The worker picks queued/failed jobs, runs only stages
-- whose stage_status[stage].state != 'done', and commits per stage, so a crash resumes at the
-- first non-done stage. UNIQUE(content_hash) means one job per distinct media.
CREATE TABLE IF NOT EXISTS jobs (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    video_id      bigint REFERENCES videos(id) ON DELETE CASCADE,
    content_hash  text NOT NULL UNIQUE,
    status        text NOT NULL DEFAULT 'queued'
                  CHECK (status IN ('queued', 'running', 'failed', 'complete')),
    current_stage text,
    stage_status  jsonb NOT NULL DEFAULT '{}'::jsonb,
    attempts      int  NOT NULL DEFAULT 0,
    error         text,
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs (status);

-- Sentence-level transcript units (from KITS SRT). This is Kanomori's transcript retrieval
-- granularity. text is original JP; text_norm is normalized; tsv holds JP-tokenized lexical
-- index; embedding is BGE-M3 dense.
CREATE TABLE IF NOT EXISTS transcript_segments (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    video_id    bigint NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    seq         int    NOT NULL,
    start_sec   double precision NOT NULL,
    end_sec     double precision NOT NULL,
    text        text   NOT NULL,
    text_norm   text   NOT NULL,
    embedding   vector(1024),
    tsv         tsvector,
    UNIQUE (video_id, seq)
);

-- HNSW for dense ANN (cosine). GIN for lexical. (video_id, start_sec) for timeline/context
-- lookups (nearby transcript around a hit).
CREATE INDEX IF NOT EXISTS transcript_embedding_idx
    ON transcript_segments USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS transcript_tsv_idx
    ON transcript_segments USING gin (tsv);
CREATE INDEX IF NOT EXISTS transcript_video_time_idx
    ON transcript_segments (video_id, start_sec);
