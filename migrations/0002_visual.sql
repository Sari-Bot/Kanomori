-- 0002_visual.sql — visual retrieval schema for screenshot search.
--
-- DINOv2 ViT-B/14 frame embeddings are pinned at vector(768). pHash values are stored as
-- signed bigint while preserving the unsigned 64-bit bit pattern; hamming distance queries
-- must cast XOR results to bit(64): bit_count((phash # :q)::bit(64)).

CREATE TABLE IF NOT EXISTS frames (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    video_id    bigint NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    ts_sec      double precision NOT NULL,
    frame_path  text NOT NULL,
    phash       bigint,
    embedding   vector(768),
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (video_id, ts_sec)
);

CREATE INDEX IF NOT EXISTS frames_embedding_idx
    ON frames USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS frames_video_time_idx
    ON frames (video_id, ts_sec);

CREATE TABLE IF NOT EXISTS ocr_segments (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    video_id    bigint NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    frame_id    bigint NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    ts_sec      double precision NOT NULL,
    text        text NOT NULL,
    confidence  double precision,
    bbox        jsonb NOT NULL DEFAULT '{}'::jsonb,
    tsv         tsvector
);

CREATE INDEX IF NOT EXISTS ocr_segments_tsv_idx
    ON ocr_segments USING gin (tsv);
CREATE INDEX IF NOT EXISTS ocr_segments_video_time_idx
    ON ocr_segments (video_id, ts_sec);

CREATE TABLE IF NOT EXISTS scene_segments (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    video_id    bigint NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    start_sec   double precision NOT NULL,
    end_sec     double precision NOT NULL,
    scene_type  text NOT NULL CHECK (
        scene_type IN (
            'singing', 'chatting', 'gaming', 'waiting',
            'superchat', 'announcement', 'collab'
        )
    ),
    confidence  double precision
);

CREATE INDEX IF NOT EXISTS scene_segments_video_start_idx
    ON scene_segments (video_id, start_sec);
