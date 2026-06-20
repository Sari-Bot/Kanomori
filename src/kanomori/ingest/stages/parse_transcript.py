"""Stage: parse_transcript — SRT -> transcript_segments rows (text + norm + embedding + tsv).

Idempotent: deletes any existing segments for the video first, so a re-run replaces rather
than appends. Embeds all segment texts in one batch and tokenizes for the JP full-text index.
"""

from __future__ import annotations

from pathlib import Path

from pgvector.psycopg import register_vector

from kanomori.ingest.artifacts import srt_path_for
from kanomori.srt import parse_srt
from kanomori.text import normalize, tokenize_for_fts


def run(conn, ctx) -> None:
    # Derive the SRT path from content_hash so this stage works on a resumed run where
    # transcribe was skipped (and so ctx.srt_path was never set this session).
    srt_path = ctx.srt_path or str(srt_path_for(ctx.content_hash))
    sentences = parse_srt(Path(srt_path).read_text(encoding="utf-8"))
    if not sentences:
        raise RuntimeError(f"no transcript parsed from {srt_path}")

    register_vector(conn)
    # Idempotent: clear prior rows for this video before inserting.
    conn.execute("DELETE FROM transcript_segments WHERE video_id = %s", (ctx.video_id,))

    embedder = ctx.embedder
    texts = [s["text"] for s in sentences]
    norms = [normalize(t) for t in texts]
    vectors = embedder.embed_texts(norms)

    with conn.cursor() as cur:
        for seq, (sent, norm, vec) in enumerate(zip(sentences, norms, vectors, strict=True)):
            cur.execute(
                """
                INSERT INTO transcript_segments
                    (video_id, seq, start_sec, end_sec, text, text_norm, embedding, tsv)
                VALUES (%s, %s, %s, %s, %s, %s, %s, to_tsvector('simple', %s))
                """,
                (
                    ctx.video_id, seq, sent["start"], sent["end"], sent["text"], norm,
                    vec, tokenize_for_fts(norm),
                ),
            )
