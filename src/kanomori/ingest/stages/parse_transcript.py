"""Stage: parse_transcript — SRT -> transcript_segments rows (text + norm + embedding + tsv).

Idempotent: deletes any existing segments for the video first, so a re-run replaces rather
than appends. Embeds all segment texts in one batch and tokenizes for the JP full-text index.

Compute/persist split (Task #12): ``compute`` parses the SRT, normalizes each cue, and embeds
the normalized texts in one batch, returning a :class:`ParseTranscriptResult` of
:class:`TranscriptSegmentRow` (embeddings carried as base64 float32 via the codec). ``persist``
takes a live conn + the resolved video_id and runs the DELETE + INSERTs, deriving the ``tsv``
column coordinator-side with ``to_tsvector`` from the carried ``text_norm`` (the wire contract
intentionally omits tsv).
"""

from __future__ import annotations

from pathlib import Path

from pgvector.psycopg import register_vector

from kanomori.ingest.artifacts import srt_path_for
from kanomori.ingest.stage_result import ParseTranscriptResult, TranscriptSegmentRow
from kanomori.srt import parse_srt
from kanomori.text import normalize, tokenize_for_fts


def run(conn, ctx) -> None:
    result = compute(ctx)
    persist(conn, ctx.video_id, result)


def compute(ctx) -> ParseTranscriptResult:
    """Parse + normalize + embed the SRT into transcript rows. No DB connection.

    The SRT path is derived from content_hash so this works on a resumed run where transcribe was
    skipped (and ctx.srt_path was never set this session).
    """
    srt_path = ctx.srt_path or str(srt_path_for(ctx.content_hash))
    sentences = parse_srt(Path(srt_path).read_text(encoding="utf-8"))
    if not sentences:
        raise RuntimeError(f"no transcript parsed from {srt_path}")

    texts = [s["text"] for s in sentences]
    norms = [normalize(t) for t in texts]
    vectors = ctx.embedder.embed_texts(norms)

    segments = [
        TranscriptSegmentRow.build(
            seq=seq,
            start_sec=sent["start"],
            end_sec=sent["end"],
            text=sent["text"],
            text_norm=norm,
            embedding=vec,
        )
        for seq, (sent, norm, vec) in enumerate(zip(sentences, norms, vectors, strict=True))
    ]
    return ParseTranscriptResult(segments=segments)


def persist(conn, video_id, result: ParseTranscriptResult) -> None:
    """Replace this video's transcript_segments rows; derive tsv coordinator-side."""
    register_vector(conn)
    # Idempotent: clear prior rows for this video before inserting.
    conn.execute("DELETE FROM transcript_segments WHERE video_id = %s", (video_id,))

    with conn.cursor() as cur:
        for row in result.segments:
            cur.execute(
                """
                INSERT INTO transcript_segments
                    (video_id, seq, start_sec, end_sec, text, text_norm, embedding, tsv)
                VALUES (%s, %s, %s, %s, %s, %s, %s, to_tsvector('simple', %s))
                """,
                (
                    video_id, row.seq, row.start_sec, row.end_sec, row.text, row.text_norm,
                    row.vector(), tokenize_for_fts(row.text_norm),
                ),
            )
