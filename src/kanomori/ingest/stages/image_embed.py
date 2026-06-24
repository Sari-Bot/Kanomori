"""Stage: image_embed — compute pHash and DINOv2 embeddings for frame rows.

Compute/persist split (Task #12): ``compute`` sources frames from disk (no DB), computes each
frame's pHash + embedding, and returns an :class:`ImageEmbedResult` of :class:`ImageEmbedRow`
keyed by ``ts_sec`` (embedding carried as base64 float32 via the codec). ``persist`` resolves
``frame_id`` by ``(video_id, ts_sec)`` and UPDATEs frames SET phash, embedding.
"""

from __future__ import annotations

from pathlib import Path

from pgvector.psycopg import register_vector

from kanomori.ingest.artifacts import frames_on_disk
from kanomori.ingest.stage_result import ImageEmbedResult, ImageEmbedRow

_EMBEDDER = None


def _embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from kanomori.embed.image_embedder import DINOv2Embedder

        _EMBEDDER = DINOv2Embedder()
    return _EMBEDDER


def compute_frame_phash(path: Path) -> int:
    from PIL import Image

    from kanomori.embed.phash import compute_phash

    with Image.open(path) as image:
        return compute_phash(image)


def embed_frame(path: Path):
    return _embedder().embed_image_path(str(path))


def run(conn, ctx):
    result = compute(ctx)
    if result == "skipped":
        return "skipped"
    persist(conn, ctx.video_id, result)
    return None


def compute(ctx):
    """pHash + embed every on-disk frame; return ImageEmbedResult or "skipped".

    Frames come from :func:`frames_on_disk` (no DB), ts recovered from each JPEG name. Returns the
    "skipped" sentinel when the video has no frames, preserving run()'s contract.
    """
    frames = frames_on_disk(ctx.content_hash)
    if not frames:
        return "skipped"

    rows = [
        ImageEmbedRow.build(
            ts_sec=ts_sec,
            phash=compute_frame_phash(path),
            embedding=embed_frame(path),
        )
        for ts_sec, path in frames
    ]
    return ImageEmbedResult(rows=rows)


def persist(conn, video_id, result: ImageEmbedResult) -> None:
    """Write each frame's phash + embedding, resolving frame_id by (video_id, ts_sec)."""
    register_vector(conn)
    frame_ids = {
        ts_sec: frame_id
        for frame_id, ts_sec in conn.execute(
            "SELECT id, ts_sec FROM frames WHERE video_id = %s", (video_id,)
        ).fetchall()
    }
    with conn.cursor() as cur:
        for row in result.rows:
            frame_id = frame_ids.get(row.ts_sec)
            if frame_id is None:
                continue
            cur.execute(
                "UPDATE frames SET phash = %s, embedding = %s WHERE id = %s",
                (row.phash, row.vector(), frame_id),
            )
