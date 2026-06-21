"""Stage: image_embed — compute pHash and DINOv2 embeddings for frame rows."""

from __future__ import annotations

from pathlib import Path

from pgvector.psycopg import register_vector

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
    rows = conn.execute(
        "SELECT id, frame_path FROM frames WHERE video_id = %s ORDER BY ts_sec",
        (ctx.video_id,),
    ).fetchall()
    if not rows:
        return "skipped"

    register_vector(conn)
    with conn.cursor() as cur:
        for frame_id, frame_path in rows:
            path = Path(frame_path)
            cur.execute(
                "UPDATE frames SET phash = %s, embedding = %s WHERE id = %s",
                (compute_frame_phash(path), embed_frame(path), frame_id),
            )
    return None
