"""Stage: register — compute content_hash and upsert the videos row + jobs row.

content_hash is sha256 of the media bytes; it is the idempotency key for the whole pipeline.
Re-registering the same bytes returns the existing video_id (no duplicate row).
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def run(conn, ctx) -> None:
    path = Path(ctx.media_path)
    ctx.content_hash = _sha256_file(path)

    # Upsert the video by content_hash (idempotent). ON CONFLICT keeps the existing row.
    row = conn.execute(
        """
        INSERT INTO videos
            (content_hash, source_url, source_platform, title, media_path, stream_type)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (content_hash) DO UPDATE SET content_hash = EXCLUDED.content_hash
        RETURNING id
        """,
        (
            ctx.content_hash, ctx.source_url, ctx.source_platform, ctx.title,
            ctx.media_path, ctx.stream_type,
        ),
    ).fetchone()
    ctx.video_id = row[0]

    # Ensure a jobs row exists for this content_hash, linked to the video.
    conn.execute(
        """
        INSERT INTO jobs (video_id, content_hash, status)
        VALUES (%s, %s, 'running')
        ON CONFLICT (content_hash) DO UPDATE SET video_id = EXCLUDED.video_id
        """,
        (ctx.video_id, ctx.content_hash),
    )
