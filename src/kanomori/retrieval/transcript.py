"""Transcript candidate generation: lexical (tsvector) + dense (pgvector) → RRF.

Hybrid retrieval over ``transcript_segments``. The lexical path tokenizes the query with the
same fugashi tokenizer used at ingest (symmetric matching on Japanese) and ranks by
``ts_rank``; the dense path embeds the query and ranks by cosine distance over the HNSW index.
The two rankings are fused with scale-free RRF — no score normalization. ``candidates`` is the
public entry; the per-path helpers are exposed for testing and for the merge layer.
"""

from __future__ import annotations

import numpy as np

from kanomori.fusion import reciprocal_rank_fusion
from kanomori.models import Candidate, Modality
from kanomori.text import tokenize_for_fts


def lexical_candidates(conn, query: str, *, k: int = 50) -> list[Candidate]:
    """Full-text candidates ranked by ts_rank over the JP-tokenized tsv column."""
    tokens = tokenize_for_fts(query)
    if not tokens:
        return []
    rows = conn.execute(
        """
        SELECT id, video_id, start_sec, ts_rank(tsv, plainto_tsquery('simple', %s)) AS rank
        FROM transcript_segments
        WHERE tsv @@ plainto_tsquery('simple', %s)
        ORDER BY rank DESC
        LIMIT %s
        """,
        (tokens, tokens, k),
    ).fetchall()
    return [
        Candidate(
            video_id=vid, ts_sec=start, modality=Modality.TRANSCRIPT,
            rank=i, raw_score=float(rank), segment_id=sid,
        )
        for i, (sid, vid, start, rank) in enumerate(rows)
    ]


def dense_candidates(conn, query_vec: np.ndarray, *, k: int = 50) -> list[Candidate]:
    """Dense candidates ranked by cosine distance (HNSW) — nearest first."""
    rows = conn.execute(
        """
        SELECT id, video_id, start_sec, embedding <=> %s AS distance
        FROM transcript_segments
        WHERE embedding IS NOT NULL
        ORDER BY distance
        LIMIT %s
        """,
        (query_vec, k),
    ).fetchall()
    return [
        Candidate(
            video_id=vid, ts_sec=start, modality=Modality.TRANSCRIPT,
            rank=i, raw_score=float(dist), segment_id=sid,
        )
        for i, (sid, vid, start, dist) in enumerate(rows)
    ]


def candidates(conn, query: str, embedder, *, k: int = 50) -> list[Candidate]:
    """Hybrid transcript candidates: fuse lexical + dense rankings with RRF.

    Returns Candidates re-ranked by fused score (best first), deduplicated by segment_id.
    ``embedder`` exposes ``embed_query(str) -> np.ndarray``.
    """
    lexical = lexical_candidates(conn, query, k=k)
    dense = dense_candidates(conn, embedder.embed_query(query), k=k)

    by_segment: dict[int, Candidate] = {}
    for c in (*lexical, *dense):
        by_segment.setdefault(c.segment_id, c)

    fused = reciprocal_rank_fusion(
        [[c.segment_id for c in lexical], [c.segment_id for c in dense]], k=60
    )
    out: list[Candidate] = []
    for rank, (segment_id, score) in enumerate(fused):
        base = by_segment[segment_id]
        out.append(
            Candidate(
                video_id=base.video_id, ts_sec=base.ts_sec, modality=Modality.TRANSCRIPT,
                rank=rank, raw_score=score, segment_id=segment_id,
            )
        )
    return out
