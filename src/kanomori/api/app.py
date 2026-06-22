"""FastAPI app factory and endpoints.

``create_app`` wires a lifespan that opens the DB pool and preloads the text embedder once
(so the CPU query path doesn't pay model cold-start per request), and registers the routes:

- ``POST /search/transcript`` — hybrid transcript search → ranked moment hits.
- ``POST /ingest`` — enqueue an ingestion job (the worker runs it); returns the job id.
- ``GET  /ingest/{job_id}`` — report job status.

``get_embedder`` is module-level so tests can override it with a fake; production builds the
real BGE-M3 embedder lazily inside the lifespan.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from kanomori.db import close_pool, connection
from kanomori.models import IngestRequest, IngestResponse, JobStatusResponse, SearchResponse
from kanomori.retrieval import merge, screenshot, transcript

_embedder = None
_image_embedder = None
_ocr_reader = None
SCREENSHOT_FILE = File(...)
SCREENSHOT_K = Form(10)


def get_embedder():
    """Return the process-wide text embedder, constructing it on first use.

    Overridden in tests with a deterministic fake. Production uses BGE-M3 (lazy import keeps
    torch out of module import time).
    """
    global _embedder
    if _embedder is None:
        from kanomori.embed.text_embedder import BGEEmbedder

        _embedder = BGEEmbedder()
    return _embedder


def get_image_embedder():
    """Return the process-wide image embedder, constructing it only for screenshot search."""
    global _image_embedder
    if _image_embedder is None:
        from kanomori.embed.image_embedder import DINOv2Embedder

        _image_embedder = DINOv2Embedder()
    return _image_embedder


def get_ocr_reader():
    """Return the process-wide upload OCR reader for screenshot search."""
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = screenshot.UploadOcrReader()
    return _ocr_reader


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the embedder once on startup so the first query isn't slow.
    get_embedder()
    yield
    close_pool()


def create_app() -> FastAPI:
    app = FastAPI(title="Kanomori", lifespan=lifespan)

    @app.post("/search/transcript", response_model=SearchResponse)
    def search_transcript(req: dict) -> SearchResponse:
        query = (req or {}).get("query", "")
        k = (req or {}).get("k", 10)
        with connection() as conn:
            cands = transcript.candidates(conn, query, get_embedder(), k=k)
            hits = merge.merge_from_db(conn, cands, k=k)
        return SearchResponse(hits=hits)

    @app.post("/search/screenshot", response_model=SearchResponse)
    async def search_screenshot(
        file: UploadFile = SCREENSHOT_FILE,
        k: int = SCREENSHOT_K,
    ) -> SearchResponse:
        image = await file.read()
        with connection() as conn:
            cands = screenshot.candidates(
                conn,
                image,
                get_image_embedder(),
                ocr_reader=get_ocr_reader(),
                k=k,
            )
            hits = merge.merge_from_db(conn, cands, k=k)
        return SearchResponse(hits=hits)

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(req: IngestRequest) -> IngestResponse:
        # content_hash is computed by the register stage from the file; for the queue we use a
        # cheap placeholder key (md5 of the path) so /ingest stays fast and never blocks on
        # hashing a multi-GB file. register reconciles to the real sha256 when the worker runs.
        queue_key = hashlib.md5(req.media_path.encode()).hexdigest()  # noqa: S324 - not security
        request_payload = json.dumps(req.model_dump())
        with connection() as conn:
            row = conn.execute(
                """
                INSERT INTO jobs (content_hash, status, stage_status)
                VALUES (%s, 'queued', jsonb_build_object('request', %s::jsonb))
                ON CONFLICT (content_hash) DO UPDATE SET status = 'queued'
                RETURNING id
                """,
                (queue_key, request_payload),
            ).fetchone()
            conn.commit()
        return IngestResponse(job_id=row[0], content_hash=queue_key, status="queued")

    @app.get("/ingest/{job_id}", response_model=JobStatusResponse)
    def ingest_status(job_id: int) -> JobStatusResponse:
        with connection() as conn:
            row = conn.execute(
                "SELECT status, current_stage, stage_status, error FROM jobs WHERE id = %s",
                (job_id,),
            ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="job not found")
        return JobStatusResponse(
            job_id=job_id, status=row[0], current_stage=row[1],
            stage_status=row[2] or {}, error=row[3],
        )

    return app


app = create_app()
