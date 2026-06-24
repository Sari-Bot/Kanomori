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

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from kanomori.config import get_settings
from kanomori.db import close_pool, connection
from kanomori.media_source import MediaSourceError, get_media_source, iter_manifest
from kanomori.models import (
    BatchIngestRequest,
    BatchIngestResponse,
    IngestRequest,
    IngestResponse,
    JobStatusResponse,
    SearchResponse,
)
from kanomori.retrieval import merge, screenshot, transcript

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "web" / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

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

    # Coordinator API: remote workers claim/heartbeat/push-stage/complete jobs (bearer-auth'd).
    from kanomori.api.jobs import build_jobs_router

    app.include_router(build_jobs_router())

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
        # content_hash (sha256 of the media) is computed by the register stage when the worker
        # runs — not here, so /ingest stays fast and never blocks hashing a multi-GB file. We
        # enqueue a job with content_hash = NULL and stash the request in stage_status->'request'
        # for the worker to rebuild the IngestContext. register later UPDATEs this same row to the
        # real hash (keyed by job id), so there's exactly one row per job — no md5(path) orphan.
        request_payload = json.dumps(req.model_dump())
        with connection() as conn:
            row = conn.execute(
                """
                INSERT INTO jobs (content_hash, status, stage_status)
                VALUES (NULL, 'queued', jsonb_build_object('request', %s::jsonb))
                RETURNING id
                """,
                (request_payload,),
            ).fetchone()
            conn.commit()
        return IngestResponse(job_id=row[0], content_hash=None, status="queued")

    @app.post("/ingest/batch", response_model=BatchIngestResponse)
    def ingest_batch(req: BatchIngestRequest) -> BatchIngestResponse:
        # Enqueue one job per manifest line, the same way /ingest enqueues a single request:
        # content_hash = NULL (the register stage resolves it when the worker runs) and the
        # verbatim manifest record stashed in stage_status->'request' for the worker to rebuild
        # its IngestContext. The record's `path` field is the worker's canonical source key, so we
        # store the record as-is (no media_path translation). Re-running a batch is idempotent:
        # a record whose `path` already has a queued/running job is skipped, not re-enqueued.
        source = get_media_source()
        try:
            records = iter_manifest(source, req.manifest_path)
        except MediaSourceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        enqueued: list[int] = []
        skipped: list[str] = []
        with connection() as conn:
            for record in records:
                path = record.get("path")
                dup = conn.execute(
                    """
                    SELECT 1 FROM jobs
                    WHERE status IN ('queued', 'running')
                      AND stage_status->'request'->>'path' = %s
                    LIMIT 1
                    """,
                    (path,),
                ).fetchone()
                if dup is not None:
                    skipped.append(path)
                    continue
                row = conn.execute(
                    """
                    INSERT INTO jobs (content_hash, status, stage_status)
                    VALUES (NULL, 'queued', jsonb_build_object('request', %s::jsonb))
                    RETURNING id
                    """,
                    (json.dumps(record),),
                ).fetchone()
                enqueued.append(row[0])
            conn.commit()
        return BatchIngestResponse(enqueued=enqueued, skipped=skipped, total=len(records))

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

    @app.get("/result/{video_id}")
    def result(video_id: int, ts: float = 0.0):
        """Moment-detail view: video + source link, nearby transcript, preview frames, OCR,
        scene_type at the timestamp. 404 if the video is unknown."""
        from dataclasses import asdict

        from kanomori.retrieval.result import result_detail

        with connection() as conn:
            detail = result_detail(conn, video_id, ts)
        if detail is None:
            raise HTTPException(status_code=404, detail="video not found")
        return asdict(detail)

    # --- Server-rendered UI (Jinja2 + htmx) -------------------------------------------
    _register_ui(app)

    # Serve derived media (short preview thumbnails only — never source video).
    media_root = Path(get_settings().media_root)
    media_root.mkdir(parents=True, exist_ok=True)
    app.mount("/media", StaticFiles(directory=str(media_root)), name="media")

    return app


def _hit_snippet(conn, video_id: int, ts_sec: float) -> str | None:
    """Best transcript text at/around a hit's timestamp, for display on a result card."""
    row = conn.execute(
        "SELECT text FROM transcript_segments "
        "WHERE video_id = %s ORDER BY abs(start_sec - %s) LIMIT 1",
        (video_id, ts_sec),
    ).fetchone()
    return row[0] if row else None


def _register_ui(app: FastAPI) -> None:
    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return templates.TemplateResponse(request, "index.html")

    @app.post("/ui/search/transcript", response_class=HTMLResponse)
    def ui_search_transcript(request: Request, query: str = Form(""), k: int = Form(10)):
        hits_view: list[dict] = []
        if query.strip():
            with connection() as conn:
                cands = transcript.candidates(conn, query, get_embedder(), k=k)
                hits = merge.merge_from_db(conn, cands, k=k)
                for h in hits:
                    d = h.model_dump()
                    d["snippet"] = _hit_snippet(conn, h.video_id, h.ts_sec)
                    hits_view.append(d)
        return templates.TemplateResponse(request, "_results.html", {"hits": hits_view})

    @app.post("/ui/search/screenshot", response_class=HTMLResponse)
    async def ui_search_screenshot(
        request: Request, file: UploadFile = SCREENSHOT_FILE, k: int = SCREENSHOT_K
    ):
        image = await file.read()
        with connection() as conn:
            cands = screenshot.candidates(
                conn, image, get_image_embedder(), ocr_reader=get_ocr_reader(), k=k
            )
            hits = merge.merge_from_db(conn, cands, k=k)
            hits_view = []
            for h in hits:
                d = h.model_dump()
                d["snippet"] = _hit_snippet(conn, h.video_id, h.ts_sec)
                hits_view.append(d)
        return templates.TemplateResponse(request, "_results.html", {"hits": hits_view})

    @app.get("/ui/result/{video_id}", response_class=HTMLResponse)
    def ui_result(request: Request, video_id: int, ts: float = 0.0):
        from kanomori.retrieval.result import result_detail

        with connection() as conn:
            detail = result_detail(conn, video_id, ts)
        if detail is None:
            raise HTTPException(status_code=404, detail="video not found")
        return templates.TemplateResponse(request, "result.html", {"detail": detail})


app = create_app()
