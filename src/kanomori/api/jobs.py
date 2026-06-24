"""The coordinator ``/jobs/*`` router: remote workers claim, heartbeat, push stages, finish.

This is the HTTP half of Task #13. A worker (#14) does the GPU compute off-box and talks to the
coordinator, which owns the database, through these endpoints:

* ``POST /jobs/claim``               — lease the oldest eligible job (or 204 when idle).
* ``POST /jobs/{id}/heartbeat``      — extend the lease mid-stage.
* ``POST /jobs/{id}/stage/{name}``   — multipart: a ``result_file`` JSON upload + the stage's
                                       binary artifacts (frame JPEGs / the SRT). Persisted
                                       atomically.
* ``POST /jobs/{id}/complete``       — mark the job done.
* ``POST /jobs/{id}/fail``           — mark the job failed (records error, bumps attempts).

Every endpoint requires ``Authorization: Bearer <coordinator_token>``. The token is the shared
secret in ``Settings.coordinator_token``. **Fail closed:** if it is unset we reject all /jobs
calls with 503 rather than silently running auth-disabled — a misconfigured coordinator must not
accept anonymous mutations. Fencing: every mutation carries the ``lease_epoch`` the worker holds;
a stale epoch (the job was re-claimed) -> 0 rows -> HTTP 409, telling the zombie worker to stop.
"""

from __future__ import annotations

import hmac
import json

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from kanomori.config import get_settings
from kanomori.db import connection
from kanomori.ingest import lease
from kanomori.ingest.artifacts import frame_dir_for, srt_path_for

# FastAPI dependency defaults are constructed once at import (B008-safe vs inline calls).
LEASE_EPOCH_FORM = Form(...)
STAGE_RESULT_FILE = File(default=None)
STAGE_FILES = File(default_factory=list)


# --- request/response models -------------------------------------------------------------


class ClaimRequest(BaseModel):
    worker_id: str
    lease_seconds: int = lease.DEFAULT_LEASE_SECONDS


class ClaimResponse(BaseModel):
    job_id: int
    content_hash: str | None
    lease_epoch: int
    request: dict
    stages_done: list[str]


class HeartbeatRequest(BaseModel):
    lease_epoch: int
    lease_seconds: int = lease.DEFAULT_LEASE_SECONDS


class CompleteRequest(BaseModel):
    lease_epoch: int


class FailRequest(BaseModel):
    lease_epoch: int
    error: str


class OkResponse(BaseModel):
    ok: bool = True


# --- auth --------------------------------------------------------------------------------


def require_coordinator_token(authorization: str | None = Header(default=None)) -> None:
    """Gate every /jobs call on the shared bearer token. Fail closed when unconfigured.

    503 (not 401) when no token is set: that's an operator misconfiguration, not a bad client
    credential — surfacing it distinctly stops the coordinator from ever running auth-disabled.
    Token comparison is constant-time (``hmac.compare_digest``) and the token is never logged.
    """
    token = get_settings().coordinator_token
    if not token:
        raise HTTPException(status_code=503, detail="coordinator token not configured")
    expected = f"Bearer {token}"
    if not authorization or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


# --- artifact routing --------------------------------------------------------------------


def _save_artifacts(stage_name: str, content_hash: str | None, files: list[UploadFile]) -> None:
    """Write a stage's uploaded binaries to their deterministic content-hash-keyed locations.

    frames -> ``frame_dir_for(content_hash)/<name>`` (one JPEG per FrameRow artifact name);
    transcribe -> ``srt_path_for(content_hash)``. ``content_hash`` must already be set on the
    job (register runs first and reconciles it) for any stage that carries artifacts. Filenames
    are reduced to their basename to prevent path traversal outside the artifact dir.
    """
    spec = lease.STAGE_SPECS[stage_name]
    if not files or spec.artifact_kind is None:
        return
    if content_hash is None:
        raise HTTPException(
            status_code=409,
            detail=f"cannot store {stage_name} artifacts before register sets content_hash",
        )

    if spec.artifact_kind == "frame":
        frame_dir = frame_dir_for(content_hash)
        frame_dir.mkdir(parents=True, exist_ok=True)
        for upload in files:
            name = _safe_name(upload.filename)
            (frame_dir / name).write_bytes(upload.file.read())
    elif spec.artifact_kind == "srt":
        srt_path = srt_path_for(content_hash)
        srt_path.parent.mkdir(parents=True, exist_ok=True)
        # transcribe declares exactly one SRT artifact; take the first uploaded file.
        srt_path.write_bytes(files[0].file.read())


def _safe_name(filename: str | None) -> str:
    """Basename only — strip any directory components a client may have attached."""
    from pathlib import PurePosixPath

    if not filename:
        raise HTTPException(status_code=400, detail="artifact missing a filename")
    name = PurePosixPath(filename).name
    if not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="invalid artifact filename")
    return name


# --- router ------------------------------------------------------------------------------


def build_jobs_router() -> APIRouter:
    """Construct the /jobs router. Auth is enforced on every route via the dependency."""
    router = APIRouter(prefix="/jobs", dependencies=[Depends(require_coordinator_token)])

    @router.post("/claim")
    def claim_job(req: ClaimRequest):
        with connection() as conn:
            claimed = lease.claim(conn, req.worker_id, req.lease_seconds)
            conn.commit()
        if claimed is None:
            return Response(status_code=204)
        return JSONResponse(ClaimResponse(**claimed).model_dump())

    @router.post("/{job_id}/heartbeat", response_model=OkResponse)
    def heartbeat_job(job_id: int, req: HeartbeatRequest):
        with connection() as conn:
            ok = lease.heartbeat(conn, job_id, req.lease_epoch, req.lease_seconds)
            conn.commit()
        if not ok:
            raise HTTPException(status_code=409, detail="stale lease epoch")
        return OkResponse()

    @router.post("/{job_id}/stage/{stage_name}", response_model=OkResponse)
    def push_stage(
        job_id: int,
        stage_name: str,
        lease_epoch: int = LEASE_EPOCH_FORM,
        result_file: UploadFile | None = STAGE_RESULT_FILE,
        files: list[UploadFile] = STAGE_FILES,
    ):
        """Persist one stage's result + artifacts atomically, fenced on ``lease_epoch``.

        One transaction: fence-check the epoch (409 if stale), parse ``result_file`` into the
        stage's model, save uploaded artifacts to disk, call the stage's ``persist``, then mark
        the stage done. Either the whole stage lands or none of it does.
        """
        spec = lease.STAGE_SPECS.get(stage_name)
        if spec is None:
            raise HTTPException(status_code=404, detail=f"unknown stage {stage_name!r}")

        with connection() as conn:
            # Lock the job row and read identity; fence on the current epoch in the same txn.
            # The fence is checked before parsing the payload so a stale (re-claimed) worker is
            # told to stop (409) regardless of what it sends.
            row = conn.execute(
                "SELECT content_hash, video_id, lease_epoch FROM jobs WHERE id=%s FOR UPDATE",
                (job_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="job not found")
            content_hash, video_id, current_epoch = row
            if current_epoch != lease_epoch:
                conn.rollback()
                raise HTTPException(status_code=409, detail="stale lease epoch")

            parsed = _parse_result(spec, result_file)
            _save_artifacts(stage_name, content_hash, files)

            new_video_id = lease.persist_stage(
                conn, stage_name, parsed, job_id=job_id, video_id=video_id,
                content_hash=content_hash,
            )
            if new_video_id is not None:
                video_id = new_video_id  # register produced it; downstream stages need it

            if not lease.mark_stage_done(conn, job_id, lease_epoch, stage_name):
                conn.rollback()
                raise HTTPException(status_code=409, detail="stale lease epoch")
            conn.commit()
        return OkResponse()

    @router.post("/{job_id}/complete", response_model=OkResponse)
    def complete_job(job_id: int, req: CompleteRequest):
        with connection() as conn:
            ok = lease.complete(conn, job_id, req.lease_epoch)
            conn.commit()
        if not ok:
            raise HTTPException(status_code=409, detail="stale lease epoch")
        return OkResponse()

    @router.post("/{job_id}/fail", response_model=OkResponse)
    def fail_job(job_id: int, req: FailRequest):
        with connection() as conn:
            ok = lease.fail(conn, job_id, req.lease_epoch, req.error)
            conn.commit()
        if not ok:
            raise HTTPException(status_code=409, detail="stale lease epoch")
        return OkResponse()

    return router


def _parse_result(spec: lease.StageSpec, result_file: UploadFile | None):
    """Parse the optional multipart ``result_file`` into the stage's StageResult model.

    No-DB stages (locate_media) carry no model and accept an absent ``result_file`` — we hand
    persist a None the no-op ignores. Oversized result files are rejected 413; malformed JSON or a
    payload that fails model validation is a client error (400), not a server fault.
    """
    if spec.model is None:
        return None
    text = _read_result_text(result_file)
    if not text.strip():
        raise HTTPException(status_code=400, detail="missing result payload for this stage")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid result JSON: {exc}") from exc
    try:
        return spec.model.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - surface validation failure as a 400
        raise HTTPException(status_code=400, detail=f"result failed validation: {exc}") from exc


def _read_result_text(result_file: UploadFile | None) -> str:
    if result_file is None:
        return ""
    max_bytes = get_settings().stage_result_max_bytes
    payload = result_file.file.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"stage result file exceeded maximum size of {max_bytes} bytes",
        )
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid result UTF-8: {exc}") from exc
