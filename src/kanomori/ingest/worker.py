"""The ingestion worker: single-machine DB-direct loop *and* a distributed coordinator client.

Two run modes share this module:

* **Single-machine (#12, unchanged).** ``claim_one`` locks a queued/failed job with
  ``FOR UPDATE SKIP LOCKED`` and marks it running; ``claim_and_run_one`` runs the whole pipeline
  in-process against the local database, marking the job failed (recording the error, bumping
  ``attempts``) on exception. Kept intact for the local MVP deployment.

* **Distributed (#14).** ``run_one_distributed`` claims a job *over HTTP* from the coordinator,
  fetches the source video from the configured :class:`MediaSource`, runs each stage's
  ``compute(ctx)`` locally (GPU-tolerant), and pushes per-stage results + artifacts back to the
  coordinator — with a background heartbeat extending the lease, resume (skip ``stages_done``),
  and fencing (a 409 from any mutation means the lease was lost: stop without completing). The
  coordinator owns the database; this worker never touches it.

``main`` is the ``kanomori-worker`` entry point. By default it runs the distributed loop; flags
select one-shot (``--once``) or a no-coordinator dry run (``--compute-only``) that proves the
compute chain runs on a manifest sample and every StageResult round-trips on the wire.
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import sys
import threading
import time
from pathlib import Path

from kanomori.config import get_settings
from kanomori.ingest.artifacts import frame_dir_for, srt_path_for
from kanomori.ingest.coordinator_client import CoordinatorClient
from kanomori.ingest.lease import DEFAULT_LEASE_SECONDS, STAGE_SPECS
from kanomori.ingest.pipeline import STAGES, IngestContext, make_embedder, run_full
from kanomori.media_source import MediaSource, get_media_source, iter_manifest

# A job is retried on failure (per-stage status makes the retry resume, not restart), but a
# permanently-broken job must not loop forever. Once a failed job reaches MAX_ATTEMPTS it is
# left in 'failed' and no longer claimed; an operator can reset attempts to requeue it.
MAX_ATTEMPTS = 3


class _WorkerLog:
    """Minimal line-oriented worker logging with optional verbose detail lines."""

    def __init__(self, worker_id: str, *, verbose: bool = False) -> None:
        self.worker_id = worker_id
        self.verbose = verbose

    def progress(self, message: str) -> None:
        print(f"[worker] {message}", flush=True)

    def detail(self, message: str) -> None:
        if self.verbose:
            self.progress(message)

    def error(self, message: str) -> None:
        print(f"[worker] {message}", file=sys.stderr, flush=True)


# =========================================================================================
# Single-machine path (DB-direct) — unchanged from #12; kept for the local MVP deployment.
# =========================================================================================


def claim_one(conn) -> dict | None:
    """Lock and claim the oldest eligible queued/failed job; mark it running. None if none.

    Eligible = queued, or failed with attempts below MAX_ATTEMPTS (so a permanently-failing
    job stops being re-claimed once it exhausts its retries). Returns a dict with the job id
    and the request fields needed to build an IngestContext.
    """
    row = conn.execute(
        """
        SELECT id, content_hash, stage_status -> 'request' AS request
        FROM jobs
        WHERE status = 'queued'
           OR (status = 'failed' AND attempts < %s)
        ORDER BY id
        FOR UPDATE SKIP LOCKED
        LIMIT 1
        """,
        (MAX_ATTEMPTS,),
    ).fetchone()
    if row is None:
        return None

    job_id, content_hash, request = row
    request = request or {}
    conn.execute(
        "UPDATE jobs SET status = 'running', updated_at = now() WHERE id = %s", (job_id,)
    )
    conn.commit()
    return {"id": job_id, "content_hash": content_hash, **request}


def _context_from_claim(claim: dict) -> IngestContext:
    return IngestContext(
        media_path=claim["media_path"],
        source_url=claim.get("source_url"),
        source_platform=claim.get("source_platform"),
        title=claim.get("title"),
        stream_type=claim.get("stream_type"),
        separate=claim.get("separate", False),
    )


def claim_and_run_one(conn) -> bool:
    """Claim one job and run the pipeline. Returns True if a job ran, False if idle.

    On failure, marks the job failed, records the error, and bumps attempts — leaving it
    eligible for a later retry (per-stage status means the retry resumes, not restarts).
    """
    claim = claim_one(conn)
    if claim is None:
        return False

    try:
        run_full(conn, _context_from_claim(claim))
    except Exception as exc:  # noqa: BLE001 - record any failure on the job and move on
        conn.rollback()
        conn.execute(
            """
            UPDATE jobs
            SET status = 'failed', error = %s, attempts = attempts + 1, updated_at = now()
            WHERE id = %s
            """,
            (str(exc), claim["id"]),
        )
        conn.commit()
    return True


# =========================================================================================
# Distributed path (#14) — claim/compute/push/heartbeat against the coordinator over HTTP.
# =========================================================================================


def _source_store_path(request: dict) -> str:
    """The source-store-relative path the worker fetches from the MediaSource.

    ``/ingest/batch`` (manifest) records the store-relative ``path``; plain ``/ingest`` records
    only ``media_path`` (a path on the legacy ingestion host, reused as the store key in the
    distributed model). Prefer ``path`` when present, else fall back to ``media_path``.
    """
    path = request.get("path") or request.get("media_path")
    if not path:
        raise ValueError("ingest request carries neither 'path' nor 'media_path'")
    return path


def _cache_dest(cache_dir: Path, store_path: str) -> Path:
    """Local cache location for a fetched source, keyed by a sanitized store path.

    Self-reclaim optimization: a worker that already fetched this source (e.g. a re-claimed job
    after a transient failure) reuses the on-disk copy instead of re-downloading. The key is the
    store path with path separators flattened, preserving the suffix so ffmpeg/ffprobe still see a
    real extension.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", store_path)
    return Path(cache_dir) / safe


def _context_from_request(request: dict, media_path: str, job_id: int, embedder) -> IngestContext:
    """Build the IngestContext for a distributed run from the enqueued request payload."""
    return IngestContext(
        media_path=media_path,
        source_url=request.get("source_url"),
        source_platform=request.get("source_platform"),
        title=request.get("title"),
        stream_type=request.get("stream_type"),
        separate=request.get("separate", False),
        language=request.get("language", "japanese"),
        job_id=job_id,
        embedder=embedder,
    )


def _empty_result_for(stage_name: str):
    """A valid-but-empty StageResult for a stage that compute() reported as "skipped".

    The coordinator still needs the stage marked done (so resume skips it), and its persist is a
    harmless no-op on empty rows. locate_media has no model -> None (pushed as "").
    """
    model = STAGE_SPECS[stage_name].model
    if model is None:
        return None
    empties = {
        "parse_transcript": {"segments": []},
        "frames": {"frames": [], "scene_timestamps": []},
        "ocr": {"rows": []},
        "classify": {"segments": [], "stream_type": "chatting"},
        "image_embed": {"rows": []},
        "transcribe": {},
        "register": {},
    }
    return model.model_validate(empties.get(stage_name, {}))


def _gather_artifacts(stage_name: str, result, content_hash: str) -> list[tuple[str, bytes]]:
    """Read the binary artifacts a stage produced, as ``(name, bytes)`` for the multipart push.

    Uses ``STAGE_SPECS[stage].artifact_kind`` (the shared registry — no second hardcoded table):
    ``"frame"`` reads each JPEG named by ``FramesResult.artifacts()`` from ``frame_dir_for``;
    ``"srt"`` reads the single ``srt_path_for``. Other stages carry no artifacts.
    """
    kind = STAGE_SPECS[stage_name].artifact_kind
    if kind is None or result is None:
        return []
    if kind == "frame":
        frame_dir = frame_dir_for(content_hash)
        files: list[tuple[str, bytes]] = []
        for ref in result.artifacts():
            path = frame_dir / ref.name
            files.append((ref.name, path.read_bytes()))
        return files
    if kind == "srt":
        srt_path = srt_path_for(content_hash)
        return [(srt_path.name, srt_path.read_bytes())]
    return []


def _result_json(result) -> str:
    """Serialize a StageResult to the wire ``result`` field; "" for the no-model stage."""
    if result is None:
        return ""
    return result.model_dump_json()


class _Heartbeat:
    """Background lease-extender: pings the coordinator every ``interval`` until stopped.

    Long stages (transcribe, frames) can outlast a single lease window, so a daemon thread
    heartbeats while a stage computes. If a heartbeat returns False the lease was lost (the job
    was re-claimed and our epoch is stale): ``fenced`` is set so the run loop aborts at the next
    checkpoint instead of pushing into a job it no longer owns.
    """

    def __init__(self, client, job_id: int, lease_epoch: int, lease_seconds: int) -> None:
        self._client = client
        self._job_id = job_id
        self._lease_epoch = lease_epoch
        self._lease_seconds = lease_seconds
        self._interval = max(1.0, lease_seconds / 3)
        self._stop = threading.Event()
        self.fenced = threading.Event()
        self.reason: str | None = None
        self._thread = threading.Thread(target=self._run, name="coord-heartbeat", daemon=True)

    def _run(self) -> None:
        # Wait first, then beat: a freshly-claimed lease is already fresh, so an immediate beat is
        # redundant. ``wait`` returns True when stopped, ending the loop promptly between stages.
        while not self._stop.wait(self._interval):
            try:
                if not self._client.heartbeat(self._job_id, self._lease_epoch, self._lease_seconds):
                    self.reason = "heartbeat-409"
                    self.fenced.set()
                    return
            except Exception:  # noqa: BLE001 - a transient heartbeat error shouldn't kill the run
                continue

    def __enter__(self) -> _Heartbeat:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)


def run_one_distributed(
    client,
    source: MediaSource,
    worker_id: str,
    lease_seconds: int,
    *,
    cache_dir: Path | None = None,
    verbose: bool = False,
    logger: _WorkerLog | None = None,
) -> bool:
    """Claim one job from the coordinator and run it remotely. True if a job ran, False if idle.

    Flow: claim -> fetch source (self-reclaim cache) -> for each pipeline stage not in
    ``stages_done``: ``compute(ctx)`` (skip-aware), gather artifacts, push result + files to the
    coordinator -> complete. A push returning False (HTTP 409) means the lease was lost: stop
    immediately without completing. Any exception is reported via ``fail`` (best-effort).
    """
    log = logger or _WorkerLog(worker_id, verbose=verbose)
    claimed = client.claim(worker_id, lease_seconds)
    if claimed is None:
        log.progress(f"idle worker={worker_id}")
        return False

    job_id = claimed["job_id"]
    lease_epoch = claimed["lease_epoch"]
    request = claimed.get("request") or {}
    stages_done = set(claimed.get("stages_done") or [])
    # content_hash is known after register; on resume the coordinator hands back the prior one.
    content_hash = claimed.get("content_hash")

    if cache_dir is None:
        cache_dir = Path(get_settings().media_root) / "_source_cache"

    try:
        store_path = _source_store_path(request)
        local = _cache_dest(cache_dir, store_path)
        language = request.get("language", "japanese")
        log.progress(f"claimed job={job_id} epoch={lease_epoch} source={store_path}")
        log.detail(
            "detail claimed "
            f"job={job_id} epoch={lease_epoch} content_hash={content_hash or '-'} "
            f"stages_done={','.join(sorted(stages_done)) or '-'} "
            f"separate={request.get('separate', False)} language={language} "
            f"cache_path={local}"
        )
        source_action = "reuse" if local.exists() else "fetch"
        log.progress(f"source job={job_id} action={source_action} source={store_path}")
        log.detail(f"source detail job={job_id} action={source_action} cache_path={local}")
        if not local.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            source.fetch(store_path, local)

        ctx = _context_from_request(request, str(local), job_id, make_embedder())
        ctx.content_hash = content_hash

        with _Heartbeat(client, job_id, lease_epoch, lease_seconds) as hb:
            for name, module in STAGES:
                if name in stages_done:
                    log.progress(f"stage resume-skip job={job_id} stage={name}")
                    continue
                if hb.fenced.is_set():
                    log.error(
                        f"lease-lost job={job_id} epoch={lease_epoch} "
                        f"reason={hb.reason or 'heartbeat'}"
                    )
                    return True  # lease lost mid-run; stop without completing

                log.progress(f"stage start job={job_id} stage={name}")
                outcome = module.compute(ctx)
                if outcome == "skipped":
                    result = _empty_result_for(name)
                    outcome_label = "skipped"
                elif outcome is None and STAGE_SPECS[name].model is not None:
                    # A model-bearing stage that returned nothing: push an empty result so the
                    # coordinator still marks it done (defensive; the real stages don't do this).
                    result = _empty_result_for(name)
                    outcome_label = "empty"
                else:
                    result = outcome
                    outcome_label = "no-model" if STAGE_SPECS[name].model is None else "result"

                # register is the first stage to learn content_hash; capture it for artifact paths.
                if name == "register" and result is not None:
                    content_hash = result.content_hash
                    ctx.content_hash = content_hash

                files = _gather_artifacts(name, result, content_hash)
                log.detail(
                    f"stage detail job={job_id} stage={name} "
                    f"outcome={outcome_label} artifacts={len(files)}"
                )
                if not client.push_stage(
                    job_id, name, lease_epoch, _result_json(result), files
                ):
                    log.error(
                        f"lease-lost job={job_id} epoch={lease_epoch} "
                        f"stage={name} reason=push-409"
                    )
                    return True  # 409: lease lost — stop, don't complete
                log.progress(f"stage done job={job_id} stage={name}")
                log.detail(f"push ok job={job_id} stage={name} artifacts={len(files)}")

        if not client.complete(job_id, lease_epoch):
            log.error(f"lease-lost job={job_id} epoch={lease_epoch} reason=complete-409")
            return True
        log.progress(f"complete job={job_id}")
    except Exception as exc:  # noqa: BLE001 - report any failure to the coordinator and move on
        error = str(exc)
        log.error(f"failed job={job_id} error={error}")
        fail_status = "sent"
        try:
            reported = client.fail(job_id, lease_epoch, error)
            fail_status = "sent" if reported else "fenced"
        except Exception as report_exc:  # noqa: BLE001 - best-effort only
            fail_status = f"error:{report_exc}"
        log.detail(f"fail-report job={job_id} status={fail_status}")
    return True


# =========================================================================================
# --compute-only: no-coordinator dry run over a manifest sample (proves compute + wire shape).
# =========================================================================================


def run_compute_only(
    source: MediaSource,
    *,
    manifest_index: int = 0,
    media_path: str | None = None,
    cache_dir: Path | None = None,
) -> None:
    """Run the full compute() chain over one sample and assert each StageResult round-trips.

    No coordinator, no DB, no push. The sample is the ``--media-path`` override or the
    ``manifest_index``-th manifest entry. Each stage's result is serialized
    (``model_dump_json``) and re-parsed (``model_validate``) to prove the wire contract holds,
    then a per-stage summary is printed (rows produced / artifacts written).
    """
    if cache_dir is None:
        cache_dir = Path(get_settings().media_root) / "_source_cache"

    if media_path is not None:
        store_path = media_path
        request: dict = {"media_path": media_path}
    else:
        records = iter_manifest(source)
        if not records:
            raise SystemExit("--compute-only: manifest is empty; nothing to run")
        record = records[manifest_index]
        store_path = _source_store_path(record)
        request = dict(record)

    local = _cache_dest(cache_dir, store_path)
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        source.fetch(store_path, local)

    ctx = _context_from_request(request, str(local), job_id=0, embedder=make_embedder())

    print(f"--compute-only: {store_path} -> {local}")
    for name, module in STAGES:
        outcome = module.compute(ctx)
        if name == "register" and outcome is not None:
            ctx.content_hash = outcome.content_hash

        if outcome == "skipped":
            print(f"  {name:<16} skipped")
            continue

        model = STAGE_SPECS[name].model
        if model is None or outcome is None:
            print(f"  {name:<16} ok (no model)")
            continue

        payload = outcome.model_dump_json()
        reparsed = model.model_validate_json(payload)
        assert reparsed.model_dump() == outcome.model_dump(), f"{name} did not round-trip"
        artifacts = outcome.artifacts() if hasattr(outcome, "artifacts") else []
        rows = _summarize_rows(outcome)
        print(f"  {name:<16} ok  rows={rows} artifacts={len(artifacts)}")


def _summarize_rows(result) -> int:
    """Best-effort row count for the --compute-only summary (0 for artifact-only stages)."""
    for attr in ("segments", "rows", "frames"):
        value = getattr(result, attr, None)
        if value is not None:
            return len(value)
    return 0


# =========================================================================================
# Entry point
# =========================================================================================


def _default_worker_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kanomori-worker",
        description="Distributed ingestion worker: claim jobs from the coordinator and run them.",
    )
    parser.add_argument(
        "--once", action="store_true", help="run exactly one claim+run cycle, then exit"
    )
    parser.add_argument(
        "--compute-only",
        action="store_true",
        help="dry run: run the compute chain over a manifest sample, no coordinator/DB/push",
    )
    parser.add_argument(
        "--source",
        choices=["local", "webdav"],
        default=None,
        help="override KANOMORI_MEDIA_SOURCE for this run",
    )
    parser.add_argument("--worker-id", default=None, help="override the worker id")
    parser.add_argument(
        "--manifest-index",
        type=int,
        default=0,
        help="--compute-only: which manifest entry to run (default 0)",
    )
    parser.add_argument(
        "--media-path",
        default=None,
        help="--compute-only: a source-store path to run instead of a manifest entry",
    )
    parser.add_argument(
        "--lease-seconds",
        type=int,
        default=DEFAULT_LEASE_SECONDS,
        help=f"lease length requested on claim (default {DEFAULT_LEASE_SECONDS})",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="seconds to sleep when the coordinator is idle (default 5.0)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print detailed distributed-worker debug lines in addition to progress output",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """``kanomori-worker`` entry point: distributed loop, with --once / --compute-only modes."""
    args = _build_parser().parse_args(argv)

    if args.source is not None:
        os.environ["KANOMORI_MEDIA_SOURCE"] = args.source
        get_settings.cache_clear()

    source = get_media_source()

    if args.compute_only:
        run_compute_only(
            source, manifest_index=args.manifest_index, media_path=args.media_path
        )
        return

    settings = get_settings()
    client = CoordinatorClient(settings.coordinator_url, settings.coordinator_token)
    worker_id = args.worker_id or _default_worker_id()
    logger = _WorkerLog(worker_id, verbose=args.verbose)

    logger.progress(f"startup worker={worker_id} coordinator={settings.coordinator_url}")
    while True:
        ran = run_one_distributed(
            client,
            source,
            worker_id,
            args.lease_seconds,
            verbose=args.verbose,
            logger=logger,
        )
        if args.once:
            break
        if not ran:
            time.sleep(args.poll_interval)


if __name__ == "__main__":  # pragma: no cover
    main()
