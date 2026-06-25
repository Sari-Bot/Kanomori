# Distributed Ingestion Usage

This document describes the current distributed ingesting system as implemented in Kanomori.
It is both an operator runbook and a developer reference for the coordinator/worker protocol.

For the high-level product and system rationale, see [README.md](/Users/lb/Documents/Code/kanomori/README.md) and [ARCHITECTURE.md](/Users/lb/Documents/Code/kanomori/docs/ARCHITECTURE.md). This file focuses on how to run the distributed path that exists today.

## System Shape

The distributed path has three boundaries:

- Coordinator: the FastAPI app owns the `jobs` table, all stage persistence, and all derived artifacts under `MEDIA_ROOT/{content_hash}/...`.
- Worker: `kanomori-worker` claims jobs over HTTP, fetches source media from the configured `MediaSource`, runs stage `compute()` locally, and pushes stage results back to the coordinator.
- Source store: read-only input storage. In development this is usually `samples/`; in production it is a WebDAV-backed tree with the same layout.

The worker never talks to PostgreSQL directly in distributed mode. The coordinator is the only process that mutates the database.

## Responsibilities

### Coordinator

- Exposes `POST /ingest` and `POST /ingest/batch` to enqueue work.
- Exposes `/jobs/*` for remote workers:
  - `POST /jobs/claim`
  - `POST /jobs/{job_id}/heartbeat`
  - `POST /jobs/{job_id}/stage/{stage_name}`
  - `POST /jobs/{job_id}/complete`
  - `POST /jobs/{job_id}/fail`
- Persists stage rows and binary artifacts atomically.
- Enforces lease fencing with `lease_epoch`.

### Worker

- Polls the coordinator for claimable jobs.
- Fetches the source video identified by `stage_status.request.path` or `media_path`.
- Runs the stage compute chain locally.
- Sends a heartbeat while long stages are running.
- Pushes each completed stage immediately so resume can continue from the last persisted stage.

### Source Store

- Stores source video and `manifest.jsonl`.
- Does not store derived artifacts.
- Uses identical layout for `local` and `webdav` modes.

## Prerequisites

Shared prerequisites:

- `uv`
- `ffmpeg`
- PostgreSQL with pgvector via `docker compose`
- A sibling KITS checkout at the path resolved by `KANOMORI_KITS_DIR`

Coordinator host:

- Python environment with the app dependencies
- Database access
- Writable `KANOMORI_MEDIA_ROOT`

Worker host:

- Python environment with ingestion/model dependencies
- Access to the source store
- Access to the coordinator URL
- GPU support if you expect real transcription, scene classification, and image embedding performance

Use the aggregate dependency groups for exact syncs:

```bash
uv sync --group worker-cpu
uv sync --group worker-cuda
```

Do not use `uv sync --group ocr-cuda` as a full worker install target; exact sync removes packages
from groups that are not selected. `worker-cuda` is the reproducible full CUDA worker target.

## Configuration

The runtime settings come from `src/kanomori/config.py`. The distributed path depends on these variables.

### Coordinator

Required or normally-set variables:

```bash
KANOMORI_DATABASE_URL=postgresql://kanomori:kanomori@localhost:5433/kanomori
KANOMORI_MEDIA_ROOT=./media
KANOMORI_COORDINATOR_TOKEN=replace-this-shared-secret
```

Relevant notes:

- `KANOMORI_COORDINATOR_TOKEN` protects `/jobs/*`.
- If `KANOMORI_COORDINATOR_TOKEN` is unset, `/jobs/*` fails closed with HTTP `503`.
- The token is shared with workers and sent as `Authorization: Bearer <token>`.

### Worker

Required or normally-set variables:

```bash
KANOMORI_COORDINATOR_URL=http://coordinator-host:8000
KANOMORI_COORDINATOR_TOKEN=replace-this-shared-secret
KANOMORI_MEDIA_SOURCE=local
KANOMORI_MEDIA_SOURCE_ROOT=./samples
KANOMORI_KITS_DIR=/Users/lb/Documents/Code/KITS
KANOMORI_STAGE_PARSE_TRANSCRIPT_DEVICE=cpu
KANOMORI_STAGE_OCR_DEVICE=cpu
KANOMORI_STAGE_CLASSIFY_DEVICE=cpu
KANOMORI_STAGE_IMAGE_EMBED_DEVICE=cpu
```

For WebDAV-backed source media:

```bash
KANOMORI_MEDIA_SOURCE=webdav
KANOMORI_MEDIA_SOURCE_URL=https://dav.example.com/store
KANOMORI_MEDIA_SOURCE_USER=...
KANOMORI_MEDIA_SOURCE_PASSWORD=...
```

Relevant notes:

- `KANOMORI_MEDIA_SOURCE=local` means the worker reads from a local mirror such as `samples/`.
- `KANOMORI_MEDIA_SOURCE=webdav` means the worker reads source files and `manifest.jsonl` via HTTPS GET.
- The distributed worker uses `KANOMORI_COORDINATOR_URL`; the default is `http://localhost:8000`, which only fits a worker colocated with the coordinator.
- Stage-device settings are worker-local only. One worker still claims a whole job and runs every stage locally; these flags only decide whether that worker runs `parse_transcript`, `ocr`, `classify`, and `image_embed` on CPU or GPU.
- `cpu` forces CPU execution for the stage. `gpu` is fail-fast: if the worker cannot initialize that stage on the process-visible default GPU, the stage errors instead of silently dropping to CPU.
- `transcribe` is unchanged and still runs through the KITS subprocess path.
- `KANOMORI_STAGE_OCR_DEVICE=gpu` requires `KANOMORI_INGEST_OCR_BACKEND=cuda` or `tensorrt`. Setting `KANOMORI_INGEST_OCR_BACKEND=onnxruntime` with GPU OCR is rejected.
- Query-time OCR and query-time embedding are not affected by these worker settings.

### WSL GPU OCR Notes

For WSL GPU workers, `KANOMORI_STAGE_OCR_DEVICE=gpu` needs more than just a visible NVIDIA GPU.
The worker process must import the GPU `onnxruntime` wheel and see both the WSL driver libraries
and the NVIDIA runtime libraries shipped in the venv.

Recommended setup:

```bash
uv sync --group worker-cuda
```

If `uv run python -c 'import onnxruntime as ort; print(ort.get_available_providers())'`
still shows only `['AzureExecutionProvider', 'CPUExecutionProvider']`, repair the venv by
removing the CPU `onnxruntime` wheel and reinstalling the GPU wheel cleanly:

```bash
cd /path/to/Kanomori
uv pip uninstall onnxruntime
uv pip install --reinstall onnxruntime-gpu==1.27.0
```

Before launching the worker on WSL, export the CUDA library paths in the same shell:

```bash
KANOMORI_ROOT=/path/to/Kanomori
KANOMORI_NVIDIA_SITE="$KANOMORI_ROOT/.venv/lib/python3.12/site-packages/nvidia"
KANOMORI_NVIDIA_LIBS="$(find "$KANOMORI_NVIDIA_SITE" -mindepth 2 -maxdepth 3 -type f \( -name "libcudart.so*" -o -name "libcudnn.so*" -o -name "libcublas.so*" \) -printf "%h\n" | sort -u | paste -sd: -)"
export LD_LIBRARY_PATH="$KANOMORI_NVIDIA_LIBS:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

After that, this probe should show `CUDAExecutionProvider`:

```bash
uv run python - <<'PY'
from kanomori.ocr_tensorrt import ensure_cuda_ep_available
from kanomori.ocr import OcrBackendUnavailable
import onnxruntime as ort

ensure_cuda_ep_available(OcrBackendUnavailable)
print(ort.get_available_providers())
PY
```

## Source Store Layout

The distributed system expects the same layout described in [samples/README.md](/Users/lb/Documents/Code/kanomori/samples/README.md).

```text
<root>/
  manifest.jsonl
  <title>_<date>/
    video.mp4
```

`manifest.jsonl` is authoritative. Each line is one JSON object. Example:

```json
{"path":"鹿乃的2月18日歌回直播_2024-02-18/video.mp4","title":"鹿乃的2月18日歌回直播","streamed_at":"2024-02-18","source_platform":"bilibili","source_url":"https://...","separate":true}
```

Important details:

- `path` is the worker's canonical source key.
- `POST /ingest/batch` stores the full manifest record under `jobs.stage_status.request`.
- Re-running the same batch skips records whose `path` already has a `queued` or `running` job.
- `separate: true` is the explicit override for singing streams when you want KITS vocal isolation even if the title heuristic would not catch it.

## Bring-Up Sequence

### 1. Start the database and apply migrations

```bash
docker compose up -d
uv run kanomori-migrate
```

### 2. Start the coordinator

```bash
uv run uvicorn kanomori.api.app:app --host 0.0.0.0 --port 8000
```

The coordinator must be able to write derived artifacts under `KANOMORI_MEDIA_ROOT`.

### 3. Start one or more workers

Continuous polling worker:

```bash
uv run kanomori-worker --worker-id gpu-worker-01
```

One-shot worker pass:

```bash
uv run kanomori-worker --once --worker-id gpu-worker-01
```

Useful flags:

```bash
uv run kanomori-worker --help
```

The implemented CLI supports:

- `--once`
- `--compute-only`
- `--source {local,webdav}`
- `--worker-id`
- `--manifest-index`
- `--media-path`
- `--lease-seconds`
- `--poll-interval`

## Enqueueing Work

### Single job

`POST /ingest` accepts a local-ingest request shape. In the distributed model, `media_path` is stored in the request payload and later reused by the worker as the source-store key when `path` is absent.

Example:

```bash
curl -X POST http://localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "media_path": "2024_talk_cut/video.mp4",
    "title": "2024 talk cut",
    "source_platform": "local",
    "source_url": "file://samples/2024_talk_cut/video.mp4",
    "separate": false
  }'
```

Expected response shape:

```json
{"job_id": 123, "content_hash": null, "status": "queued"}
```

### Batch enqueue

`POST /ingest/batch` reads a manifest from the configured source store.

Default manifest:

```bash
curl -X POST http://localhost:8000/ingest/batch \
  -H 'Content-Type: application/json' \
  -d '{"manifest_path":"manifest.jsonl"}'
```

Custom manifest path:

```bash
curl -X POST http://localhost:8000/ingest/batch \
  -H 'Content-Type: application/json' \
  -d '{"manifest_path":"batches/phase1.jsonl"}'
```

Expected response shape:

```json
{
  "enqueued": [101, 102],
  "skipped": ["2024_talk_cut/video.mp4"],
  "total": 3
}
```

## Monitoring

### Poll job status

Use `GET /ingest/{job_id}`:

```bash
curl http://localhost:8000/ingest/123
```

Response fields:

- `status`: overall job state
- `current_stage`: the stage currently being worked on
- `stage_status`: per-stage progress plus the stored request payload
- `time_costs`: ordered per-stage compute durations as `{"stage": "<name>", "seconds": <float>}`
- `error`: failure string, if any

`time_costs` is empty for queued jobs. It records only terminal successful stages (`done` and
`skipped`), uses pipeline stage order, and replaces a prior entry when a stage is rerun
successfully. Total job compute time is derived by summing `time_costs[].seconds`.

Typical values:

- `queued`
- `running`
- `complete`
- `failed`

### Watch worker logs

At startup the worker prints:

```text
kanomori distributed worker <worker_id> started; polling <coordinator_url>
```

This is the quickest confirmation that the process is in distributed mode rather than `--compute-only`.

## Dry Runs

The worker supports a no-coordinator dry run that exercises the compute path and StageResult serialization without touching the database or `/jobs/*`.

From the local source mirror:

```bash
uv run kanomori-worker --compute-only --source local --manifest-index 0
```

Run against a specific source-store path:

```bash
uv run kanomori-worker --compute-only --source local --media-path 2024_talk_cut/video.mp4
```

Use this before involving the coordinator when validating worker dependencies or model setup on a new machine.

## Failure and Recovery

The distributed system is resumable by stage, not by an in-memory worker session.

What happens on failure:

- Each claim returns a `lease_epoch`.
- The worker heartbeats during long stages.
- Every mutating `/jobs/*` call includes `lease_epoch`.
- If the lease expires and another worker reclaims the job, stale updates get HTTP `409`.
- The worker must stop when fenced; it no longer owns the job.

What gets resumed:

- The coordinator records completed stages in `stage_status`.
- On reclaim, the next worker receives `stages_done` and skips already-persisted stages.
- Stage artifacts are pushed at stage boundaries, so a crash after `transcribe` should not force transcription to rerun if the SRT was already persisted.

Retry cap:

- Failed jobs are only reclaimable while `attempts < MAX_ATTEMPTS`.
- `MAX_ATTEMPTS` is currently `3` in [worker.py](/Users/lb/Documents/Code/kanomori/src/kanomori/ingest/worker.py).

Operational consequences:

- A stuck or disconnected worker should not leave a job permanently wedged in `running`.
- A permanently broken job eventually stays `failed` until an operator intervenes.

## Developer Notes

### Stage split

The distributed path depends on the compute/persist split:

- `compute(ctx) -> StageResult`
- `persist(conn, video_id, result)` on the coordinator

The single-machine path still calls `run(conn, ctx)` and remains valid. Distributed workers call `compute()` locally and the coordinator calls `persist()` after receiving the stage payload.

### Wire contract

The `/jobs/{job_id}/stage/{stage_name}` endpoint accepts:

- form field `lease_epoch`
- optional form field `compute_seconds` for compute-only stage wall time, rounded to 3 decimals
- optional file field `result_file` containing the UTF-8 JSON-serialized `StageResult`
- multipart file field `files` for binary artifacts such as frame JPEGs or the SRT

Model-bearing stages upload `result_file`; model-less stages such as `locate_media` omit it.
The coordinator enforces a stage-result upload cap via `KANOMORI_STAGE_RESULT_MAX_BYTES`
(default `67108864` bytes) and returns `413` when exceeded.

`compute_seconds` is persisted atomically with stage completion. A failed push or fenced worker
(`409`) does not create a `time_costs` entry.

Artifacts are written to deterministic paths:

- frames: `MEDIA_ROOT/{content_hash}/frames/...`
- transcript: `MEDIA_ROOT/{content_hash}/transcript.srt`

### Claim and heartbeat semantics

`POST /jobs/claim` returns either:

- `200` with `{job_id, content_hash, lease_epoch, request, stages_done}`
- `204` when the coordinator is idle

`POST /jobs/{job_id}/heartbeat` returns:

- `200` when the lease was extended
- `409` when the worker is stale and has been fenced off

The worker heartbeat interval is `max(1.0, lease_seconds / 3)`.

### Fencing

Fencing is the core protection against zombie workers:

- the coordinator reads and locks the job row
- it compares the submitted `lease_epoch` with the current one
- mismatch means another worker reclaimed the job
- the coordinator returns `409`
- the stale worker aborts without completing the job

### Batch dedup

`POST /ingest/batch` currently deduplicates only against jobs already in `queued` or `running` state by comparing:

```text
stage_status->'request'->>'path'
```

That means:

- re-running a batch while jobs are still queued/running is idempotent
- completed or failed records are not filtered by this enqueue-time dedup alone

## Code Map

Primary implementation files:

- [src/kanomori/api/app.py](/Users/lb/Documents/Code/kanomori/src/kanomori/api/app.py)
- [src/kanomori/api/jobs.py](/Users/lb/Documents/Code/kanomori/src/kanomori/api/jobs.py)
- [src/kanomori/ingest/worker.py](/Users/lb/Documents/Code/kanomori/src/kanomori/ingest/worker.py)
- [src/kanomori/ingest/coordinator_client.py](/Users/lb/Documents/Code/kanomori/src/kanomori/ingest/coordinator_client.py)
- [src/kanomori/ingest/lease.py](/Users/lb/Documents/Code/kanomori/src/kanomori/ingest/lease.py)
- [src/kanomori/media_source.py](/Users/lb/Documents/Code/kanomori/src/kanomori/media_source.py)
- [src/kanomori/config.py](/Users/lb/Documents/Code/kanomori/src/kanomori/config.py)
- [samples/README.md](/Users/lb/Documents/Code/kanomori/samples/README.md)
- [docs/plans/2026-06-24-distributed-ingestion-design.md](/Users/lb/Documents/Code/kanomori/docs/plans/2026-06-24-distributed-ingestion-design.md)

Use the design note for intent and trade-offs. Use the code files above as the current source of truth when behavior diverges.
