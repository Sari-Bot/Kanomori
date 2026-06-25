# Distributed Ingestion

[简体中文](distributed-ingestion.zh-CN.md) | English

This document describes the distributed coordinator/worker path implemented in Kanomori today.

For local setup, start with [Getting Started](getting-started.md). For system rationale, see
[Architecture](ARCHITECTURE.md).

## System roles

- Coordinator: FastAPI app that owns the `jobs` table, stage persistence, and final artifacts
- Worker: `kanomori-worker` process that claims jobs, runs stages locally, and pushes results back
- Source store: read-only input tree, either a local mirror or a WebDAV-backed tree

In distributed mode, the worker does not write PostgreSQL directly. The coordinator is the only
database writer.

## Coordinator endpoints

Job creation:

- `POST /ingest`
- `POST /ingest/batch`
- `GET /ingest/{job_id}`

Remote worker control:

- `POST /jobs/claim`
- `POST /jobs/{job_id}/heartbeat`
- `POST /jobs/{job_id}/stage/{stage_name}`
- `POST /jobs/{job_id}/complete`
- `POST /jobs/{job_id}/fail`

## Shared prerequisites

- `uv`
- `ffmpeg`
- PostgreSQL + pgvector
- KITS available at `KANOMORI_KITS_DIR`

Worker install targets:

```bash
uv sync --group worker-cpu
uv sync --group worker-cuda
```

Use the full worker groups instead of selecting only a leaf OCR group.

## Coordinator configuration

Common variables:

```bash
KANOMORI_DATABASE_URL=postgresql://kanomori:kanomori@localhost:5433/kanomori
KANOMORI_MEDIA_ROOT=./media
KANOMORI_COORDINATOR_TOKEN=replace-this-shared-secret
```

Notes:

- `KANOMORI_COORDINATOR_TOKEN` protects `/jobs/*`
- If it is unset, `/jobs/*` fails closed
- The same token must be set on every worker

## Worker configuration

Typical local-worker variables:

```bash
KANOMORI_COORDINATOR_URL=http://localhost:8000
KANOMORI_COORDINATOR_TOKEN=replace-this-shared-secret
KANOMORI_MEDIA_SOURCE=local
KANOMORI_MEDIA_SOURCE_ROOT=./samples
KANOMORI_KITS_DIR=/path/to/KITS
KANOMORI_STAGE_PARSE_TRANSCRIPT_DEVICE=cpu
KANOMORI_STAGE_OCR_DEVICE=cpu
KANOMORI_STAGE_CLASSIFY_DEVICE=cpu
KANOMORI_STAGE_IMAGE_EMBED_DEVICE=cpu
```

WebDAV-backed source store:

```bash
KANOMORI_MEDIA_SOURCE=webdav
KANOMORI_MEDIA_SOURCE_URL=https://dav.example.com/store
KANOMORI_MEDIA_SOURCE_USER=...
KANOMORI_MEDIA_SOURCE_PASSWORD=...
```

Stage-device notes:

- One worker claims a whole job and runs every stage locally
- `cpu` forces CPU execution for the stage
- `gpu` requires GPU execution for that stage and fails fast if unavailable
- `KANOMORI_STAGE_OCR_DEVICE=gpu` requires `KANOMORI_INGEST_OCR_BACKEND=cuda` or `tensorrt`

## Source store layout

The distributed path expects the same layout described in [../samples/README.md](../samples/README.md):

```text
<root>/
  manifest.jsonl
  <title>_<date>/
    video.mp4
```

Example manifest line:

```json
{"path":"鹿乃的2月18日歌回直播_2024-02-18/video.mp4","title":"鹿乃的2月18日歌回直播","streamed_at":"2024-02-18","source_platform":"bilibili","source_url":"https://...","separate":true}
```

`path` is the worker's canonical source key.

## Bring-up sequence

Start the coordinator side:

```bash
docker compose up -d
uv run kanomori-migrate
uv run uvicorn kanomori.api.app:app --host 0.0.0.0 --port 8000
```

Start a worker:

```bash
uv run kanomori-worker --worker-id gpu-worker-01
```

One-shot cycle:

```bash
uv run kanomori-worker --once --worker-id gpu-worker-01
```

Dry-run without coordinator:

```bash
uv run kanomori-worker --compute-only --source local --manifest-index 0
```

## Enqueueing work

Single job:

```bash
curl -X POST http://localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"media_path":"2024_talk_cut/video.mp4","source_url":"https://example.invalid/video"}'
```

Manifest batch:

```bash
curl -X POST http://localhost:8000/ingest/batch \
  -H 'Content-Type: application/json' \
  -d '{"manifest_path":"manifest.jsonl"}'
```

## Monitoring and recovery

Watch worker logs for claim, stage start, stage completion, heartbeat extension, and failure
messages. If a worker dies after partial progress, the coordinator can re-lease the job and later
workers continue from the last persisted stage instead of restarting the entire pipeline.

## WSL CUDA OCR note

On WSL GPU workers, if ONNX Runtime only reports `AzureExecutionProvider` and
`CPUExecutionProvider`, the environment is usually inconsistent rather than the application code
being wrong. The documented recovery path is:

```bash
uv sync --group worker-cuda
uv pip uninstall onnxruntime
uv pip install --reinstall onnxruntime-gpu==1.27.0
```

Then ensure `LD_LIBRARY_PATH` includes both `/usr/lib/wsl/lib` and the NVIDIA runtime libraries
inside the virtual environment before launching the worker.

## Related code

- `src/kanomori/api/app.py`
- `src/kanomori/api/jobs.py`
- `src/kanomori/ingest/worker.py`
- `src/kanomori/ingest/coordinator_client.py`
- `src/kanomori/ingest/lease.py`
- `src/kanomori/media_source.py`
- `src/kanomori/config.py`
