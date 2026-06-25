# CUDA Worker Docker Deployment

This is a worker-only deployment scheme. The FastAPI coordinator and PostgreSQL stay on the
coordinator host; the CUDA container only claims jobs from `/jobs/*`, reads source media, runs
`kanomori-worker`, and pushes stage results/artifacts back.

## Why This Shape

- The worker never talks to PostgreSQL in distributed mode.
- `KANOMORI_MEDIA_ROOT` is still writable inside the worker because source downloads, frames, SRTs,
  and temporary artifacts are produced locally before upload.
- KITS stays a sibling checkout and is invoked through `uv run kits subtitle`; the container mounts
  the checkout and keeps a Linux `.venv` in a named Docker volume.
- Use aggregate uv groups for worker installs. Running `uv sync --group ocr-cuda` alone is an exact
  sync of only that selected group and can remove `ingest`/`embed` dependencies from the venv.

## Dependency Targets

Use these exact sync targets:

```bash
uv sync                         # core + dev only
uv sync --group ingest          # CPU OCR ingestion support
uv sync --group worker-cpu      # full CPU worker: ingest + embed
uv sync --group worker-cuda     # full CUDA worker: ingest-base + CUDA OCR + embed
uv sync --group worker-trt      # full TensorRT worker: worker-cuda + TensorRT bindings
```

`worker-cuda` intentionally does not include the CPU `onnxruntime` wheel. It uses
`onnxruntime-gpu`, which provides the importable `onnxruntime` module and exposes
`CUDAExecutionProvider`.

If you must add one leaf group to an already-hand-tuned venv without pruning anything, use
`uv sync --inexact --group <group>`. For reproducible deploys, prefer the aggregate groups above.

## Coordinator Host

Start the coordinator the same way as the existing distributed ingestion runbook:

```bash
docker compose up -d
uv run kanomori-migrate
KANOMORI_COORDINATOR_TOKEN=replace-this-shared-secret \
uv run uvicorn kanomori.api.app:app --host 0.0.0.0 --port 8000
```

Enqueue work with the existing batch endpoint:

```bash
curl -X POST http://coordinator-host:8000/ingest/batch \
  -H 'Content-Type: application/json' \
  -d '{"manifest_path":"manifest.jsonl"}'
```

## CUDA Worker Host

Install the NVIDIA driver, Docker, Docker Compose, and NVIDIA Container Toolkit. Confirm Docker can
see the GPU before using Kanomori:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04 nvidia-smi
```

Create the worker env file:

```bash
cp .env.cuda-worker.example .env.cuda-worker
```

Edit at least:

```bash
KANOMORI_COORDINATOR_URL=http://coordinator-host:8000
KANOMORI_COORDINATOR_TOKEN=replace-this-shared-secret
KANOMORI_KITS_DIR_HOST=/absolute/path/to/KITS
KANOMORI_MEDIA_SOURCE_ROOT_HOST=/absolute/path/to/source-store
```

Build and prewarm the worker image:

```bash
docker compose -f docker-compose.cuda-worker.yml --env-file .env.cuda-worker build
docker compose -f docker-compose.cuda-worker.yml --env-file .env.cuda-worker run --rm \
  cuda-worker sh -lc 'cd /opt/kits && uv sync'
```

Start continuous polling:

```bash
docker compose -f docker-compose.cuda-worker.yml --env-file .env.cuda-worker up -d
```

Watch logs:

```bash
docker compose -f docker-compose.cuda-worker.yml --env-file .env.cuda-worker logs -f cuda-worker
```

## Smoke Checks

Check CUDA provider discovery inside the built worker image:

```bash
docker compose -f docker-compose.cuda-worker.yml --env-file .env.cuda-worker run --rm \
  cuda-worker uv run --no-sync python - <<'PY'
from kanomori.ocr import OcrBackendUnavailable
from kanomori.ocr_tensorrt import ensure_cuda_ep_available
import onnxruntime as ort

ensure_cuda_ep_available(OcrBackendUnavailable)
print(ort.get_available_providers())
PY
```

Run one compute-only sample without touching the coordinator:

```bash
docker compose -f docker-compose.cuda-worker.yml --env-file .env.cuda-worker run --rm \
  cuda-worker uv run --no-sync kanomori-worker --compute-only --source local --manifest-index 0
```

Run one distributed claim cycle:

```bash
docker compose -f docker-compose.cuda-worker.yml --env-file .env.cuda-worker run --rm \
  cuda-worker uv run --no-sync kanomori-worker --once --worker-id cuda-worker-smoke
```

## Notes

- `KANOMORI_STAGE_*_DEVICE=gpu` is fail-fast. If CUDA is not visible, the stage fails instead of
  silently dropping to CPU.
- `KANOMORI_INGEST_OCR_BACKEND=cuda` is required when `KANOMORI_STAGE_OCR_DEVICE=gpu`.
- For WebDAV source media, set `KANOMORI_MEDIA_SOURCE=webdav` plus URL/user/password variables;
  the `/data/source` mount is unused in that mode.
- Model caches live in Docker volumes so rebuilds do not redownload every model.
