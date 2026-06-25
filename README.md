# Kanomori

Multimodal, **moment-level** retrieval over VTuber (鹿乃 / Kano Mahoro) livestream archives.
Given a screenshot, a transcript fragment, lyrics, or a vague memory, Kanomori recovers the
exact **source stream and timestamp** — not just a title or tag match.

> Even if you only remember a line, a screenshot, a song, or a vague moment, Kanomori helps
> recover the original livestream and timestamp.

This is a Phase-1 MVP under active construction. See `HANDOFF.md` for current build status,
`docs/ARCHITECTURE.md` for decisions and rationale, and `kanomori_project_white_paper.md` for
the full product vision.

Distributed ingestion usage for the current coordinator/worker system lives in
`docs/distributed-ingestion.md`.

## How it works

Two loosely-coupled halves:

- **Offline ingestion (GPU-tolerant, batch):** media → audio → transcript → frames → OCR →
  scene classification → image embeddings → indexes. Resumable and idempotent.
- **Online query (CPU, low-latency):** lexical + vector + metadata search with scene-aware
  reranking over a small candidate set.

ASR is delegated to the sibling **[KITS](https://github.com/kanbereina/KITS)** project (a
Kano-tuned kotoba-whisper pipeline), invoked as a **subprocess** — never imported — so its
GPU/torch stack stays isolated from Kanomori's CPU query path. Kanomori does storage and
retrieval; KITS produces transcripts.

## Stack

FastAPI · PostgreSQL + pgvector (one datastore: HNSW vectors + tsvector full-text + metadata)
· BGE-M3 embeddings · DINOv2 + SigLIP (image) · PySceneDetect · fugashi (Japanese
tokenization) · uv · Jinja2 + htmx UI. Single-machine deployment; GPU only for offline
ingestion.

## Setup

Requires [uv](https://docs.astral.sh/uv/), Docker (+ Compose), and ffmpeg. A sibling KITS
checkout (`uv sync`-ed in its own venv) is needed only to run real transcription; tests mock it.

```bash
uv sync                       # core + dev deps (fast; no torch)
uv sync --group ingest        # + frame/OCR/tokenization deps (offline host)
uv sync --group embed         # + embedding/scene models (pulls CPU torch)
uv sync --group ocr-eval      # optional OCR bake-off engines (not production default)
uv sync --group ocr-cuda      # optional ONNX Runtime CUDA OCR on NVIDIA Linux
uv sync --group ocr-trt       # optional CUDA bindings for NVIDIA TensorRT OCR

cp .env.example .env          # adjust DATABASE_URL / KITS_DIR / MEDIA_ROOT if needed
docker compose up -d          # PostgreSQL + pgvector on localhost:5433
uv run kanomori-migrate       # apply migrations/*.sql
```

## Run

```bash
uv run pytest                                  # unit + mocked-integration tests
uv run ruff check .                            # lint
uv run uvicorn kanomori.api.app:app --reload   # API (available from Step 1)
uv run kanomori-worker                         # ingestion worker loop (available from Step 1)
```

Then `POST /ingest` with a local clip's `media_path`, poll `GET /ingest/{job_id}` until
`complete`, and `POST /search/transcript` with a phrase to get ranked stream + timestamp hits.

## OCR refinement

OCR is configured as `model + backend` so the ingestion and screenshot-query paths can use the
same `OcrReader -> list[OcrResult]` contract while swapping inference backends underneath.
Defaults are:

```bash
KANOMORI_INGEST_OCR_MODEL=ppocrv5_server
KANOMORI_INGEST_OCR_BACKEND=onnxruntime
KANOMORI_QUERY_OCR_MODEL=ppocrv5_server
KANOMORI_QUERY_OCR_BACKEND=onnxruntime
```

Offline workers also support per-stage CPU/GPU pinning for the stages that can reasonably run on
either device:

```bash
KANOMORI_STAGE_PARSE_TRANSCRIPT_DEVICE=cpu
KANOMORI_STAGE_OCR_DEVICE=cpu
KANOMORI_STAGE_CLASSIFY_DEVICE=cpu
KANOMORI_STAGE_IMAGE_EMBED_DEVICE=cpu
```

These are worker-local settings, not per-job routing. `cpu` forces CPU execution for that stage.
`gpu` means the worker must initialize that stage on the process-visible default GPU; if CUDA or
the required GPU backend is unavailable, the stage fails fast instead of silently falling back.

Supported models are `legacy_rapidocr`, `ppocrv5_mobile`, and `ppocrv5_server`. Supported
backends are `onnxruntime`, `cuda`, and `tensorrt`; `legacy_rapidocr` supports only
`onnxruntime`.

CPU OCR setup:

```bash
uv sync --group ingest --group ocr-eval
```

CUDA OCR setup uses ONNX Runtime's CUDA Execution Provider. Use this when TensorRT recall is
poor but GPU latency is still needed:

```bash
uv sync --group ingest --group ocr-cuda
```

The CUDA backend preloads NVIDIA wheel libraries from `site-packages/nvidia/*/lib` before
importing ONNX Runtime, so a normal uv-managed venv is enough on most NVIDIA Linux hosts. On WSL,
if the driver library is not discoverable, also expose the WSL driver path before running OCR:

```bash
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Verify provider discovery:

```bash
uv run python - <<'PY'
from kanomori.ocr_tensorrt import ensure_cuda_ep_available
from kanomori.ocr import OcrBackendUnavailable

ensure_cuda_ep_available(OcrBackendUnavailable)
print("CUDAExecutionProvider available")
PY
```

TensorRT remains available for speed experiments and requires an NVIDIA Linux host, the
`ocr-trt` dependency group, and the TensorRT Python package from NVIDIA's package index, for
example:

```bash
uv sync --group ocr-trt
uv pip install --extra-index-url https://pypi.nvidia.com/ tensorrt
```

Runtime OCR falls back from unavailable CUDA or TensorRT to ONNX Runtime when fallback is
allowed, but benchmark runs fail instead so backend numbers stay honest.

Ingest OCR is slightly stricter when `KANOMORI_STAGE_OCR_DEVICE=gpu`: the worker disables silent
fallback and requires `KANOMORI_INGEST_OCR_BACKEND` to be `cuda` or `tensorrt`. Query-time OCR
keeps the previous fallback behavior and remains independent of the worker stage-device settings.

To benchmark PP-OCRv5 candidates:

```bash
uv sync --group ingest --group ocr-eval --group ocr-cuda
uv run kanomori-ocr-benchmark \
  --cases eval/ocr/kanomori_frames.jsonl \
  --models ppocrv5_server \
  --backends onnxruntime,cuda,tensorrt
```

The deprecated `--engines legacy_rapidocr,rapidocr_ppocrv5_mobile` form still works for old
ONNX-only bake-offs.

Benchmark cases are JSONL records with an image path and expected visible terms:

```json
{"id":"frame-001","image":"media/.../frame.jpg","expected_terms":["鹿乃","歌枠"]}
```

`image` may be a glob such as `media/*/frames/frame_000000_000.jpg`, but it must match
exactly one file. `expected_terms` should be phrase-level visible substrings: split long
sentences into stable chunks that matter for retrieval, while avoiding single-character terms.

## Layout

```
src/kanomori/
  config.py db.py models.py migrate.py
  srt.py fusion.py scene.py text.py kits_client.py   # pure logic + the KITS seam
  ingest/    embed/    retrieval/    api/    web/
migrations/   # forward-only plain SQL
tests/        # unit/ integration/ fixtures/
samples/      # manual ingest inputs (gitignored; README tracked)
```

## License

MIT (Kanomori's own code). KITS is AGPL-3.0 and is used only as an external subprocess.
Archive content is copyright-sensitive: Kanomori stores derived indexes, source links, and
short preview thumbnails — it does not host source video.
