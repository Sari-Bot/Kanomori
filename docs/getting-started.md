# Getting Started

[简体中文](getting-started.zh-CN.md) | English

This guide gets a local Kanomori instance to the point where you can enqueue an ingest job and
run a search query against it.

## Prerequisites

- `uv`
- Docker + Docker Compose
- `ffmpeg`
- Python 3.12 through `uv`
- A sibling KITS checkout for real transcription

## Install dependency groups

Choose the smallest dependency set that matches your task:

```bash
uv sync
uv sync --group ingest
uv sync --group embed
uv sync --group worker-cpu
uv sync --group worker-cuda
```

Guidance:

- `uv sync` is enough for core API work and most pure-logic tests
- `ingest` adds frame extraction, OCR, and JP tokenization support
- `embed` adds text/image/audio model dependencies
- `worker-cpu` is the full CPU worker target
- `worker-cuda` is the full CUDA worker target

## Configure the environment

```bash
cp .env.example .env
```

Important variables:

- `KANOMORI_DATABASE_URL`: PostgreSQL + pgvector DSN
- `KANOMORI_KITS_DIR`: sibling KITS checkout
- `KANOMORI_MEDIA_ROOT`: derived artifacts output directory
- `KANOMORI_MEDIA_SOURCE_ROOT`: source-store mirror, `./samples` in local development
- `KANOMORI_AUDIO_ASR_MODEL`: required for `POST /search/audio`; set it to the
  kotoba-whisper model version that produced the indexed corpus transcripts

For distributed workers, also set:

- `KANOMORI_COORDINATOR_TOKEN`
- `KANOMORI_COORDINATOR_URL`

## Start the database

```bash
docker compose up -d
uv run kanomori-migrate
```

The default Compose stack exposes PostgreSQL on `localhost:5433`.

## Run the API

```bash
uv run uvicorn kanomori.api.app:app --reload
```

Useful routes:

- `POST /search/transcript`
- `POST /search/screenshot`
- `POST /search/audio` (requires `KANOMORI_AUDIO_ASR_MODEL`)
- `POST /ingest`
- `POST /ingest/batch`
- `GET /ingest/{job_id}`
- `/` for the small Jinja2 + htmx demo UI

## Run a worker

Local continuous worker:

```bash
uv run kanomori-worker
```

One-shot pass:

```bash
uv run kanomori-worker --once --worker-id local-smoke
```

Compute-only dry run:

```bash
uv run kanomori-worker --compute-only --source local --manifest-index 0
```

## First ingest cycle

Single item:

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

Poll job status:

```bash
curl http://localhost:8000/ingest/<job_id>
```

## First search query

Transcript search:

```bash
curl -X POST http://localhost:8000/search/transcript \
  -H 'Content-Type: application/json' \
  -d '{"query":"歌枠","k":5}'
```

Screenshot search is multipart and is usually easiest to exercise through the demo UI or a small
HTTP client script.

Audio search is also multipart. It transcribes the uploaded query clip with kotoba-whisper, then
searches the existing transcript index. Before using it, install the `embed` dependency group and
set `KANOMORI_AUDIO_ASR_MODEL` in `.env` to the same kotoba-whisper model version used when the
corpus was transcribed.

```bash
curl -X POST http://localhost:8000/search/audio \
  -F file=@clip.wav \
  -F k=5
```

## Where to go next

- [Architecture](ARCHITECTURE.md)
- [Distributed Ingestion](distributed-ingestion.md)
- [CUDA Worker Docker Deployment](cuda-worker-docker.md)
- [Sample Corpus Layout](../samples/README.md)
