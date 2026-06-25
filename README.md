# Kanomori

[简体中文](README.zh-CN.md) | English

<p align="center">
  <img src="imgs/logo.png" alt="Kanomori logo" width="360">
</p>

Kanomori is a multimodal, moment-level retrieval system for VTuber livestream archives.
Given a transcript fragment, screenshot, lyric line, or vague memory, it aims to recover the
exact source stream and timestamp.

The current codebase is a Phase-1 MVP focused on real ingestion and retrieval infrastructure:
distributed offline ingestion, transcript search, screenshot search, and scene-aware reranking.

## What is implemented

- `POST /search/transcript` for transcript-first retrieval over indexed segments
- `POST /search/screenshot` for screenshot retrieval with OCR + image embeddings
- `POST /ingest` and `POST /ingest/batch` for enqueueing offline ingestion jobs
- `GET /ingest/{job_id}` for job status
- `POST /jobs/*` coordinator endpoints for remote workers
- A small server-rendered demo UI built with Jinja2 + htmx

## System shape

Kanomori is split into two loosely coupled halves:

- Offline ingestion: register media, extract audio and frames, transcribe through KITS,
  run OCR, classify scenes, build embeddings, and persist derived artifacts
- Online query: run low-latency transcript and screenshot retrieval against PostgreSQL
  indexes and rerank candidates into timestamped hits

The query path is CPU-oriented. Heavy model work belongs to the offline worker path.

## Quickstart

Prerequisites:

- `uv`
- Docker + Docker Compose
- `ffmpeg`
- A sibling [KITS](https://github.com/kanbereina/KITS) checkout for real transcription

Install only what you need:

```bash
uv sync
uv sync --group ingest
uv sync --group embed
uv sync --group worker-cpu
uv sync --group worker-cuda
cp .env.example .env
docker compose up -d
uv run kanomori-migrate
```

Run the main services:

```bash
uv run uvicorn kanomori.api.app:app --reload
uv run kanomori-worker
```

Then:

1. `POST /ingest` or `POST /ingest/batch`
2. Poll `GET /ingest/{job_id}`
3. Query with `POST /search/transcript` or `POST /search/screenshot`

For a step-by-step setup guide, see [docs/getting-started.md](docs/getting-started.md).

## Documentation

- [Getting Started](docs/getting-started.md)
- [Docs Index](docs/README.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Distributed Ingestion](docs/distributed-ingestion.md)
- [CUDA Worker Docker Deployment](docs/cuda-worker-docker.md)
- [Sample Corpus Layout](samples/README.md)

## Current status

Implemented now:

- Transcript search
- Screenshot search
- Resumable ingestion pipeline
- Distributed coordinator/worker ingestion
- OCR backend selection and worker stage device pinning

Planned later:

- Audio snippet search
- Karaoke and edited clip reverse search
- Vague-memory and evidence-based QA workflows

## Repository layout

```text
src/kanomori/            application code
docs/                    maintainer and operator docs
samples/                 local source-store mirror and manifest examples
migrations/              forward-only SQL migrations
tests/                   unit and integration tests
imgs/                    project assets
```

## Related files

- `kanomori_project_white_paper.md`: original product vision
- `HANDOFF_TO_GPT.md`: internal continuation notes for ongoing development sessions

Public docs in this README set describe the code as currently implemented.

## Acknowledgements

Kanomori began as a personal project inspired by the livestream archives and creative works of Kano (鹿乃).

The visual direction of the project is inspired by the picture book 「こまったましろ」, written by 鹿乃 and illustrated by 水玉子.

All rights to the original characters, illustrations, and related works belong to their respective creators and publishers.

Kanomori is an independent fan-created project and is not affiliated with or endorsed by the original creators.

## License

Kanomori's source code is licensed under MIT.

KITS is a separate AGPL-3.0 project and is used only as an external subprocess. Archive content
and source streams remain subject to the rights of their original owners.
