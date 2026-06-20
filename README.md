# Kanomori

Multimodal, **moment-level** retrieval over VTuber (鹿乃 / Kano Mahoro) livestream archives.
Given a screenshot, a transcript fragment, lyrics, or a vague memory, Kanomori recovers the
exact **source stream and timestamp** — not just a title or tag match.

> Even if you only remember a line, a screenshot, a song, or a vague moment, Kanomori helps
> recover the original livestream and timestamp.

This is a Phase-1 MVP under active construction. See `HANDOFF.md` for current build status,
`docs/ARCHITECTURE.md` for decisions and rationale, and `kanomori_project_white_paper.md` for
the full product vision.

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
