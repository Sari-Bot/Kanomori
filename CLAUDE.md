# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Current theme: **Scale & Evaluate.** The MVP is built and past its initial
milestones — the full offline ingestion pipeline, transcript search, screenshot
search, and scene-aware reranking all exist and run (see "MVP scope" for the
done/queued breakdown). The work now is *not* the next modality; it is (1)
ingesting a real corpus at scale across multiple machines, and (2) standing up
the evaluation loop so retrieval quality is measured before more modalities are
built. Building audio/karaoke/clip search on a tiny corpus with no eval loop is
premature — the white paper names overengineering-before-measurement as the
primary risk.

Treat the white paper as the source of truth for product *scope and intent*. For
*as-built* behavior, the code is authoritative — where the two disagree, the code
wins and the white paper is the older plan. This file summarizes the decisions a
contributor needs up front; see "Commands" below for how to actually run things.

## What Kanomori is

A **multimodal, moment-level retrieval system** for VTuber livestream
archives (initial dataset: 鹿乃 / Kano Mahoro). The goal is to recover the
exact source stream **and timestamp** from incomplete inputs — a screenshot, a
transcript fragment, lyrics, an audio snippet, an edited clip, or a vague
natural-language memory. It is explicitly *not* a title/tag video search engine;
every result must answer "which stream, which timestamp, what was said, what was
on screen, and why is this match likely."

## Architecture that spans multiple components

The system is two loosely-coupled halves — keep them separate:

- **Offline ingestion (GPU-tolerant, batch):** register video metadata →
  extract audio → extract frames (ffmpeg) → ASR transcript (faster-whisper) →
  OCR (PaddleOCR/EasyOCR) → perceptual hashing (pHash/dHash) → scene
  classification (CLIP, *routing only*) → audio fingerprints → build indexes.
- **Online query (CPU, low-latency):** text search, vector lookup, hash lookup,
  metadata filtering, and reranking over a small candidate set. GPU is only
  touched when a freshly uploaded image/audio snippet must be processed at query
  time. Do not let online query paths depend on heavy offline compute.

Retrieval is **per-modality candidate generation → merge by (video, timestamp)
→ weighted rerank**. Each input type (transcript, screenshot, audio,
multimodal) has its own candidate pipeline (white paper §7.6) that feeds a
shared merge/rerank stage.

### Two design constraints that are easy to get wrong

1. **CLIP is for scene *routing* only** (singing / chatting / gaming / waiting /
   superchat / announcement / collaboration), never the precision retrieval
   engine. Visually static scenes (chatting streams) must be retrieved
   transcript-first, with OCR and metadata; leaning on visual similarity there
   is a known failure mode.
2. **Ranking is scene-aware, not a fixed formula.** Weight profiles differ by
   stream type (white paper §8): chatting weights transcript highest, gaming
   weights visual similarity highest, karaoke weights audio match highest.
   Implement ranking as per-scene weight profiles, not one global score.

## Core data entities

`videos`, `frames`, `transcript_segments`, `ocr_segments`,
`audio_fingerprints`, `scene_segments`, `search_results`, `user_feedback`.
Transcripts are segmented into 10–30s windows storing **both** original and
normalized text, with full-text (BM25) and semantic-embedding indexes. Japanese
text normalization is a first-class concern (ASR misrecognizes names, songs, and
game terms — fuzzy + semantic search and stored confidence scores mitigate this).

## Recommended stack (from white paper §12)

Backend FastAPI · PostgreSQL · ffmpeg · faster-whisper (ASR) ·
PaddleOCR/EasyOCR · imagehash · FAISS (vectors) · Meilisearch *or* Postgres
full-text · Vue/React/Next frontend · local filesystem for media (MinIO later).
Later upgrades: Qdrant, Elasticsearch/OpenSearch, Celery/RQ, Docker Compose.
A single CPU machine (+ optional consumer GPU for offline) is the intended MVP
deployment — avoid premature infra.

## Roadmap status — what's done, what's queued

Done (Phase 1–2 + scene-aware rerank): video import, frame extraction, transcript
generation (via KITS subprocess), OCR, scene classification, transcript search,
screenshot search, scene-aware merge/rerank, and timestamp-level results with
nearby transcript + preview frames. The 8-stage resumable ingestion pipeline and
the FastAPI query API + htmx UI are implemented.

In flight (current **Scale & Evaluate** theme): distributed multi-machine
ingestion of a real corpus, and populating + running the eval suites (`eval/`
currently holds only OCR cases; the white-paper §13 query sets are not built yet).

Queued behind Scale & Evaluate — don't pull forward without a reason: audio
snippet search, karaoke + full clip reverse search, vague-memory + evidence-based
AI Q&A, then custom-trained visual models, real-time indexing, knowledge graph,
recommendations, public deployment. Roadmap order: audio snippet → karaoke + clip
reverse search → vague-memory + evidence-based AI Q&A.

## Evaluation

Top-5 accuracy is the headline metric (users visually verify candidates), valued
above Top-1. Also track MRR, timestamp error range, search latency, indexing
time per video-hour, OCR hit rate, and user correction rate. The white paper
(§13) specifies a concrete eval set: 100 screenshots, 100 transcript queries,
50 audio snippets, 50 vague-memory queries.

## Domain / handling constraints

Archive content has copyright and platform-terms exposure. Store metadata and
derived indexes rather than redistributing full video, surface **source links**
instead of hosting, keep previews short, support private/local deployment, and
respect takedown requests. Keep this in mind when designing storage and any
user-facing media delivery.

## Commands

Dependency groups are opt-in to keep core setup torch-free (`tool.uv`
`default-groups = ["dev"]`):

```bash
uv sync                       # core + dev (fast; no torch) — enough for pure-logic tests
uv sync --group ingest        # frame/OCR/JP-tokenization deps (offline host)
uv sync --group embed         # embedding + scene models (pulls CPU torch)
uv sync --group ocr-cuda      # optional ONNX Runtime CUDA OCR (NVIDIA Linux x86_64 only)
```

Database + migrations (Postgres+pgvector runs in Docker on host port 5433):

```bash
cp .env.example .env          # adjust KANOMORI_DATABASE_URL / KITS_DIR / MEDIA_ROOT
docker compose up -d
uv run kanomori-migrate       # apply migrations/*.sql (forward-only)
```

Run the services:

```bash
uv run uvicorn kanomori.api.app:app --reload   # query + ingest API and htmx UI
uv run kanomori-worker                         # ingestion worker: polls jobs, runs the pipeline
```

### Distributed ingestion (multi-machine)

Workers never touch Postgres — they claim jobs over the coordinator's
authenticated `/jobs/*` HTTP API, pull source video from a `MediaSource`
(`local` dir or `webdav`), run the pipeline, and push each stage's result +
artifacts back. Set a shared `KANOMORI_COORDINATOR_TOKEN` on the coordinator and
every worker (unset ⇒ `/jobs` refuses all calls with 503); point remote workers
at `KANOMORI_COORDINATOR_URL`.

```bash
uv run kanomori-migrate                        # 0003_lease.sql adds the lease/heartbeat columns
# Enqueue a whole manifest (dedup by source path; re-runs skip already-queued):
curl -X POST localhost:8000/ingest/batch -d '{"manifest_path":"manifest.jsonl"}'

uv run kanomori-worker                         # distributed loop: claim → run → push, with heartbeat
uv run kanomori-worker --once                  # one claim+run cycle, then exit
uv run kanomori-worker --worker-id gpu-box-2   # override the worker id (default: host+pid)
uv run kanomori-worker --compute-only          # NO coordinator/DB: run the pipeline over a manifest
                                               #   sample and assert every StageResult round-trips
uv run kanomori-worker --compute-only --media-path 2024_talk_cut/video.mp4   # pick the sample
```

A job is leased with a fencing `lease_epoch`; every mutation is gated on it, so a
slow worker whose lease lapsed is rejected (409) once another worker reclaims the
job. Heartbeats extend the lease; an expired lease makes the job claimable again
without redoing stages already marked done (expensive KITS transcribe + scene
detect are checkpointed as artifacts). See
`docs/plans/2026-06-24-distributed-ingestion-design.md`.

Tests + lint:

```bash
uv run pytest                                  # all tests
uv run pytest tests/unit/test_kits_client.py   # one file
uv run pytest tests/unit/test_kits_client.py::test_name   # one test
uv run pytest -m "not requires_db"             # skip DB-backed integration tests
uv run ruff check .
```

Test markers (`pyproject.toml`): `requires_db` needs live Postgres+pgvector;
`requires_models` needs ML models (BGE-M3/DINOv2/SigLIP) loaded. Plain `uv run
pytest` runs everything; deselect with `-m "not requires_db"` for a lightweight run.

Ingest flow: `POST /ingest` with a `media_path` (single job), or `POST
/ingest/batch` with a `manifest_path` (enqueue a whole `manifest.jsonl`). Poll
`GET /ingest/{job_id}` until `complete`, then `POST /search/transcript` or
`/search/screenshot`.

## A note on stack drift

The white paper §12 names a *candidate* stack; the as-built choices differ in
places. Trust the code: ASR is **KITS** (a kotoba-whisper pipeline) invoked as a
**subprocess** (never imported — it is AGPL with a GPU/torch stack kept isolated),
not faster-whisper in-process; scene classification is **SigLIP** and image
embeddings are **DINOv2** (not CLIP); vectors live in **pgvector** (HNSW), not
FAISS; lexical search is **Postgres tsvector** (JP-tokenized with fugashi), not
Meilisearch. One datastore (Postgres) holds vectors + lexical + metadata.
