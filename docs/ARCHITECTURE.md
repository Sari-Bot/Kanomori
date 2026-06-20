# Kanomori Architecture & Decisions

Engineering decisions and rationale for the Phase-1 MVP. The product vision and full scope
live in `kanomori_project_white_paper.md` (kept intact as the original design); this document
records **how we are building it** and **where we deviate from the white paper and why**.
Sequencing lives in the approved plan; current build status lives in `HANDOFF.md`.

## What Kanomori is

Multimodal, moment-level retrieval over VTuber (鹿乃 / Kano Mahoro) livestream archives:
given a screenshot, a transcript fragment, lyrics, or a vague memory, return the exact source
stream **and timestamp** — not a title/tag match. Every result should answer: which stream,
which timestamp, what was said, what was on screen, and why the match is likely.

## System shape: two loosely-coupled halves

- **Offline ingestion (GPU-tolerant, batch):** register media → locate/extract audio →
  transcript (via KITS subprocess) → frames → OCR → scene classification → image
  embeddings/pHash → build indexes. Resumable, idempotent, runs on a GPU host.
- **Online query (CPU, low-latency):** lexical + vector + metadata search and scene-aware
  rerank over a small candidate set. **No GPU on the query path** except when a freshly
  uploaded screenshot must be embedded at query time (lazy-loaded, isolated).

These never share a process. Ingestion writes derived indexes to PostgreSQL; the API reads
them.

## KITS integration (the ASR front-end)

KITS (`/Users/lb/Documents/Code/KITS`, v1.5.2, **AGPL-3.0-or-later**) already implements
Kano-tuned ASR: kotoba-whisper-v2.2 (JP-distilled), silence-based long-audio segmentation,
punctuation restoration, hallucination suppression, vocal separation, SRT output.

**Decision: consume KITS strictly as a subprocess (`uv run kits subtitle ...`), never import
it.** Rationale, in priority order:

1. **Dependency isolation.** KITS pins a CUDA `torch` (pytorch-cu128 index) plus
   `transformers`/`audio-separator`/`onnxruntime-gpu`/`punctuators`, with non-trivial
   resolver gymnastics. Merging that into Kanomori's CPU-only query stack invites conflicts.
   Keeping KITS in its own uv venv sidesteps all of it.
2. **GPU/CPU split forces a process boundary anyway.** KITS transcription hard-requires
   CUDA/MPS (`KITS/src/kits/transcriber.py:86`); Kanomori's query layer is CPU. Transcription
   already lives in a separate offline worker, so "library vs subprocess" is really just
   "in-process call vs `subprocess.run` inside that worker" — and subprocess wins on isolation.
3. **Stable, inspectable artifacts.** SRT files are cacheable, resumable, human-readable
   ingestion artifacts — better suited to batch offline work than passing live Python objects.
4. **License containment.** Subprocess "mere aggregation" keeps AGPL obligations contained to
   KITS, leaving Kanomori free to choose its own license. (Secondary to the engineering
   reasons above, but a real benefit.)

**Boundary contract** (`kanomori.kits_client`):

- Invoke `uv run kits subtitle -i <audio> -o <out.srt>` with `cwd=KITS_DIR`, **argv as a list**
  (no shell string → no injection from filenames). Capture stdout/stderr to a log artifact,
  check the return code, assert the SRT is non-empty; raise `KitsError` otherwise.
- Flags used: `--separate` (vocal isolation for karaoke), `--filter-game valorant`
  (repeatable), `--language` (default `japanese`). Segmentation/punctuation defaults are good
  — don't override in the MVP.
- **Kanomori has its own SRT parser** (`kanomori.srt`): we cannot `import kits.parse_srt`. It
  mirrors KITS's lenient parser (`KITS/src/kits/subtitle.py:367`) — split on blank lines, find
  the first `HH:MM:SS,mmm --> HH:MM:SS,mmm` line by regex, join remaining lines as text,
  tolerate a missing index — so round-trips agree.
- **Granularity:** KITS emits `Sentence{start, end, text}` (a few seconds each). This *is*
  Kanomori's transcript retrieval unit; no word-level alignment is pursued (kotoba's distilled
  decoder can't do word timestamps anyway, and sentence-level matches search granularity).

## Deviations from the white paper

The white paper's product thinking is the source of truth; several of its *tech* choices are
dated or over-built for an MVP. We deviate as follows:

| White paper | Kanomori MVP | Why |
|---|---|---|
| Postgres + FAISS + Meilisearch + FS (4 stores) | **One Postgres** (pgvector HNSW + tsvector GIN + metadata) | At MVP scale (~36k transcript + ~144k frame vectors; low millions even at multi-thousand-hour) one store does filtered-vector + lexical + metadata in a single query — no cross-store sync, no FAISS↔row id drift. FAISS is a library (no filtering/CRUD/persistence); if we ever outgrow pgvector we jump to Qdrant, skipping FAISS. |
| Fixed linear score weights | **Reciprocal Rank Fusion** + scene-aware profile multipliers | RRF is scale-free — no normalizing disparate score ranges (the part that breaks linear fusion). |
| CLIP as a retrieval engine | **pHash prefilter → DINOv2 (near-dup) + SigLIP (scene routing)** | pHash only catches near-exact dupes (breaks on re-encode/overlay/crop — exactly what clips contain). DINOv2 is stronger for visual instance matching; SigLIP > vanilla CLIP for text-routable scene classification. CLIP is routing-only, never precision. |
| Fixed-interval frame sampling | **Scene-change (PySceneDetect) + sampling in long static spans** | More information per frame, less storage, aligns with how clips are actually cut. |
| Celery/Redis background jobs | **`jobs` table + worker loop** | No broker needed at single-machine MVP scale; a Postgres `SELECT ... FOR UPDATE SKIP LOCKED` loop is enough. |
| One global ranking formula | **Per-scene weight profiles** (white paper §8 itself asks for this) | chatting=transcript-weighted, gaming=visual-weighted, karaoke=lyrics/OCR-weighted. |

What we keep from the white paper: the offline/online split, faster-whisper-family ASR (via
KITS), scene-aware ranking as the core idea, **Top-5 accuracy as the headline metric**, and
the "store indexes + source links, don't redistribute video" copyright posture.

## Data model

`CREATE EXTENSION vector;`. Embedding dims are **pinned in SQL** (authoritative), not derived
from config strings:

- **BGE-M3 dense = `vector(1024)`** (transcript_segments)
- **DINOv2 ViT-B/14 = `vector(768)`** (frames; arrives in migration 0002)

Core tables (0001): `videos` (idempotency key `content_hash` = sha256 of media; stores
`source_url` link + derived `media_path`), `jobs` (per-stage `stage_status jsonb`, one row per
`content_hash`), `transcript_segments` (orig + normalized text, `embedding vector(1024)`,
`tsv`, indexes: HNSW cosine + GIN + `(video_id, start_sec)`). Visual tables (0002, deferred):
`frames`, `ocr_segments`, `scene_segments`. `audio_fingerprints` reserved for Phase 3.

Migrations are forward-only plain SQL applied by `kanomori.migrate` (`uv run kanomori-migrate`)
and tracked in `schema_migrations` — deliberately lighter than Alembic for an MVP.

## Retrieval

Cross-lingual by design (JP content; JP/ZH/EN queries): the query is embedded with BGE-M3
regardless of language, and the lexical path tokenizes the query with the same JP tokenizer
used at ingest.

- **transcript**: lexical (`tsv @@ plainto_tsquery`, `ts_rank`) + dense (`embedding <=> qvec`,
  HNSW cosine) → RRF.
- **screenshot**: pHash prefilter (`bit_count(phash # q) <= t`) + DINOv2 vector search +
  OCR-the-upload → `ocr_segments.tsv` + SigLIP scene-route → RRF.
- **merge**: bucket candidates by `(video_id, round(ts_sec / BUCKET))` (BUCKET ≈ 5–10s); look
  up each bucket's `scene_type`; apply that scene's weight profile; fuse via RRF; optional CPU
  cross-encoder rerank over the merged top-k. Returns `SearchHit{video_id, ts_sec, score,
  scene_type, why}`.

## Open decision points

- **Japanese full-text tokenization.** Stock Postgres can't segment Japanese (no spaces ⇒ one
  token). **Decision:** tokenize application-side with **fugashi (MeCab)** in `kanomori.text`,
  store space-joined tokens, index/query with `to_tsvector('simple', …)` /
  `plainto_tsquery('simple', …)` — symmetric, keeps the stock `pgvector/pgvector:pg16` image.
  Dense BGE-M3 cushions lexical gaps. **Upgrade path:** pgroonga / pg_bigm n-gram FTS (custom
  image) if recall on ASR typos proves insufficient.
- **BGE-M3 CPU query latency** (~560M params). **Mitigation:** preload once in the FastAPI
  lifespan; consider ONNX/int8 or a smaller fallback (bge-small / e5-small) if p95 exceeds
  target. Benchmark during Step 1; the embedding model is swappable behind `text_embedder`.
- **Karaoke `--separate` trigger.** No cheap pre-transcription signal for "is this singing."
  **Decision:** title/metadata keyword heuristic (歌枠 / karaoke / 歌 / cover / setlist) +
  manual `separate` override on the ingest request. SigLIP's "singing" label is only known
  post-classification — too late to cheaply re-transcribe. Accepted limitation.
- **GPU availability for ingestion.** Transcription hard-requires CUDA/MPS; OCR/DINOv2/SigLIP
  tolerate CPU (slower). Query path is CPU-only by design. CI mocks the KITS subprocess.

## Security & copyright posture

- **`/ingest` is unauthenticated in the local MVP** and triggers a GPU subprocess + filesystem
  writes. **Any non-local deployment MUST add authentication before exposing `/ingest`**, and
  should rate-limit `/search/screenshot` (it lazy-loads image models).
- **Store derived indexes + source links + short preview thumbnails only — never host source
  video.** `samples/` (manual inputs) and `media/` (derived artifacts) are both gitignored;
  derived media is wipe-and-rebuild.
