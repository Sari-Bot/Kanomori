# Kanomori — Handoff

Living status doc for the Phase-1 MVP build. Read this first when resuming. The full design
lives in `docs/ARCHITECTURE.md` (decisions + rationale) and the approved plan at
`~/.claude/plans/let-s-make-the-modification-happy-candle.md` (sequencing). This file tracks
**where we are right now** and **what to do next**.

## TL;DR

Multimodal moment-level retrieval over 鹿乃 (Kano) livestream archives: given a screenshot,
a transcript fragment, or lyrics, return the exact stream + timestamp. ASR is delegated to
the sibling **KITS** project (subprocess, not imported). Kanomori does storage + retrieval.

## Testing strategy (DECIDED — Option A: one real clip, mock the rest)

Only **one stage needs a GPU**: KITS transcription (later also SigLIP/DINOv2). Everything
else is CPU logic, tested without video via unit tests + a **mocked KITS subprocess**.

- **One real clip now:** `samples/2024_talk_cut.mp4` (chatting/talking, clear JP speech).
  Used to run the Step-1 end-to-end proof **once** on this Mac's MPS (short clip → fast).
  Confirms the KITS seam for real (catches SRT-format drift, path handling, etc.).
- **Everything else mocked** until a better GPU is available. Integration tests mock
  `kits_client.transcribe` to drop a fixture `.srt`; no GPU needed in CI or local dev.
- **Steps 2–3:** the single chatting clip exercises the frame/OCR/screenshot code paths but
  **does not validate scene-aware ranking quality** — chatting is the transcript-weighted
  (easy) profile. Gaming/karaoke profiles (visual-/lyrics-weighted) stay unverified until
  those clip types are added. Known, accepted gap for the MVP.

## Status by step

- [x] **Step 0 — Scaffolding** (done)
  - Done: git init + initial commit (`aa482fa`); `pyproject.toml` (uv, src layout, core +
    opt-in `ingest`/`embed` groups, ruff E/F/I/UP/B/SIM, pytest markers); `docker-compose.yml`
    (pgvector pg16 on host port **5433**); `.gitignore`; `.env.example`; `config.py`
    (pydantic-settings); `db.py` (psycopg3 pool + pgvector adapters); `migrations/0001_init.sql`
    (videos / jobs / transcript_segments + HNSW + GIN); `migrate.py` (forward-only runner);
    `samples/` + README; `models.py` (cross-layer pydantic contracts); `docs/ARCHITECTURE.md`;
    `README.md`; `tests/test_scaffolding.py` (5 green smoke tests).
  - Verified: `uv sync` (core+dev, no torch) ✓; `uv run ruff check .` clean ✓;
    `uv run pytest` 5 passed ✓; pgvector container + `uv run kanomori-migrate` apply
    `0001_init.sql` ✓ (see below).
- [ ] **Step 1 — Transcript vertical slice** (blocked by Step 0): `srt.py` (+tests), `text.py`
  (JP normalize/tokenize via fugashi), `kits_client.py`, ingest stages register/locate_media/
  transcribe/parse_transcript, `pipeline.py` + `worker.py`, `embed/text_embedder.py` (BGE-M3),
  `retrieval/transcript.py` + `fusion.py` (RRF), transcript search endpoint. **Proof:**
  `samples/2024_talk_cut.mp4` → KITS → segments in PG → `/search/transcript` returns the right
  timestamp.
- [ ] **Step 2 — Frames + OCR + image** (blocked by Step 1)
- [ ] **Step 3 — Scene-aware rerank + merge** (blocked by Step 2)
- [ ] **Step 4 — Minimal UI (Jinja2 + htmx)** (blocked by Step 3)
- [ ] **Step 5 — Eval harness + hardening** (blocked by Step 4)

(Task tracker mirrors these: Step 0 = task #1, Step 1 = #2, Step 2 = #3, Step 3 = #4,
Step 4 = #5, Step 5 = #6.)

## Key invariants (don't break these)

- **Never `import kits`.** KITS is AGPL and GPU-heavy; reach it only via `kanomori.kits_client`
  shelling out to `uv run kits subtitle ...` with `cwd=KITS_DIR`, argv as a **list**.
- **Kanomori needs its own SRT parser** (`srt.py`) mirroring KITS's lenient parser — can't
  import `kits.parse_srt`. Match its leniency so round-trips agree.
- **Online query path is CPU-only.** GPU only for offline ingestion + KITS.
- **Embedding dims are pinned in SQL:** transcript = `vector(1024)` (BGE-M3), frames =
  `vector(768)` (DINOv2 ViT-B/14, arrives in 0002).
- **Idempotency key = `content_hash` (sha256 of media).** Re-ingest of same bytes is a no-op;
  per-stage status in `jobs.stage_status` makes ingestion resumable.
- **Copyright posture:** store derived indexes + source links + short thumbnails; never host
  source video. `samples/` (inputs) and `media/` (derived) are both gitignored.

## Environment (verified)

uv 0.11, Python 3.13 (KITS supports <3.15), ffmpeg 8, Docker + Compose present. No `psql`
client → run migrations via `uv run kanomori-migrate` (Python/psycopg), not the CLI.
KITS checkout at `/Users/lb/Documents/Code/KITS`.

## How to run (once Step 0 finishes)

```bash
uv sync                                   # core + dev only (fast; no torch)
docker compose up -d                      # pgvector on localhost:5433
uv run kanomori-migrate                   # apply migrations/*.sql
uv run pytest                             # unit + mocked-integration
# Step 1 onward:
uv run uvicorn kanomori.api.app:app --reload
uv run kanomori-worker                    # ingestion worker loop
```

## Open decision points (tracked in ARCHITECTURE.md)

- **JP full-text tokenization** — app-side fugashi → `to_tsvector('simple', tokens)`; dense
  vectors cushion gaps; pgroonga/pg_bigm later.
- **BGE-M3 CPU query latency** (~560M params) — preload in lifespan; consider ONNX/int8 or a
  smaller fallback if p95 is too high. Benchmark during Step 1.
- **Karaoke `--separate` trigger** — title-keyword heuristic + manual override; no cheap
  pre-ASR signal.
