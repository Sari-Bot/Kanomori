# Kanomori — Handoff to GPT Agent

Self-contained handoff for a fresh GPT agent to continue the Phase-1 MVP implementation.
This file bundles the current state, the plan, the architecture decisions, the next step's
specifics, and the gotchas — everything you need to pick up and build.

## 1. TL;DR — what this project is

**Kanomori**: multimodal, **moment-level** retrieval over 鹿乃 (Kano Mahoro) VTuber
livestream archives. Given a screenshot, a transcript fragment, or lyrics, return the exact
source stream **and timestamp** — not a title/tag match.

**Two loosely-coupled halves:**
- **Offline ingestion** (GPU-tolerant, batch): register media → extract audio → transcript (via
  KITS subprocess) → frames → OCR → scene classification → image embeddings → indexes.
  Resumable, idempotent, runs on a GPU host.
- **Online query** (CPU, low-latency): lexical + vector + metadata search + scene-aware
  rerank over a small candidate set. **No GPU on the query path.**

**ASR is delegated to KITS** (`/Users/lb/Documents/Code/KITS`, v1.5.2), consumed strictly as
a subprocess — never imported. Kanomori does storage + retrieval.

## 2. Current state — what's built

**Git on `main`:** HEAD = `1fb7710` (`test: add retrieval eval harness`).

Latest main history:

```
1fb7710 test: add retrieval eval harness
e47b175 Merge branch 'feature/visual-e2e-validation'
d996318 test: add visual real-model e2e smoke
330ba37 Merge feature/step3-scene-aware-merge
23f371a feat: add scene-aware result merge
fa18626 Merge feature/step2-visual-search
9528d61 feat: add visual screenshot search
1874ff4 Update HANDOFF: Step 1 done, proven end-to-end
```

The `feature/visual-e2e-validation` branch is now **merged** (`e47b175`), and a retrieval
eval harness has landed on top (`1fb7710`). No unmerged feature branches remain.

**Main worktree status:** `main` has unrelated pre-existing local working-tree changes:
deleted `HANDOFF.md`, deleted `kanomori_project_white_paper.md` (both still present in HEAD —
the deletions are unstaged), and untracked `HANDOFF_TO_GPT.md`. Do not stage or revert those
unless the user explicitly asks.

**Steps completed on `main`:** Step 0, Step 1, Step 2, Step 3, plus an eval-harness baseline
(part of Step 5). Step 4 (minimal UI) and the rest of Step 5 (hardening) remain.

**Step 2 done:** visual schema + frame extraction + OCR + SigLIP scene classification +
DINOv2/pHash frame indexing + `/search/screenshot`.

**Step 3 done:** scene-aware merge/rerank (`retrieval/merge.py`) is used by transcript and
screenshot API paths.

**Eval harness done (`evaluation.py` + `test_retrieval_eval_real_models.py`):** loads an eval
suite (transcript + screenshot query cases), runs retrieval, reports top-k hit presence and
timestamp error. Real-model eval test is `requires_models` and opt-in.

**Validation status (verified this session):**
- `uv run pytest -m "not requires_models"` → all pass (no failures/errors), `uv run ruff
  check .` → clean. ~104 test functions total; the non-model subset is the bulk.
- `uv run pytest -m requires_models` → **all pass** (BGE-M3 text + DINOv2/SigLIP visual + the
  real-model eval), with `KANOMORI_VISUAL_E2E_SAMPLE` pointing at the sample clip. Requires
  `uv sync --group embed --group ingest` first (torchvision is in the `embed` group and is
  needed by DINOv2's `AutoImageProcessor` — if you see "AutoImageProcessor requires the
  Torchvision library", your venv is stale; re-run `uv sync --group embed`).

**Known gaps:**
1. The visual **ranking profiles (gaming/karaoke) remain unverified for quality** — only the
   chatting sample clip exists, so merge weighting for visual-/lyrics-heavy scenes is exercised
   by code paths but not measured. A labelled multi-scene eval set is Step-5 hardening work.
2. Step-1's transcript path is proven end-to-end with real components; the **visual ranking
   profiles (gaming/karaoke) remain unverified** for quality — only the chatting clip exists.

All implementation slices used test-first or test-backed changes.

**Step-1 transcript proof ran with real components:** `samples/2024_talk_cut.mp4` (63s,
chatting/talking, clear JP speech) → ffmpeg → KITS (kotoba-whisper on MPS, 37s) → 23
transcript segments with real BGE-M3 embeddings + Japanese tsvector in PostgreSQL →
hybrid RRF search returns correct timestamps (冷や汗→19.0s, ライブ初参戦→35.0s,
スーパーチャットありがとう→33.3s).

**Visual proof (`test_visual_e2e_real_models.py`, now on `main`):** the same sample clip runs
through real ffmpeg frame extraction, RapidOCR, SigLIP scene classification, DINOv2 frame
embedding, screenshot candidate generation, and scene-aware merge — **once torchvision is
installed** (see gap #1). It deliberately skips KITS: it proves that once transcription is
done, later visual work can be validated without rerunning ASR.

### What's built (file inventory)

**Pure logic** (no DB/GPU; tested independently):
- `src/kanomori/srt.py` — SRT parser mirroring KITS's lenient parser (CRLF, missing index,
  multiline text, fractional timecodes). `Sentence{start, end, text}` TypedDict.
- `src/kanomori/text.py` — NFKC normalize + `tokenize_for_fts()` via fugashi/MeCab (lazy
  import so base imports never require fugashi).
- `src/kanomori/fusion.py` — `reciprocal_rank_fusion()` (scale-free RRF with per-list weights,
  stable tie-break).

**Database:**
- `migrations/0001_init.sql` — `videos`, `jobs`, `transcript_segments` (with
  `vector(1024)` HNSW cosine + tsvector GIN + `(video_id, start_sec)` indexes).
  `CREATE EXTENSION vector;`. Forward-only, applied by `src/kanomori/migrate.py`.
- `migrations/0002_visual.sql` — `frames`, `ocr_segments`, `scene_segments`.
  `frames.embedding` is `vector(768)` for DINOv2; frame pHash is stored as signed `bigint`.
- `src/kanomori/db.py` — psycopg3 pool with pgvector adapters, lazy init, `connection()`
  context manager.
- `src/kanomori/config.py` — `Settings` (pydantic-settings, env-prefixed `KANOMORI_`). Cached.

**KITS boundary:**
- `src/kanomori/kits_client.py` — `transcribe()`: builds argv list → `subprocess.run(cwd=KITS_DIR,
  capture_output=True)` → writes full `.kits.log` artifact → returns the output SRT path.
  Raises `KitsError`. **Paths are resolved to absolute** before building argv (KITS runs
  with a different cwd). Injectable `runner=` for tests.

**Ingestion pipeline:**
- `src/kanomori/ingest/pipeline.py` — `IngestContext` dataclass + `STAGES` list + `run_full()`
  (resumable: skips stages already marked `done` in `jobs.stage_status`, commits per stage).
  `make_embedder()` is overridable.
- `src/kanomori/ingest/artifacts.py` — deterministic content-hash-keyed artifact paths:
  `artifact_dir()`, `audio_path_for()`, `srt_path_for()`, `frame_dir_for()`,
  `frame_path_for()`.
- `src/kanomori/ingest/stages/` — stages are registered in `pipeline.STAGES`:
  - `register.py` — sha256 of media → upsert `videos` + `jobs` rows.
  - `locate_media.py` — ffmpeg extract 16kHz mono WAV; `decide_separate()` karaoke heuristic
    (歌枠/karaoke/cover/...); injectable `_run`.
  - `transcribe.py` — calls `kits_transcribe()`; falls back to deterministic `audio_path_for()`
    when `ctx.audio_path` is unset (resumed runs).
  - `parse_transcript.py` — SRT → transcript_segments rows (text + norm + embedding + tsv);
    delete-then-insert (idempotent); falls back to `srt_path_for()`.
  - `frames.py` — ffprobe video detection + PySceneDetect scene cuts + fixed interval
    samples; extracts downscaled JPEG preview frames.
  - `ocr.py` — RapidOCR per frame → `ocr_segments` with Japanese-tokenized `tsv`.
  - `classify.py` — SigLIP zero-shot scene labels; collapses consecutive same-label frames
    into `scene_segments`; updates `videos.stream_type`.
  - `image_embed.py` — computes frame pHash + DINOv2 image embeddings.
- `src/kanomori/ingest/worker.py` — `claim_one()` (FOR UPDATE SKIP LOCKED), `claim_and_run_one()`
  (marks failed + bumps attempts), `main()` polling loop.

**Embedding:**
- `src/kanomori/embed/text_embedder.py` — `BGEEmbedder`: lazy-loads BGE-M3 dense (1024-d,
  `embed_query` / `embed_texts`). `EMBED_DIM = 1024` pinned constant.
- `src/kanomori/embed/phash.py` — pHash helper that converts unsigned 64-bit image hashes
  into PostgreSQL-compatible signed `bigint`.
- `src/kanomori/embed/image_embedder.py` — `DINOv2Embedder` (768-d) and `SigLIPClassifier`;
  both load lazily to keep the default environment light.

**Retrieval:**
- `src/kanomori/retrieval/transcript.py` — `lexical_candidates()` (tsvector + ts_rank) +
  `dense_candidates()` (pgvector cosine HNSW) → `candidates()` (RRF fusion, dedup by
  segment_id). Returns `list[Candidate]`.
- `src/kanomori/retrieval/screenshot.py` — screenshot candidate generation from uploaded
  image bytes: pHash candidates + DINOv2 dense candidates + OCR lexical candidates, fused by
  RRF.
- `src/kanomori/retrieval/merge.py` — scene-aware bucket merge. It buckets candidates by
  `(video_id, round(ts_sec / 8s))`, looks up `scene_segments`, applies scene-specific modality
  weights, and returns `SearchHit{video_id, ts_sec, score, scene_type, why}`.

**API:**
- `src/kanomori/api/app.py` — FastAPI `create_app()` factory with lifespan (pool +
  preloaded text and image embedders). `POST /search/transcript`, `POST /search/screenshot`,
  `POST /ingest`, `GET /ingest/{id}`. `get_embedder()` / `get_image_embedder()` are
  module-level for test override.

**Cross-layer contracts:**
- `src/kanomori/models.py` — pydantic: `Modality(StrEnum)`, `Candidate`, `SearchHit`,
  `IngestRequest/Response`, `JobStatusResponse`, `SearchResponse`.

**Tests:**
- `tests/unit/` — `test_srt.py`, `test_text.py`, `test_fusion.py`, `test_kits_client.py`,
  `test_text_embedder.py`, `test_locate_media.py`, `test_transcribe_stage.py`,
  `test_phash.py`, `test_frames_stage.py`, `test_merge.py`.
- `tests/integration/` — `test_transcript_retrieval.py`, `test_ingest_pipeline.py`,
  `test_worker.py`, `test_api.py`, `test_visual_schema.py`, `test_visual_stages.py`,
  `test_screenshot_retrieval.py`. `conftest.py` provides `db_conn` (rollback isolation +
  autouse TRUNCATE for committing tests), `fake_embedder` (deterministic unit-norm vectors),
  `_close_pool_at_session_end`.
- `tests/integration/test_visual_e2e_real_models.py` exists only on
  `feature/visual-e2e-validation` until that branch is merged.
- `tests/fixtures/sample.srt` — 4‑cue SRT in KITS format.

## 3. Key invariants (DO NOT BREAK)

1. **Never `import kits`.** KITS is AGPL and GPU-heavy. Reach it only via
   `kanomori.kits_client` shelling out to `uv run kits subtitle ...` with `cwd=KITS_DIR`,
   argv as a **list** (no shell string).
2. **Kanomori has its own SRT parser** (`srt.py`). Can't import `kits.parse_srt`. Mirror
   KITS's leniency.
3. **Online query path is CPU-only.** GPU only for offline ingestion + KITS.
4. **Embedding dims are pinned in SQL** (authoritative, not config strings):
   transcript = `vector(1024)` (BGE-M3), frames = `vector(768)` (DINOv2 ViT-B/14).
5. **Idempotency key = `content_hash`** (sha256 of media). Re-ingest same bytes = no-op.
   Resumable via `jobs.stage_status` per-stage marking.
6. **Cross-stage artifacts use deterministic content-hash paths** (`ingest/artifacts.py`).
   A stage that reads a prior stage's output must derive it from `content_hash`, falling
   back when the IngestContext field is `None` (skipped stages leave context holes on resume).
   See gotcha #3.
7. **KITS receives absolute paths only.** `transcribe()` calls `.resolve()` on both paths
   (KITS runs with `cwd=KITS_DIR`; relative paths resolve against the wrong dir).
   See gotcha #2.
8. **Copyright posture:** derived indexes + source links + short thumbnails only; never
   host source video. `samples/` and `media/` are gitignored.

## 4. Environment (verified)

- **uv 0.11**, Python 3.13, ffmpeg 8, Docker + Compose
- **KITS checkout:** `/Users/lb/Documents/Code/KITS` (uv-synced in its own venv; `kits` CLI works)
- **PostgreSQL + pgvector:** `docker compose up -d` → pg16 on `localhost:5433`, user/pass/db = `kanomori`
- **No psql client** → run migrations via `uv run kanomori-migrate` (Python/psycopg)
- **GPU:** Apple Silicon MPS (Mac). KITS uses MPS; DINOv2/SigLIP step will too.
- **Test clip:** `samples/2024_talk_cut.mp4` (63s, chatting/talking, 17MB)

## 5. Architecture summary — patterns to follow when extending

### Adding a new ingestion stage

1. Create `src/kanomori/ingest/stages/<name>.py` with `def run(conn, ctx) -> None`.
2. Add an artifact path helper to `ingest/artifacts.py` if the stage produces a file.
3. Register the stage in `pipeline.py`'s `STAGES` list (order matters).
4. Add `ctx` fields to `IngestContext` if downstream stages need them.
5. Always fall back to the deterministic artifact path when the `ctx` field might be `None`
   (see gotcha #3).
6. Write unit tests (injectable dependencies) + an integration test through `run_full`.

### Adding a new modality searcher

Follow `retrieval/transcript.py`'s pattern:
- One function per candidate source (e.g. `lexical_candidates`, `dense_candidates`).
- A public `candidates()` that fuses them with `reciprocal_rank_fusion()`.
- Each returns `list[Candidate]` with `modality` set.
- Use `retrieval.merge.merge_from_db()` to combine per-modality lists into `SearchHit`s.

### Adding a new table

- Create `migrations/NNNN_descriptive.sql` (forward-only plain SQL).
- Pin embedding dims in SQL (authoritative) — keep them in sync with the chosen model.
- Re-run `uv run kanomori-migrate` (idempotent; already-applied migrations are skipped).

### Writing tests

- **Unit tests** go in `tests/unit/` — no DB, no GPU, no models. Test pure logic.
- **Integration tests** go in `tests/integration/` — use `db_conn` fixture (requires the
  pgvector container; `uv sync --group ingest` for fugashi). Mark `pytestmark = pytest.mark.requires_db`.
- **Tests that need real models** use `@pytest.mark.requires_models` (mirrors KITS's
  `requires_torch`). The lightweight `uv sync` skips them via `-m "not requires_models"`.
- **Tests that mock KITS** patch `kanomori.ingest.stages.transcribe.kits_transcribe` to
  drop the fixture SRT (no GPU). See `test_ingest_pipeline.py`'s `fake_transcribe` fixture.
- **Tests that mock the embedder** use `conftest.py`'s `fake_embedder` — deterministic
  unit-norm vectors; real pgvector math, only the model substituted.
- **Pool teardown** is handled by the session-scoped `_close_pool_at_session_end` fixture.
  **Table isolation** is handled by the autouse `_clean_tables` fixture (TRUNCATE before
  each test, because committing tests leak rows past rollback-based isolation).

## 6. Next step: fix torchvision gap, then Step 4 (minimal UI)

### Immediate fix (do this first — it's blocking visual validation)

The `embed` dependency group is missing **torchvision**, which DINOv2's `AutoImageProcessor`
requires. Right now the 2 `requires_models` visual/eval tests error at import. This is the
first thing to fix before trusting any visual end-to-end claim.

```bash
cd /Users/lb/Documents/Code/kanomori
# Add torchvision to the [dependency-groups] embed list in pyproject.toml, then:
uv sync --group embed --group ingest
# Confirm DINOv2/SigLIP load and the real-model visual + eval tests pass:
KANOMORI_VISUAL_E2E_SAMPLE=/Users/lb/Documents/Code/kanomori/samples/2024_talk_cut.mp4 \
  uv run pytest -m requires_models -q
```
(TDD note: the failing `requires_models` tests already exist and currently error — adding
torchvision is the minimal change to turn them green. Watch them go red→green.)

### Then: Step 4 — minimal UI (the remaining unbuilt step)

Steps 0–3 + the eval-harness baseline are done. The transcript and screenshot search APIs
work (`POST /search/transcript`, `POST /search/screenshot`), and `retrieval/merge.py` already
returns scene-aware `SearchHit{video_id, ts_sec, score, scene_type, why}`. What's missing is a
human-facing UI.

Per the approved plan, build **server-rendered Jinja2 + htmx, no SPA, no build step**:
- **Search page** — a query box (text) + an image upload (screenshot), posting to the existing
  search endpoints.
- **Results** — cards showing thumbnail / timestamp / transcript snippet / "why" badges (the
  `why` dict from `SearchHit`) / source link.
- **Result detail** — `GET /result/{video_id}?ts=` returning the moment with nearby transcript
  (±window via `(video_id, start_sec)` index), preview frames (from `frames`), OCR context,
  and a jump-to-source link. This endpoint does **not exist yet** — add it.
- Mount frame thumbnails via FastAPI `StaticFiles` from `MEDIA_ROOT`. Keep previews short;
  never serve source video (copyright invariant #8).
- `jinja2` is already a core dep; `python-multipart` (for the screenshot upload) is too.

### After Step 4: rest of Step 5 (hardening), then audio is a separate design

The eval harness exists but runs on one sample (proves code paths, not retrieval quality).
Remaining Step-5 hardening: a real labelled eval set (white paper §13: 100 transcript + 100
screenshot queries, Top-5 as headline), worker retry/attempts-cap exercise, and writing the
real run commands into the repo `CLAUDE.md`. Audio snippet / karaoke search (Phase 3) stays
out — `Modality.AUDIO` remains reserved and weight `0` in Phase-1 profiles; it needs its own
design pass for `audio_fingerprints` storage, query representation, and eval data.

## 7. Test patterns (quick reference)

```bash
# Lightweight: core + dev only (no torch, no fugashi)
uv sync && uv run pytest -m "not requires_db and not requires_models"

# Integration: needs pgvector container
docker compose up -d && uv sync --group ingest && uv run pytest -m "requires_db"

# Full: everything including real BGE-M3/DINOv2; needs local sample for visual E2E
uv sync --group embed --group ingest && uv run pytest

# Real visual E2E smoke (currently on feature/visual-e2e-validation)
KANOMORI_VISUAL_E2E_SAMPLE=/Users/lb/Documents/Code/kanomori/samples/2024_talk_cut.mp4 \
  uv run pytest tests/integration/test_visual_e2e_real_models.py -q -m requires_models

# Single test file
uv run pytest tests/unit/test_srt.py -v
```

Fixtures available (from `tests/integration/conftest.py`):
- `db_conn` — pooled pgvector connection, rolled-back transaction. Skip if no DB reachable.
- `fake_embedder` — deterministic `FakeEmbedder` (1024-d unit-norm vectors, seeded per text
  hash. `embed_query(text) -> np.ndarray`, `embed_texts(texts) -> list[np.ndarray]`).

## 8. Gotchas — bugs the real run caught

These are non-obvious and easy to reintroduce. Read before writing ingestion/retrieval code.

### Gotcha #1: kits_client must capture full stderr to a log artifact

KITS crashes deep in transformers/torch with long tracebacks. The `KitsError` message is
necessarily bounded (exception messages shouldn't be multi-KB), so `transcribe()` writes
a full `<out>.kits.log` next to the SRT (argv, cwd, returncode, **untruncated** stdout+stderr)
on every run. On failure, the error message references this log path. **Always read the
`.kits.log` first when a transcribe stage fails.**

### Gotcha #2: kits_client must pass absolute paths to KITS

KITS runs with `cwd=KITS_DIR`, so a relative `-i`/`-o` path resolves against KITS's
directory, not kanomori's. `transcribe()` calls `.resolve()` on both paths before building
argv and `mkdir`-ing. If you ever call `build_kits_argv` directly, you must do the same.

### Gotcha #3: skipped stages on resume leave IngestContext fields unset

`run_full` skips stages already marked `done`. On a resumed run, a skipped stage never
executes, so the `ctx` field it would have set (`audio_path`, `srt_path`, etc.) is `None`
that session. Any downstream stage that reads that field must fall back to the deterministic
artifact path from `ingest/artifacts.py`:

```python
# Correct — works on both fresh and resumed runs:
audio = ctx.audio_path or str(audio_path_for(ctx.content_hash))

# Wrong — fails silently on resume (ctx.audio_path is None):
audio = ctx.audio_path or ctx.media_path  # ctx.media_path is the raw .mp4, undecodable
```

This bit transcribe.py (fell back to raw .mp4, caught by the e2e run) and was already
fixed for both audio and SRT. Any **new** cross-stage artifact you add must follow the
same pattern. Add an artifact path helper, use it as the fallback, and write a unit test
asserting the fallback is used when the `ctx` field is `None`.

### Gotcha #4: jsonb operators need `::text` casts on params

PostgreSQL's `jsonb -> %s` and `jsonb_build_object(%s, ...)` can't infer the type of an
untyped parameter (IndeterminateDatatype). Add `::text` casts:

```sql
-- Wrong: IndeterminateDatatype
SELECT stage_status -> %s ->> 'state' FROM jobs WHERE ...
jsonb_build_object(%s, ...)

-- Correct
SELECT stage_status -> %s::text ->> 'state' FROM jobs WHERE ...
jsonb_build_object(%s::text, ...)
```

### Gotcha #5: test isolation — TRUNCATE before tests that commit

`db_conn` rolls back each test's transaction, but tests that call `conn.commit()` (pipeline,
worker, `/ingest`) durably persist rows. The `_clean_tables` autouse fixture TRUNCATEs
`videos, jobs, frames, ocr_segments, scene_segments RESTART IDENTITY CASCADE` before each
integration test. If adding new persistent tables, update the TRUNCATE list in
`tests/integration/conftest.py`.

```python
c.execute("TRUNCATE videos, jobs, frames, ocr_segments, scene_segments RESTART IDENTITY CASCADE")
```

## 9. Quick start

```bash
cd /Users/lb/Documents/Code/kanomori

# Start the database
docker compose up -d                          # pgvector on localhost:5433

# Install deps and verify
uv sync                                       # core + dev (fast, no torch)
uv sync --group ingest --group embed          # + frames/OCR/tokenization/models
uv run kanomori-migrate                       # idempotent
uv run pytest -m "not requires_models"        # all pass (the bulk of ~104 tests)
uv run ruff check .                           # should be clean
# requires_models tests (BGE-M3 + DINOv2/SigLIP + eval) pass after `uv sync --group embed
# --group ingest`; torchvision is in the embed group. Set KANOMORI_VISUAL_E2E_SAMPLE for the
# visual E2E smoke. If you see "AutoImageProcessor requires the Torchvision library", your venv
# is stale — re-run `uv sync --group embed`.

# Run the API
uv run uvicorn kanomori.api.app:app --reload  # http://localhost:8000

# Ingest a clip
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"media_path": "samples/2024_talk_cut.mp4", "title": "test"}'
# → {"job_id": 1, "content_hash": "abc...", "status": "queued"}

# Run the worker to process it (separate terminal)
uv run kanomori-worker

# Check status, then search
curl http://localhost:8000/ingest/1
curl -X POST http://localhost:8000/search/transcript \
  -H "Content-Type: application/json" \
  -d '{"query": "冷や汗", "k": 5}'

# Search by screenshot
curl -X POST http://localhost:8000/search/screenshot \
  -F "file=@media/<content_hash>/frames/frame_000000_000.jpg" \
  -F "k=5"
```

---

## Source docs (read if you need more depth)

| File | What it covers |
|------|---------------|
| `docs/ARCHITECTURE.md` | Full engineering decisions, white-paper deviations, data model, retrieval design, open decision points |
| `HANDOFF_TO_GPT.md` | Current continuation handoff for the next GPT/Codex session |
| `migrations/0001_init.sql` | Transcript/job/video schema (authoritative transcript dim: vector(1024)) |
| `migrations/0002_visual.sql` | Visual schema (frames/OCR/scenes; authoritative image dim: vector(768)) |
| `src/kanomori/models.py` | Cross-layer pydantic contracts (Candidate, SearchHit, Modality enum) |
| `src/kanomori/ingest/pipeline.py` | IngestContext + STAGES list + run_full resume logic |
| `src/kanomori/retrieval/transcript.py` | Transcript candidate generation pattern |
| `src/kanomori/retrieval/screenshot.py` | Screenshot candidate generation pattern |
| `src/kanomori/retrieval/merge.py` | Scene-aware merge/rerank |
| `tests/integration/conftest.py` | Test fixtures (db_conn, fake_embedder, isolation) |
