# Distributed batch ingestion — design

*Status: implemented (2026-06-24). This records the design as built; where it
drifts from the original plan, the code is authoritative.*

## Goal

Run ingestion across multiple GPU machines under one operator. A central
**coordinator** (the existing FastAPI app) owns the job queue and *all*
DB/artifact persistence; **workers** claim jobs over HTTP, pull source video
from a shared store (WebDAV in prod, local `samples/` in dev), run the heavy
compute, and push results back. Worker disconnection must never deadlock a job
in `running` — handled by a **lease + heartbeat + fencing** scheme.

Locked decisions:
- Transport: **HTTP coordinator only** — workers never touch Postgres.
- Artifacts: worker **pushes** derived artifacts to the coordinator → stored
  under `MEDIA_ROOT/{content_hash}/` (unchanged layout). WebDAV is *source-only*.
- Recovery: lease with TTL; on expiry, claimable by *any* worker; resume from
  the last server-confirmed stage; a **fencing token** guards slow-vs-dead races.
- Auth: a single shared **bearer token** over TLS (or a Tailscale mesh).
- Workers are machines the operator controls (trusted-but-remote).

## Stage refactor — compute / persist split

Every pipeline stage splits into:
- `compute(ctx) -> StageResult` — pure, **no `conn`**, runs on the worker (GPU).
- `persist(conn, video_id, result)` — holds `conn`, runs on the coordinator.
- `run(conn, ctx)` is kept as `persist(conn, vid, compute(ctx))`, so the
  single-machine path and every pre-existing stage test keep working unchanged.

`register.persist` is the exception: it returns `video_id` and is called first;
`frames.persist` takes `content_hash` to lay JPEGs under `MEDIA_ROOT`. The
downstream visual stages (`ocr`/`classify`/`image_embed`) no longer
`SELECT FROM frames` — `compute` globs the frame JPEGs from disk, and `persist`
resolves `frame_id` by `(video_id, ts_sec)`.

## Wire contract + coordinator protocol

`StageResult` (Pydantic, one model per stage) is a list of **natural-keyed**
row-dicts plus a manifest of artifact files. No database ids cross the wire:
video is keyed by `content_hash`, frame by `ts_sec` (`UNIQUE(video_id, ts_sec)`).
Embedding vectors serialize as **base64 float32** (`encode_vector`/`decode_vector`
in `ingest/stage_result.py`).

`/jobs/*` router on the FastAPI app, behind a bearer token
(`require_coordinator_token`: token **unset ⇒ 503** fail-closed; wrong ⇒ 401 via
constant-time compare, never logged):
- `POST /jobs/claim` → leases the oldest eligible job; returns
  `{job_id, content_hash, request, lease_epoch, stages_done[]}` (204 when idle).
  Eligible = `queued` OR (`running` AND `lease_expires_at < now()`) OR
  (`failed` AND `attempts < MAX_ATTEMPTS`).
- `POST /jobs/{id}/heartbeat` (`lease_epoch`) → extends lease; **409 if stale**.
- `POST /jobs/{id}/stage/{name}` (multipart: form `lease_epoch`, form `result`
  JSON, file field `files`) → fence-check, run `persist()` in a txn, store
  artifacts under `MEDIA_ROOT/{content_hash}/`, mark stage done. **409 if stale**.
- `POST /jobs/{id}/complete` / `POST /jobs/{id}/fail` (`lease_epoch`, `error?`).

Fencing: every mutating call carries `lease_epoch`; the coordinator updates
`... WHERE id=%s AND lease_epoch=%s`; 0 rows → 409 → the worker knows it was
reclaimed and aborts. Push cadence is **per-stage**, so a worker dying after
transcribe keeps the expensive GPU SRT on the coordinator.

## Schema delta — `migrations/0003_lease.sql`

Adds to `jobs`: `worker_id text`, `lease_epoch int NOT NULL DEFAULT 0`,
`lease_expires_at timestamptz`, `heartbeat_at timestamptz`, and a
`jobs_claimable_idx ON jobs (status, lease_expires_at)`. Claim bumps
`lease_epoch`, sets `worker_id` and `lease_expires_at = now() + interval`.

Also fixes a latent enqueue quirk: the API used to enqueue with placeholder
`content_hash = md5(media_path)` while `register` inserted a *second* jobs row
under the real sha256. Now the source path lives in `stage_status->'request'`,
`content_hash` is nullable-until-register (`ALTER COLUMN ... DROP NOT NULL`), and
register reconciles by `job_id`. The double-row orphan is gone.

## Source access — the `MediaSource` seam

Read-only Protocol (`media_source.py`); workers only read source, derived
artifacts go to the coordinator over HTTP, never back to the source store:

```python
class MediaSource(Protocol):
    def fetch(self, path: str, dest: Path) -> Path: ...   # download → local temp
    def read_text(self, path: str) -> str: ...            # read manifest.jsonl
```

Two **real** backends (the local one is not a test double — same code path minus
the network), selected by `KANOMORI_MEDIA_SOURCE`:
- `WebDAVSource` — prod. Plain HTTPS GET (httpx). No PROPFIND/XML; the batch list
  comes from `manifest.jsonl`, not directory listing.
- `LocalDirSource` — dev, rooted at `samples/`; path-traversal guarded.

Source folder layout (simplified — no platform layer):

```
<root>/                          # webdav root  ·  local: ./samples
  manifest.jsonl                 # batch input — one job spec per line
  <title>[_<date>]/
    video.<ext>
```

Folder naming is human convenience; the manifest `path` is the source-of-truth
key. A `manifest.jsonl` line (lands in `stage_status->'request'`):

```json
{"path":"kano元気_2025-08-04/video.mp4","title":"kano元気","streamed_at":"2025-08-04","source_url":"https://…","separate":false}
```

`POST /ingest/batch` reads `manifest.jsonl` via the `MediaSource`, enqueues one
job per line, and **dedups** by `stage_status->'request'->>'path'` (re-runs skip
already-queued entries), returning `{enqueued, skipped, total}`. Promote a
validated batch to WebDAV by flipping `KANOMORI_MEDIA_SOURCE` and copying the
tree — byte-identical layout.

## Resumable units & recovery

Unit = stage. Checkpoint principle:

> A unit's checkpoint = (DB rows + product artifacts durably on the coordinator)
> + a small **recipe** of scalars that lets any worker cheaply rebuild that
> unit's *inputs* locally from the source video.

The only two genuinely expensive ops are never redone, because their *small*
outputs are checkpointed:
- **KITS transcribe (GPU)** → `transcript.srt` persisted; a resuming worker
  fetches it (KB–MB).
- **scene detect (full-decode pass)** → the scene-timestamp list is persisted as
  a recipe; a resuming worker replays ffmpeg over known timestamps (cheap),
  skipping detection.

A worker rebuilds shared local inputs **once per claim**, then runs all not-done
eligible stages in that session. Frame JPEGs upload once when `frames`
completes; resuming workers re-extract locally rather than download (the
coordinator's copy is canonical for serving). The one unavoidable cost is
re-pulling source video on **cross-worker** resume; a worker reclaiming its own
job checks a local cache by `content_hash` and skips the re-pull.

## Dry-run — two levels

- **L0 compute-only** (`kanomori-worker --compute-only`, `MEDIA_SOURCE=local`):
  the full `compute()` chain in-memory over a sample, serialize each
  `StageResult`, assert round-trip. No DB / coordinator / network. Regression
  test: `tests/integration/test_compute_only_dry_run.py` (mocks only the four
  heavy models — KITS, BGE-M3, SigLIP, DINOv2 — and runs ffmpeg/scenedetect/OCR
  for real on the 63 s `2024_talk_cut` clip).
- **L1 single-process e2e** (`kanomori-worker --once` against scratch Postgres):
  real claim/lease/fence + compute + HTTP push + persist on `samples/`. Covered
  by the `/jobs` router integration tests in `tests/integration/test_coordinator.py`.

## Test coverage

- **Unit (no DB):** `StageResult` (de)serialization incl. base64 float32 vectors;
  frame-name round-trip / scene-ts recipe replay; compute/persist split;
  `MediaSource`; `/jobs` router auth; coordinator client; batch dedup.
- **Integration (requires_db):** claim atomicity (SKIP LOCKED, one winner);
  lease expiry → reclaim with bumped epoch; fence rejection (stale epoch → 409);
  heartbeat extension; attempts cap; full claim → stage push → complete; and
  **kill-and-reclaim** — worker A marks register+transcribe done, lease lapses, B
  reclaims at `epoch+1` and learns `stages_done`, zombie A is fenced on every
  mutation, B completes (asserts the GPU SRT is not recomputed).

## Out of scope

Audio/karaoke/clip modalities, MinIO, Celery/RQ, multi-coordinator HA. This is
the **Scale & Evaluate** enabler, not a new retrieval modality.
