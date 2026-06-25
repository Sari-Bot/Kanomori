# Architecture

[简体中文](ARCHITECTURE.zh-CN.md) | English

Kanomori is a multimodal retrieval system for VTuber livestream archives. The practical goal is
to recover the exact stream and timestamp behind an incomplete memory, not just find a video title
or tag match.

## Core split

The system is intentionally divided into two halves:

- Offline ingestion: GPU-tolerant, batch-oriented, writes derived artifacts and indexes
- Online query: CPU-oriented, latency-sensitive, reads persisted indexes and reranks candidates

These halves do not share a long-running process. The ingestion side prepares data; the API side
serves queries from PostgreSQL-backed indexes.

## Main ingestion path

The current ingest pipeline is:

1. Register media and job metadata
2. Locate source media
3. Extract or prepare audio
4. Transcribe through KITS as an external subprocess
5. Parse transcript segments
6. Extract frames
7. Run OCR
8. Classify scenes and build image embeddings

The worker persists progress stage by stage so interrupted runs can resume.

## Main query path

Implemented query surfaces:

- Transcript search: lexical + dense retrieval over transcript segments
- Screenshot search: OCR + image embeddings + merge/rerank

Candidate generation is modality-specific. Final ranking is merged into timestamped hits tied to
the original stream.

## Important design choices

### KITS stays outside the app process

Kanomori calls KITS with `uv run kits subtitle ...` from `KANOMORI_KITS_DIR`. It does not import
KITS as a Python library. This keeps the ASR stack isolated from the main API/query environment
and preserves a clean boundary between offline transcription and online retrieval.

### PostgreSQL is the center of truth

The current implementation uses PostgreSQL + pgvector for job state, metadata, lexical search,
and vector search. The coordinator owns database writes in distributed mode.

### Workers talk to the coordinator over HTTP

Remote workers never write PostgreSQL directly. They claim jobs, send heartbeats, push stage
results, and mark completion through authenticated `/jobs/*` endpoints.

### Query stays lighter than ingest

Heavy compute belongs to the worker path. The online query API is designed to avoid becoming
dependent on the full offline model/toolchain.

## Current boundaries

- `src/kanomori/api/`: public API and server-rendered demo UI
- `src/kanomori/ingest/`: worker loop, stages, lease logic, and coordinator client
- `src/kanomori/retrieval/`: transcript and screenshot retrieval pipelines
- `src/kanomori/embed/`: text and image embedders
- `src/kanomori/models.py`: request and response shapes

## Roadmap boundary

The repository currently implements ingestion plus transcript/screenshot retrieval. Audio snippet
search, karaoke reverse search, clip reverse search, and broader memory-oriented assistant flows
are still future work and should not be documented as already available.
