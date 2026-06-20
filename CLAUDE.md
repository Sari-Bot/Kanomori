# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This repository is **pre-implementation**. The only file is
`kanomori_project_white_paper.md`, a detailed design document. There is no
code, build system, test suite, or dependency manifest yet, and it is not a
git repository. When you add the first implementation, you are also
establishing the tooling — see "When implementation begins" below.

Treat the white paper as the source of truth for product scope and
architecture. This file summarizes the decisions a contributor needs up front;
the white paper has the full reasoning.

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

## MVP scope — build this first, postpone the rest

In scope (Phase 1–2): video import, frame extraction every 5–10s, transcript
generation, OCR, basic scene classification, transcript search, screenshot
search, and timestamp-level results with nearby transcript + preview frames.

Explicitly **postponed**: full clip reverse search, audio fingerprinting /
karaoke search (Phase 3+), custom-trained visual models, real-time indexing,
knowledge graph, recommendations, public deployment. Don't pull these forward
without a reason — overengineering the MVP is called out as the primary risk.

Roadmap order: transcript search → screenshot + OCR → scene-aware rerank →
audio snippet → karaoke + clip reverse search → vague-memory + evidence-based
AI Q&A.

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

## When implementation begins

No commands exist yet. As the first contributor to add code, establish (and then
document here): the dependency manifest, how to run the FastAPI backend, how to
run the ingestion pipeline, and the test runner + how to run a single test.
Update this file with the real commands once they exist — do not leave invented
ones.
