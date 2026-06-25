"""Cross-layer data contracts (pydantic) shared by ingestion, retrieval, and the API.

Kept deliberately small and stable: only the types that cross a module boundary live here.
Module-local shapes stay in their own module — e.g. the SRT ``Sentence`` lives in
``kanomori.srt`` (pure logic, no pydantic), and scene weight profiles live in
``kanomori.scene``. ``scene_type`` is carried as a plain ``str`` here to avoid coupling the
API/retrieval contracts to the scene enum that arrives in a later step.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Modality(StrEnum):
    """A retrieval signal. ``why`` on a SearchHit maps each contributing modality to its
    contribution, and scene-aware weight profiles are keyed by these names."""

    TRANSCRIPT = "transcript"
    OCR = "ocr"
    VISUAL = "visual"
    SCENE = "scene"
    AUDIO = "audio"  # reserved for Phase 3 (audio fingerprinting); weight 0 in Phase-1


class Candidate(BaseModel):
    """One per-modality candidate, before cross-modality merge.

    Each modality searcher returns a rank-ordered list of these. ``rank`` (0-based) is what
    Reciprocal Rank Fusion consumes; ``raw_score`` is retained for debugging/inspection only
    (RRF is scale-free and ignores it).
    """

    video_id: int
    ts_sec: float
    modality: Modality
    rank: int
    raw_score: float | None = None
    segment_id: int | None = None  # source row (transcript_segments.id / frames.id / ...)


class SearchHit(BaseModel):
    """A merged, scene-aware-ranked result at moment granularity."""

    video_id: int
    ts_sec: float
    score: float
    scene_type: str | None = None
    why: dict[str, float] = Field(default_factory=dict)  # modality name -> contribution


# --- API request/response DTOs ---------------------------------------------------------


class IngestRequest(BaseModel):
    """Trigger ingestion of a local media file. ``media_path`` is a path on the ingestion
    host; we store a ``source_url`` link rather than redistributing the video."""

    media_path: str
    source_url: str | None = None
    source_platform: str | None = None
    title: str | None = None
    stream_type: str | None = None  # caller hint; refined by scene classification later
    separate: bool = False  # force KITS vocal isolation (karaoke); else heuristic decides


class IngestResponse(BaseModel):
    # content_hash is sha256 of the media, computed by the register stage when the worker runs
    # — not known at enqueue time. /ingest therefore returns None here; clients poll
    # GET /ingest/{job_id} for the resolved hash once registration completes.
    job_id: int
    content_hash: str | None = None
    status: str


class BatchIngestRequest(BaseModel):
    """Body for ``POST /ingest/batch``. ``manifest_path`` is resolved by the configured
    MediaSource (a source-store-relative key), not a host path; it defaults to the conventional
    ``manifest.jsonl`` at the store root."""

    manifest_path: str = "manifest.jsonl"


class BatchIngestResponse(BaseModel):
    """Result of enqueuing a manifest. ``enqueued`` are the new job ids (one per fresh manifest
    line); ``skipped`` are the source-store ``path`` strings that already had a queued/running job
    (so re-running a batch is idempotent); ``total`` is the manifest line count."""

    enqueued: list[int] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    total: int


class JobStatusResponse(BaseModel):
    job_id: int
    status: str
    current_stage: str | None = None
    stage_status: dict = Field(default_factory=dict)
    error: str | None = None


class SearchResponse(BaseModel):
    hits: list[SearchHit]
