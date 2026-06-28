"""Runtime configuration, loaded from environment / .env via pydantic-settings.

A single ``Settings`` instance (``get_settings()``, cached) is the one source of truth for
the DB DSN, the KITS checkout location, media root, and model ids. Keeping this centralized
means the ingestion worker, the API, and tests all resolve configuration the same way.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

StageDevice = Literal["cpu", "gpu"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KANOMORI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # PostgreSQL + pgvector DSN. Defaults match docker-compose.yml (host port 5433).
    database_url: str = "postgresql://kanomori:kanomori@localhost:5433/kanomori"

    # Sibling KITS checkout; ingestion runs `uv run kits subtitle` with this as cwd.
    kits_dir: Path = Path("/Users/lb/Documents/Code/KITS")

    # Root for derived media (frame thumbnails, KITS SRT/log artifacts). Never source video.
    media_root: Path = Path("./media")

    # Source store the worker READS from. "local" mirrors samples/ on disk; "webdav" is the
    # production HTTPS store. The layout is identical in both (see samples/README.md).
    media_source: str = "local"
    # Local mirror root when media_source == "local"; = samples/ in dev.
    media_source_root: Path = Path("./samples")
    # WebDAV base URL + optional basic-auth creds, used when media_source == "webdav".
    media_source_url: str | None = None
    media_source_user: str | None = None
    media_source_password: str | None = None

    # Shared bearer token authenticating workers to the coordinator /jobs router. Defined here so
    # both sides resolve the same secret; None disables auth (dev only).
    coordinator_token: str | None = None
    # Base URL a distributed worker reaches the coordinator /jobs router at. Default suits a
    # worker co-located with the coordinator; remote workers set KANOMORI_COORDINATOR_URL.
    coordinator_url: str = "http://localhost:8000"
    # Max accepted size of the worker->coordinator stage result JSON upload, in bytes.
    stage_result_max_bytes: int = 64 * 1024 * 1024

    # Model ids. Embedding dims are pinned as constants in code (see embed/ and migrations),
    # not derived from these strings, to keep the SQL schema authoritative.
    text_model: str = "BAAI/bge-m3"
    image_model: str = "facebook/dinov2-base"
    scene_model: str = "google/siglip-base-patch16-224"
    # Required for /search/audio, but intentionally has no baked model default: operators must
    # set it to the kotoba-whisper version that produced the indexed corpus.
    audio_asr_model: str | None = None
    audio_clip_max_sec: float = 35.0
    # Preload online search models during API startup. Set false for lightweight dev/test runs.
    preload_search_models: bool = True

    # OCR config is split by workload because offline ingestion can favor accuracy while
    # screenshot query may need a lower-latency profile.
    ingest_ocr_model: str = "ppocrv5_server"
    ingest_ocr_backend: str = "onnxruntime"
    query_ocr_model: str = "ppocrv5_server"
    query_ocr_backend: str = "onnxruntime"
    stage_parse_transcript_device: StageDevice = "cpu"
    stage_ocr_device: StageDevice = "cpu"
    stage_classify_device: StageDevice = "cpu"
    stage_image_embed_device: StageDevice = "cpu"
    # Deprecated migration input for the old flat engine names.
    ocr_engine: str | None = None

    # Subprocess timeout (seconds) for a single KITS transcription. None = no limit.
    kits_timeout: float | None = None


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached Settings. Override env vars to influence it in tests."""
    return Settings()
