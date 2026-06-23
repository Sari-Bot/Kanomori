"""Runtime configuration, loaded from environment / .env via pydantic-settings.

A single ``Settings`` instance (``get_settings()``, cached) is the one source of truth for
the DB DSN, the KITS checkout location, media root, and model ids. Keeping this centralized
means the ingestion worker, the API, and tests all resolve configuration the same way.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # Model ids. Embedding dims are pinned as constants in code (see embed/ and migrations),
    # not derived from these strings, to keep the SQL schema authoritative.
    text_model: str = "BAAI/bge-m3"
    image_model: str = "facebook/dinov2-base"
    scene_model: str = "google/siglip-base-patch16-224"
    ocr_engine: str = "legacy_rapidocr"

    # Subprocess timeout (seconds) for a single KITS transcription. None = no limit.
    kits_timeout: float | None = None


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached Settings. Override env vars to influence it in tests."""
    return Settings()
