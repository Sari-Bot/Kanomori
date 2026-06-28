"""Smoke tests for the Step-0 scaffolding: the package imports, config loads, and the
migration files are well-formed. These need no DB, GPU, or ML models — they keep the suite
green from the first commit and verify the src-layout / pythonpath wiring.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import kanomori
from kanomori.config import Settings
from kanomori.migrate import MIGRATIONS_DIR


def test_package_version() -> None:
    assert kanomori.__version__


def test_settings_defaults_load() -> None:
    # Construct directly (not the cached get_settings) so this is independent of the
    # environment / a present .env file.
    s = Settings(_env_file=None)
    assert s.database_url.startswith("postgresql://")
    assert s.text_model  # a non-empty model id
    assert s.ingest_ocr_model == "ppocrv5_server"
    assert s.ingest_ocr_backend == "onnxruntime"
    assert s.query_ocr_model == "ppocrv5_server"
    assert s.query_ocr_backend == "onnxruntime"
    assert s.preload_search_models is True
    assert s.stage_parse_transcript_device == "cpu"
    assert s.stage_ocr_device == "cpu"
    assert s.stage_classify_device == "cpu"
    assert s.stage_image_embed_device == "cpu"


def test_stage_device_settings_accept_cpu_and_gpu() -> None:
    s = Settings(
        stage_parse_transcript_device="gpu",
        stage_ocr_device="cpu",
        stage_classify_device="gpu",
        stage_image_embed_device="cpu",
        _env_file=None,
    )

    assert s.stage_parse_transcript_device == "gpu"
    assert s.stage_ocr_device == "cpu"
    assert s.stage_classify_device == "gpu"
    assert s.stage_image_embed_device == "cpu"


def test_preload_search_models_setting_accepts_env_false(monkeypatch) -> None:
    monkeypatch.setenv("KANOMORI_PRELOAD_SEARCH_MODELS", "false")

    s = Settings(_env_file=None)

    assert s.preload_search_models is False


def test_stage_device_settings_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="stage_parse_transcript_device"):
        Settings(stage_parse_transcript_device="cuda", _env_file=None)


def test_migrations_present_and_ordered() -> None:
    files = sorted(MIGRATIONS_DIR.glob("[0-9]*.sql"))
    assert files, "expected at least one migration"
    # Filenames are zero-padded numeric prefixes, so lexical sort == apply order.
    names = [f.name for f in files]
    assert names == sorted(names)
    assert names[0] == "0001_init.sql"


def test_init_migration_defines_core_tables() -> None:
    sql = (MIGRATIONS_DIR / "0001_init.sql").read_text(encoding="utf-8").lower()
    for table in ("videos", "jobs", "transcript_segments"):
        assert f"create table if not exists {table}" in sql
    assert "create extension if not exists vector" in sql
    assert "vector(1024)" in sql  # BGE-M3 dense dim is pinned in SQL


def test_repo_has_architecture_doc() -> None:
    root = Path(kanomori.__file__).resolve().parents[2]
    assert (root / "docs" / "ARCHITECTURE.md").is_file()
