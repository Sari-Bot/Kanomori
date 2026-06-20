"""Kanomori: multimodal moment-level retrieval for VTuber livestream archives.

The package is layered to keep an `import kanomori` cheap and dependency-light:

- Pure-logic modules (``srt``, ``fusion``, ``scene``) have no third-party deps beyond the
  standard library and can be unit-tested without a DB, GPU, or ML models.
- ``config`` / ``db`` pull in pydantic-settings / psycopg (core deps).
- Heavy ML and ingestion code (``embed``, ``ingest``) imports its stack lazily so the API
  query path and the test suite never load torch unnecessarily.

KITS (the ASR front-end) is consumed strictly as a subprocess via ``kanomori.kits_client`` —
it is never imported, keeping its GPU/torch stack out of this process.
"""

__version__ = "0.1.0"
