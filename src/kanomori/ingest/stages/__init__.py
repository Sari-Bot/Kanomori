"""Ingestion stages. Each stage is a function ``run(conn, ctx) -> None`` that does its work and
records output on the shared ``IngestContext``. Stages are registered in
``kanomori.ingest.pipeline.STAGES`` in execution order. Heavy deps are imported lazily inside
the stage that needs them so unrelated stages (and tests) don't pay for them.
"""
