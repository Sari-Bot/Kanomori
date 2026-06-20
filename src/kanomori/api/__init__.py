"""FastAPI application: the online query API (CPU-only) plus ingest enqueue/status.

The query path is CPU-only by design — the only model touched online is the (lazily-loaded)
text embedder, constructed once and shared. Ingestion itself runs in the separate worker
process; the API only enqueues a job and reports its status.
"""
