-- 0003_lease.sql — job leasing/fencing for multi-worker claim, and nullable content_hash.
--
-- Two changes, both motivated by moving from a single-machine worker to safe concurrent claim:
--
-- 1. Lease/heartbeat/fencing columns on jobs. A worker claims a job by taking a time-bounded
--    lease (lease_expires_at) and stamps its identity (worker_id); a crashed worker's job
--    becomes reclaimable once the lease lapses. lease_epoch is a monotonic fencing token bumped
--    on each (re)claim, so a resurrected zombie worker holding a stale epoch can be rejected on
--    write. heartbeat_at lets a long-running stage extend its lease while making progress.
--
-- 2. content_hash is relaxed to nullable. content_hash is sha256 of the media bytes, computed
--    by the register stage — not known at enqueue time. Previously /ingest invented an md5(path)
--    placeholder, which register then replaced by INSERTing a *second* row keyed by the real
--    sha256, orphaning the enqueued row. The fix: enqueue a job with content_hash = NULL and let
--    register UPDATE that same row to the real hash. The UNIQUE constraint stays — Postgres
--    treats NULLs as distinct, so any number of not-yet-registered jobs may carry NULL while
--    registered jobs remain unique by hash.

ALTER TABLE jobs ADD COLUMN IF NOT EXISTS worker_id        text;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS lease_epoch      int NOT NULL DEFAULT 0;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS lease_expires_at timestamptz;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS heartbeat_at     timestamptz;

ALTER TABLE jobs ALTER COLUMN content_hash DROP NOT NULL;

-- Claim queries scan for eligible jobs by status and lease expiry (queued, or lease lapsed);
-- this composite index keeps that scan off a seq scan as the jobs table grows.
CREATE INDEX IF NOT EXISTS jobs_claimable_idx ON jobs (status, lease_expires_at);
