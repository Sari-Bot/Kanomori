"""HTTP client the distributed worker uses to talk to the coordinator's ``/jobs/*`` router.

The single-machine worker (#12) owned the database directly; the distributed worker (#14) does
the GPU compute off-box and reaches the database only through the coordinator (#13). This module
is the worker side of that wire: a thin wrapper that mirrors each endpoint, always attaches the
shared bearer token, and translates the coordinator's status-code contract into return values
the run loop branches on.

Status-code policy (matches ``kanomori.api.jobs``):

* ``claim``                 — 200 -> the job dict; **204** -> ``None`` (coordinator idle).
* ``heartbeat`` / ``push_stage`` / ``complete`` / ``fail`` — 200 -> ``True``; **409** -> ``False``
  (stale lease epoch: this worker was fenced and must stop). Any *other* 4xx/5xx is a real fault
  (bad token, malformed payload, server error) and is raised, never silently swallowed.

httpx lives in the dev dependency group today (see pyproject ``[dependency-groups].dev``); it is
imported lazily here so importing this module — and the worker package — never hard-requires it.
The token is held privately and kept out of ``repr`` so it can't leak into logs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx


class CoordinatorClient:
    """Bearer-authenticated wrapper over the coordinator job endpoints.

    A live ``httpx.Client`` is built lazily on first construction unless one is injected (tests
    pass a fake recording client). ``base_url`` is the coordinator origin (e.g.
    ``http://localhost:8000``); every method targets a path beneath ``/jobs``.
    """

    def __init__(self, base_url: str, token: str | None, *, client: httpx.Client | None = None):
        self._base_url = base_url.rstrip("/")
        self._token = token
        if client is None:
            import httpx

            client = httpx.Client(timeout=httpx.Timeout(30.0))
        self._client = client

    def __repr__(self) -> str:  # token deliberately omitted — never log the secret.
        return f"CoordinatorClient(base_url={self._base_url!r})"

    # --- internals -----------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        """Bearer header for every request (empty when no token is configured — dev only)."""
        return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    def _url(self, path: str) -> str:
        return f"{self._base_url}/jobs/{path.lstrip('/')}"

    # --- endpoints -----------------------------------------------------------------------

    def claim(self, worker_id: str, lease_seconds: int) -> dict | None:
        """Claim the oldest eligible job. Returns the job dict, or ``None`` when idle (204)."""
        resp = self._client.post(
            self._url("claim"),
            json={"worker_id": worker_id, "lease_seconds": int(lease_seconds)},
            headers=self._headers(),
        )
        if resp.status_code == 204:
            return None
        resp.raise_for_status()
        return resp.json()

    def heartbeat(self, job_id: int, lease_epoch: int, lease_seconds: int) -> bool:
        """Extend the lease. ``True`` on 200; ``False`` on 409 (fenced — stop working)."""
        resp = self._client.post(
            self._url(f"{int(job_id)}/heartbeat"),
            json={"lease_epoch": int(lease_epoch), "lease_seconds": int(lease_seconds)},
            headers=self._headers(),
        )
        if resp.status_code == 409:
            return False
        resp.raise_for_status()
        return True

    def push_stage(
        self,
        job_id: int,
        stage_name: str,
        lease_epoch: int,
        result_json: str,
        files: list[tuple[str, bytes]],
    ) -> bool:
        """Push one stage's result + artifacts (multipart). ``False`` on 409, else raise on error.

        ``result_json`` rides as the optional ``result_file`` JSON upload (omitted for the no-model
        locate_media stage); ``lease_epoch`` as the ``lease_epoch`` form field. Each ``(name,
        bytes)`` in ``files`` is attached under the multipart field name ``files`` (the JPEGs /
        the SRT), named exactly as the stage's ArtifactRef so the coordinator writes them to their
        deterministic on-disk paths.
        """
        multipart = []
        if result_json:
            multipart.append(
                ("result_file", ("result.json", result_json.encode("utf-8"), "application/json"))
            )
        multipart.extend(
            ("files", (name, content, "application/octet-stream")) for name, content in files
        )
        resp = self._client.post(
            self._url(f"{int(job_id)}/stage/{stage_name}"),
            data={"lease_epoch": str(int(lease_epoch))},
            files=multipart,
            headers=self._headers(),
        )
        if resp.status_code == 409:
            return False
        resp.raise_for_status()
        return True

    def complete(self, job_id: int, lease_epoch: int) -> bool:
        """Mark the job complete. ``True`` on 200; ``False`` on 409 (stale epoch)."""
        resp = self._client.post(
            self._url(f"{int(job_id)}/complete"),
            json={"lease_epoch": int(lease_epoch)},
            headers=self._headers(),
        )
        if resp.status_code == 409:
            return False
        resp.raise_for_status()
        return True

    def fail(self, job_id: int, lease_epoch: int, error: str) -> bool:
        """Mark the job failed with ``error``. ``True`` on 200; ``False`` on 409 (stale epoch)."""
        resp = self._client.post(
            self._url(f"{int(job_id)}/fail"),
            json={"lease_epoch": int(lease_epoch), "error": error},
            headers=self._headers(),
        )
        if resp.status_code == 409:
            return False
        resp.raise_for_status()
        return True
