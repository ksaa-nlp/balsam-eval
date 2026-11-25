"""POST the job outcome to the backend's finalize endpoint."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable

SUCCEEDED = "succeeded"
FAILED = "failed"
MAX_ERROR_CHARS = 4000  # the backend's finalize DTO limit
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class FinalizeError(Exception):
    """The backend did not accept the outcome."""


def finalize(
    api_host: str,
    token: str,
    job_id: str,
    *,
    succeeded: bool,
    error: str | None = None,
    code: str | None = None,
    content_hash: str | None = None,
    attempts: int = 4,
    backoff_s: float = 2.0,
    timeout_s: float = 30.0,
    opener: Callable[..., object] = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    outcome = SUCCEEDED if succeeded else FAILED
    payload: dict[str, str] = {"outcome": outcome}
    if not succeeded and error:
        payload["error"] = error[:MAX_ERROR_CHARS]
    if not succeeded and code:
        payload["code"] = code
    if content_hash:
        payload["adapter_content_hash"] = content_hash
    request = urllib.request.Request(
        f"{api_host.rstrip('/')}/evaluation-jobs/{job_id}/finalize",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "Idempotency-Key": hashlib.sha256(f"{job_id}:{outcome}".encode()).hexdigest(),
        },
    )

    last_error = "no attempt made"
    for attempt in range(1, attempts + 1):
        try:
            with opener(request, timeout=timeout_s):
                return
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:300]}"
            if exc.code not in RETRYABLE_STATUS:
                break
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_error = str(exc)
        if attempt < attempts:
            sleep(backoff_s * attempt)
    raise FinalizeError(f"finalize for job {job_id} ({outcome}) failed: {last_error}")
