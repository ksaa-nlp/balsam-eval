"""Backend API client used by the evaluation runner."""

import logging
import time
from enum import Enum
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class JobOutcome(str, Enum):
    """Terminal outcomes accepted by ``POST /evaluation-jobs/:id/finalize``."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


def _request_with_retry(
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    json_data: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
    initial_timeout: float = 30.0,
    max_timeout: float = 120.0,
) -> requests.Response:
    """HTTP request with retry on timeout / connection errors."""
    last_exc: Optional[Exception] = None
    timeout = initial_timeout

    for attempt in range(max_retries):
        try:
            return requests.request(
                method=method,
                url=url,
                headers=headers,
                json=json_data,
                timeout=timeout,
            )
        except (requests.Timeout, requests.ConnectionError) as e:
            last_exc = e
            if attempt < max_retries - 1:
                wait = min(2 ** attempt, 30)
                timeout = min(timeout * 1.5, max_timeout)
                logger.warning(
                    "Request to %s failed (%d/%d); retrying in %ds with timeout=%.1fs",
                    url,
                    attempt + 1,
                    max_retries,
                    wait,
                    timeout,
                )
                time.sleep(wait)

    raise requests.RequestException(
        f"Failed to call {url} after {max_retries} attempts: {last_exc}"
    ) from last_exc


def finalize_job(
    *,
    api_host: str,
    finalize_token: str,
    job_id: str,
    outcome: JobOutcome,
    error: Optional[str] = None,
) -> None:
    """Call ``POST {api_host}/evaluation-jobs/{job_id}/finalize`` on the backend.

    Args:
        api_host: Base URL of the backend (no trailing slash).
        finalize_token: Per-job JWT (scope=``finalize``, ~1-week TTL) issued by
            the backend at job-launch time and passed in via the
            ``FINALIZE_TOKEN`` env var. Sent as ``Authorization: Bearer <token>``.
        job_id: Numeric evaluation-job id.
        outcome: Terminal outcome to report.
        error: Optional error message. Only honoured for ``FAILED``.
    """
    payload: dict[str, Any] = {"outcome": outcome.value}
    if outcome == JobOutcome.FAILED and error:
        payload["error"] = error

    url = f"{api_host.rstrip('/')}/evaluation-jobs/{job_id}/finalize"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {finalize_token}",
    }

    response = _request_with_retry("POST", url, headers=headers, json_data=payload)
    if response.status_code >= 400:
        raise RuntimeError(
            f"Finalize call returned HTTP {response.status_code} for job {job_id}: "
            f"{response.text[:500]}"
        )
    logger.info("Finalize OK: job=%s outcome=%s", job_id, outcome.value)
