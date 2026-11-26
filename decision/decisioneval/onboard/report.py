"""The onboarding job's input and its one report to the backend. The job never touches the DB."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import rfc8785

from decisioneval.remote.finalize import RETRYABLE_STATUS
from decisioneval.store import scrub_key
from decisioneval.transforms.adapter import auth_placement

STAGES = ("probe", "synthesize")
MAX_DETAIL = 2000


class OnboardConfigError(ValueError):
    """The onboarding job description is unusable."""


class ReportError(Exception):
    """The backend did not accept the report."""


@dataclass(frozen=True)
class OnboardRequest:
    url: str
    model: str
    auth: dict[str, Any]
    redacted_request: dict[str, Any] | None = None
    candidates: tuple[dict[str, Any], ...] = field(default_factory=tuple)


def load_request(encoded: str) -> OnboardRequest:
    try:
        document = json.loads(base64.b64decode(encoded, validate=True).decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OnboardConfigError(f"ONBOARD_REQUEST_B64 is not base64 JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise OnboardConfigError("ONBOARD_REQUEST_B64 must hold an object")
    url, model, auth = document.get("url"), document.get("model"), document.get("auth")
    if not isinstance(url, str) or not url.startswith(("https://", "http://")):
        raise OnboardConfigError("the onboarding url must be http(s)")
    if not isinstance(model, str) or not isinstance(auth, dict):
        raise OnboardConfigError("the onboarding request needs a model and an auth placement")
    redacted = document.get("redacted_request")
    candidates = document.get("candidates") or []
    if redacted is not None and not isinstance(redacted, dict):
        raise OnboardConfigError("redacted_request must be an object")
    if not isinstance(candidates, list) or not all(isinstance(entry, dict) for entry in candidates):
        raise OnboardConfigError("candidates must be a list of objects")
    return OnboardRequest(url, model, auth_placement(auth), redacted, tuple(candidates))


def outcome(
    stage: str,
    result: str,
    *,
    error_code: str | None = None,
    error_detail: str | None = None,
    adapter: dict[str, Any] | None = None,
    shape: dict[str, Any] | None = None,
    validations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "outcome": result,
        "error_code": error_code,
        "error_detail": error_detail,
        "adapter": adapter,
        "shape": shape,
        "validations": validations or [],
        "runner_image_digest": None,
    }


def adapter_report(
    block: Mapping[str, Any], auth: Mapping[str, Any], content_hash: str, *, origin: str,
    seed_name: str | None = None, parent_content_hash: str | None = None,
) -> dict[str, Any]:
    body = {key: value for key, value in block.items() if key not in ("version_id", "content_hash")}
    return {
        "origin": origin,
        "seed_name": seed_name,
        "auth": auth_placement(auth),
        **body,
        "content_hash": content_hash,
        "parent_content_hash": parent_content_hash,
    }


def shape_of(url: str, request_body: Any, reply: Any) -> dict[str, Any]:
    """The (host, path, top-level keys) fingerprint the backend groups versions by."""
    parts = urlsplit(url)
    shape = {
        "host": (parts.hostname or "").lower(),
        "path": parts.path or "/",
        "request_keys": sorted(request_body) if isinstance(request_body, dict) else [],
        "response_keys": sorted(reply) if isinstance(reply, dict) else [],
    }
    return {"fingerprint": hashlib.sha256(rfc8785.dumps(shape)).hexdigest(), **shape}


def redact(document: Any, key: str | None) -> bytes:
    """Serialise for the backend with every copy of the key replaced."""
    return json.dumps(scrub_key(document, key), ensure_ascii=False).encode("utf-8")


def _clipped(report: dict[str, Any]) -> dict[str, Any]:
    detail = report.get("error_detail")
    return {**report, "error_detail": detail[:MAX_DETAIL]} if isinstance(detail, str) else report


def post_report(
    api_host: str,
    token: str,
    run_id: str,
    report: dict[str, Any],
    *,
    key: str | None,
    attempts: int = 4,
    backoff_s: float = 2.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    request = urllib.request.Request(
        f"{api_host.rstrip('/')}/onboarding-runs/{run_id}/report",
        data=redact(_clipped(scrub_key(report, key)), key),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "Idempotency-Key": hashlib.sha256(f"{run_id}:{report['stage']}:{report['outcome']}".encode()).hexdigest(),
        },
    )
    last = "no attempt made"
    for attempt in range(1, attempts + 1):
        try:
            with opener(request, timeout=30):
                return
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:300]}"
            if exc.code not in RETRYABLE_STATUS:
                break
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last = str(exc)
        if attempt < attempts:
            sleep(backoff_s * attempt)
    raise ReportError(f"onboarding report for run {run_id} failed: {last}")
