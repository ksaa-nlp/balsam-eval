"""The runner's environment contract; see README.md for every variable."""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_DEADLINE_S = 23 * 3600.0  # the build itself is killed at 24h
LOCAL_TASKS_DIR = ".tasks"
LOCAL_RESULTS_DIR = ".results"


class ConfigError(ValueError):
    """The job description is unusable. Nothing was evaluated."""


@dataclass(frozen=True)
class RemoteConfig:
    job_id: str
    evaluation_id: str
    model: str
    model_version: str
    request_config: Mapping[str, Any]
    pool_files: tuple[str, ...]
    bucket: str | None
    results_path: str
    api_host: str | None
    finalize_token: str | None
    deadline_s: float
    local_root: Path

    @property
    def uses_gcs(self) -> bool:
        return self.bucket is not None

    @property
    def reports_outcome(self) -> bool:
        return bool(self.api_host and self.finalize_token and self.job_id)


def load_config(environ: Mapping[str, str], cwd: Path | None = None) -> RemoteConfig:
    """Parse and validate the environment. Raises `ConfigError` on any problem."""
    bucket = _optional(environ, "GCLOUD_BUCKET")
    pool_files = tuple(
        part.strip() for part in (environ.get("POOL_FILES") or "").split(",") if part.strip()
    )
    results_path = _optional(environ, "RESULTS_PATH")

    if bucket is not None:
        missing = [
            name
            for name, value in (
                ("POOL_FILES", pool_files),
                ("RESULTS_PATH", results_path),
                # A cloud job that can't finalize leaves the backend waiting on it forever.
                ("JOB_ID", _optional(environ, "JOB_ID")),
                ("API_HOST", _optional(environ, "API_HOST")),
                ("FINALIZE_TOKEN", _optional(environ, "FINALIZE_TOKEN")),
            )
            if not value
        ]
        if missing:
            raise ConfigError(f"GCLOUD_BUCKET is set, so these are required: {', '.join(missing)}")

    return RemoteConfig(
        job_id=_optional(environ, "JOB_ID") or "local",
        evaluation_id=_optional(environ, "EVALUATION_ID") or "",
        model=_optional(environ, "MODEL") or "decision-model",
        model_version=_optional(environ, "MODEL_VERSION") or "",
        request_config=_request_config(environ),
        pool_files=pool_files,
        bucket=bucket,
        results_path=results_path or LOCAL_RESULTS_DIR,
        api_host=_optional(environ, "API_HOST"),
        finalize_token=_optional(environ, "FINALIZE_TOKEN"),
        deadline_s=_deadline(environ),
        local_root=cwd or Path.cwd(),
    )


def _optional(environ: Mapping[str, str], name: str) -> str | None:
    value = (environ.get(name) or "").strip()
    return value or None


def _request_config(environ: Mapping[str, str]) -> Mapping[str, Any]:
    encoded = _optional(environ, "REQUEST_CONFIG_B64")
    plain = _optional(environ, "REQUEST_CONFIG")
    if encoded is None and plain is None:
        raise ConfigError("REQUEST_CONFIG_B64 (or REQUEST_CONFIG) is required")
    try:
        text = base64.b64decode(encoded, validate=True).decode("utf-8") if encoded else plain
        document = json.loads(text or "")
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise ConfigError(f"REQUEST_CONFIG_B64 is not valid base64 UTF-8: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"the request config is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ConfigError("the request config must be a JSON object")
    return document


def _deadline(environ: Mapping[str, str]) -> float:
    raw = _optional(environ, "RUN_DEADLINE_S")
    if raw is None:
        return DEFAULT_DEADLINE_S
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"RUN_DEADLINE_S must be a number of seconds, got {raw!r}") from exc
    if value <= 0:
        raise ConfigError("RUN_DEADLINE_S must be positive")
    return value
