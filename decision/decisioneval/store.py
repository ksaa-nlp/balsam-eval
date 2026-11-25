"""Write and read one JSON file per evaluation run.

Ported from the origin repo's `results/store.py`. What was kept is what the
invariants need: a per-file and combined dataset digest so a run says which
bytes it scored, the full per-item record so every number is re-derivable, and
credential redaction so no secret can reach a run file. What was dropped is the
BALSAM release manifest pinning, which belongs to that one dataset release
rather than to a general platform.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, quote_plus
from typing import Any
from uuid import uuid4

from decisioneval.models import ItemResult, LoadReport

SCHEMA_VERSION = 1
ACCESS_TYPES = frozenset({"api", "local"})
RUN_TYPES = frozenset({"probe", "pilot", "full"})

_SECRET_KEYS = frozenset({
    "api_key", "apikey", "authorization", "access_token", "refresh_token",
    "id_token", "password", "passwd", "private_key", "client_secret",
    "secret_key", "credentials", "credential", "bearer", "x_api_key",
    "openai_api_key", "groq_api_key", "anthropic_api_key", "gemini_api_key",
    "token", "auth", "cookie", "set_cookie",
})
_CREDENTIAL_VALUE = re.compile(
    r"^(?:Bearer\s+)?(?:sk-|sk-proj-|sk-ant-|gsk_|AIza|ghp_|gho_|xoxb-)[A-Za-z0-9_\-]{8,}$"
)


def new_run_id() -> str:
    return uuid4().hex[:12]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_run(
    *,
    run_id: str,
    run_type: str,
    model: str,
    model_version: str,
    access_type: str,
    spec_name: str,
    report: LoadReport,
    summary: dict[str, Any],
    items: list[ItemResult],
    timestamp: str | None = None,
    submission_id: str | None = None,
    mapping_origin: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Assemble the stored run document. Secrets are removed at write time.

    `mapping_origin` records how this model's mapping was obtained -- reused,
    probe, or llm -- so a reader can tell an inferred mapping from a shared one.
    """
    if run_type not in RUN_TYPES:
        raise ValueError(f"run_type must be one of {sorted(RUN_TYPES)}, got {run_type!r}")
    if access_type not in ACCESS_TYPES:
        raise ValueError(f"access_type must be api or local, got {access_type!r}")
    validation = _validation_report(report)
    datasets = [dataset.name for dataset in report.datasets]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "run_type": run_type,
        "submission_id": submission_id,
        "model": model,
        "model_version": model_version,
        "access_type": access_type,
        "spec": spec_name,
        "mapping_origin": mapping_origin,
        "notes": notes,
        "timestamp": timestamp or utc_timestamp(),
        "datasets": datasets,
        "counts": {
            "datasets": len(datasets),
            "source_test_items": validation["total_items"],
            "evaluated_items": validation["valid_items"],
            "excluded_items": validation["excluded_items"],
        },
        "fingerprint": _fingerprint(report),
        "files": {
            "used": len(report.datasets),
            "skipped": len(report.skipped),
            "used_files": [
                {"path": dataset.source_path, "name": dataset.name}
                for dataset in report.datasets
            ],
            "skipped_files": [
                {"path": skipped.path, "reason": skipped.reason} for skipped in report.skipped
            ],
        },
        "validation": validation,
        "summary": summary,
        "items": [item.to_dict() for item in items],
    }


def _validation_report(report: LoadReport) -> dict[str, Any]:
    excluded = [
        {"dataset": dataset.name, **item.to_dict()}
        for dataset in report.datasets
        for item in dataset.excluded
    ]
    valid_items = sum(len(dataset.questions) for dataset in report.datasets)
    return {
        "total_items": valid_items + len(excluded),
        "valid_items": valid_items,
        "excluded_items": len(excluded),
        "excluded": excluded,
    }


def _fingerprint(report: LoadReport) -> dict[str, Any]:
    files = [_file_fingerprint(dataset.source_path) for dataset in report.datasets]
    return {
        "data": {
            "algorithm": "sha256",
            "digest": combined_data_digest([entry["sha256"] for entry in files]),
            "files": files,
        }
    }


def _file_fingerprint(source: str) -> dict[str, Any]:
    path = Path(source)
    payload = path.read_bytes()
    return {
        "name": path.name,
        "path": source,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def combined_data_digest(file_hashes: list[str]) -> str:
    """Hash of the sorted per-file digests, so path order does not change it."""
    return hashlib.sha256("\n".join(sorted(file_hashes)).encode("utf-8")).hexdigest()


def read_run(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def redact_secrets(value: Any) -> Any:
    """Return a copy with credential fields replaced by `[redacted]`."""
    if isinstance(value, dict):
        cleaned: dict[Any, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and is_secret_key(key):
                cleaned[key] = "[redacted]"
            else:
                cleaned[key] = redact_secrets(item)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item) for item in value]
    if isinstance(value, str) and _CREDENTIAL_VALUE.fullmatch(value.strip()):
        return "[redacted]"
    return value


KEY_PLACEHOLDER = "<API_KEY>"


MIN_SCRUBBED_KEY = 8  # shorter "keys" are placeholders such as vLLM's EMPTY; scrubbing them would rewrite data


def scrub_key(value: Any, key: str | None) -> Any:
    """A copy with every copy of `key` replaced: raw, JSON-escaped, or URL-encoded in any way a client sends it.

    Name- and pattern-based redaction can't see a key echoed under a plain field
    name or carried in a URL's query string; this removes the actual value.
    """
    if not key or len(key) < MIN_SCRUBBED_KEY:
        return value
    return _scrub(value, _key_forms(key))


def clip(text: str, limit: int, key: str | None) -> str:
    """Scrub, then cut: cutting first could leave half a key the scrub no longer recognises."""
    return scrub_key(text, key)[:limit]


class KeyScrubFilter(logging.Filter):
    """Removes the key from every log line, tracebacks included."""

    def __init__(self, key: str | None):
        super().__init__()
        self.key = key

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg, record.args = scrub_key(record.getMessage(), self.key), ()
        if record.exc_info:
            # The formatter renders exc_info after filters run, so render and scrub it here instead.
            record.exc_text = scrub_key(logging.Formatter().formatException(record.exc_info), self.key)
            record.exc_info = None
        elif record.exc_text:
            record.exc_text = scrub_key(record.exc_text, self.key)
        return True


def scrub_logs(key: str | None) -> None:
    for handler in logging.getLogger().handlers:
        handler.addFilter(KeyScrubFilter(key))


def _key_forms(key: str) -> list[str]:
    encoded = {quote(key, safe=""), quote(key, safe="/"), quote_plus(key), quote_plus(key, safe="/")}
    lower_hex = {re.sub(r"%[0-9A-F]{2}", lambda match: match.group(0).lower(), form) for form in encoded}
    return sorted({key, json.dumps(key)[1:-1], *encoded, *lower_hex}, key=len, reverse=True)


def _scrub(value: Any, forms: list[str]) -> Any:
    if isinstance(value, str):
        for form in forms:
            value = value.replace(form, KEY_PLACEHOLDER)
        return value
    if isinstance(value, dict):
        return {_scrub(name, forms): _scrub(item, forms) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(item, forms) for item in value]
    return value


def is_secret_key(key: str) -> bool:
    name = key.strip().lower().replace("-", "_").replace(" ", "_")
    if name in _SECRET_KEYS:
        return True
    return name.endswith(("_api_key", "_api_token", "_secret", "_token", "_password"))


def secret_paths(value: Any, prefix: str = "$") -> list[str]:
    """Locations that still hold a credential value."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}"
            if isinstance(key, str) and is_secret_key(key):
                if item not in ("[redacted]", None):
                    found.append(path)
            else:
                found.extend(secret_paths(item, path))
        return found
    if isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(secret_paths(item, f"{prefix}[{index}]"))
        return found
    if isinstance(value, str) and _CREDENTIAL_VALUE.fullmatch(value.strip()):
        found.append(prefix)
    return found
