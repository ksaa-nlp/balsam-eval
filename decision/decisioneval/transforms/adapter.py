"""Adapter blocks: their content hash, and the seeded versions every native protocol binds to.

The hash is computed only here. The backend stores what onboarding reports and the
seed rows this module prints, so no second implementation can drift from it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import rfc8785

from decisioneval.transforms.engine import ENGINE, ENGINE_VERSION
from decisioneval.transforms.seeds import seed_transforms

AUTH_KINDS = ("bearer", "header", "query", "body_field", "none")
AUTH_PLACEMENT_KEYS = ("kind", "name", "prefix", "path")
UNHASHED_KEYS = frozenset({"version_id", "content_hash"})
DEFAULT_RETRY: dict[str, Any] = {
    "attempts": 3,
    "backoff_s": 1.0,
    "timeout_s": 60,
    "concurrency": 4,
    "on_status": [408, 425, 429, 500, 502, 503, 504],
}
# System One keys are optional (decision-systemone may have none); OpenAI Decisions always takes one.
SEEDED: tuple[tuple[str, dict[str, str]], ...] = (
    ("systemone", {"kind": "bearer"}),
    ("systemone", {"kind": "none"}),
    ("openai-decisions", {"kind": "bearer"}),
)


def auth_placement(auth: Mapping[str, Any]) -> dict[str, Any]:
    placement = {key: auth[key] for key in AUTH_PLACEMENT_KEYS if auth.get(key) is not None}
    # Only bearer tells an empty prefix from none; elsewhere both place the key alike, so they hash alike.
    if placement.get("kind") != "bearer" and placement.get("prefix") == "":
        del placement["prefix"]
    return placement


def content_hash(adapter: Mapping[str, Any], auth: Mapping[str, Any]) -> str:
    """sha256 of the RFC 8785 form of the adapter, minus its ids, plus where the key goes."""
    hashed = {key: value for key, value in adapter.items() if key not in UNHASHED_KEYS}
    hashed["auth"] = auth_placement(auth)
    return hashlib.sha256(rfc8785.dumps(hashed)).hexdigest()


def seed_adapter(name: str) -> dict[str, Any]:
    request, response = seed_transforms(name)
    return {
        "interaction": {"pattern": "sync-http-json"},
        "request_transform": request,
        "response_transform": response,
        "retry": dict(DEFAULT_RETRY),
        "engine": ENGINE,
        "engine_version": ENGINE_VERSION,
        "degraded": False,
        "derived_probabilities": False,
    }


def seed_request_config(name: str, url: str, model: str, auth: Mapping[str, Any]) -> dict[str, Any]:
    """An eval REQUEST_CONFIG bound to a seeded version, for local runs."""
    adapter = seed_adapter(name)
    return {
        "url": url,
        "model": model,
        "auth": auth_placement(auth),
        "adapter": {**adapter, "version_id": None, "content_hash": content_hash(adapter, auth)},
    }


def seed_hashes() -> dict[str, str]:
    return {f"{name}/{auth['kind']}": content_hash(seed_adapter(name), auth) for name, auth in SEEDED}


def seed_sql() -> str:
    """INSERT rows for the backend migration; re-running them is a no-op."""
    rows = []
    for name, auth in SEEDED:
        adapter = seed_adapter(name)
        values = [
            "NULL",
            "1",
            _text("seeded"),
            _text(name),
            _json(adapter["interaction"]),
            _text(adapter["request_transform"]),
            _text(adapter["response_transform"]),
            _json(auth_placement(auth)),
            _json(adapter["retry"]),
            "false",
            "false",
            _text(adapter["engine"]),
            _text(adapter["engine_version"]),
            _text(content_hash(adapter, auth)),
        ]
        rows.append(f"  ({', '.join(values)})")
    return (
        'INSERT INTO "adapter_versions" ("shape_id", "version", "origin", "seed_name", "interaction", '
        '"request_transform", "response_transform", "auth_config", "retry_config", "degraded", '
        '"derived_probabilities", "engine", "engine_version", "content_hash") VALUES\n'
        + ",\n".join(rows)
        + '\nON CONFLICT ("content_hash") DO NOTHING;\n'
    )


def _text(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _json(value: Any) -> str:
    return _text(json.dumps(value, ensure_ascii=False, separators=(",", ":"))) + "::jsonb"
