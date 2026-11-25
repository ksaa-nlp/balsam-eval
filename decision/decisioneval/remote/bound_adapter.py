"""The Phase 1 request config: a URL, a model, where the key goes, and the bound adapter version.

The runner refuses a version whose hash it can't reproduce or that was validated
on another engine, and inserts the key only after the request transform ran, so
no transform ever sees it.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from decisioneval.remote.config import ConfigError
from decisioneval.transforms.adapter import AUTH_KINDS, DEFAULT_RETRY, content_hash
from decisioneval.transforms.seeds import SEED_NAMES, seed_transforms
from decisioneval.transforms.engine import ENGINE, ENGINE_VERSION, TransformError, compile_expression
from decisioneval.transport import RetryPolicy

API_KEY_ENV = "API_KEY"
CONFIG_KEYS = frozenset({"url", "model", "auth", "adapter"})
AUTH_KEYS = frozenset({"kind", "name", "prefix", "path"})
ADAPTER_KEYS = frozenset({
    "version_id", "content_hash", "interaction", "request_transform", "response_transform",
    "retry", "engine", "engine_version", "degraded", "derived_probabilities",
})
RETRY_KEYS = frozenset({"attempts", "backoff_s", "timeout_s", "concurrency", "on_status"})
PATTERNS = frozenset({"sync-http-json", "gradio-call"})
MAX_TIMEOUT_S = 600.0
MAX_CONCURRENCY = 32
MAX_ATTEMPTS = 10


class AdapterConfigError(ConfigError):
    """The bound adapter can't be run as given."""


@dataclass(frozen=True)
class BoundAdapter:
    url: str
    model: str
    auth: Mapping[str, Any]
    version_id: int | None
    content_hash: str
    interaction: Mapping[str, Any]
    request_transform: str
    response_transform: str
    retry: RetryPolicy
    timeout_s: float
    concurrency: int
    degraded: bool
    derived_probabilities: bool


def load_bound_adapter(config: Mapping[str, Any]) -> BoundAdapter:
    """Validate the config, verify the adapter's hash and lint its transforms."""
    _only(config, CONFIG_KEYS, "request config")
    url = config.get("url")
    if not isinstance(url, str) or not url.startswith(("https://", "http://")):
        raise AdapterConfigError("request config url must be an http(s) URL")
    model = config.get("model")
    if not isinstance(model, str):
        raise AdapterConfigError("request config model must be a string")
    auth = _auth(config.get("auth"))
    adapter = config.get("adapter")
    if not isinstance(adapter, Mapping):
        raise AdapterConfigError("request config adapter must be an object")
    _only(adapter, ADAPTER_KEYS, "adapter")

    if (adapter.get("engine"), adapter.get("engine_version")) != (ENGINE, ENGINE_VERSION):
        raise AdapterConfigError(
            f"adapter was validated on {adapter.get('engine')} {adapter.get('engine_version')}, "
            f"but this runner has {ENGINE} {ENGINE_VERSION}"
        )
    expected = content_hash(adapter, auth)
    if adapter.get("content_hash") != expected:
        raise AdapterConfigError("adapter content_hash does not match its content; refusing to run it")

    interaction = adapter.get("interaction")
    if not isinstance(interaction, Mapping) or interaction.get("pattern") not in PATTERNS:
        raise AdapterConfigError(f"adapter interaction.pattern must be one of {', '.join(sorted(PATTERNS))}")
    if interaction.get("pattern") == "gradio-call" and not str(interaction.get("api_name") or "").startswith("/"):
        raise AdapterConfigError("a gradio-call interaction needs an api_name such as /predict")
    for side in ("request_transform", "response_transform"):
        try:
            compile_expression(adapter.get(side))
        except TransformError as exc:
            raise AdapterConfigError(f"adapter {side} is refused: {exc}") from exc

    retry = adapter.get("retry")
    if not isinstance(retry, Mapping):
        raise AdapterConfigError("adapter retry must be an object")
    _only(retry, RETRY_KEYS, "adapter retry")
    version_id = adapter.get("version_id")
    return BoundAdapter(
        url=url,
        model=model,
        auth=auth,
        version_id=version_id if isinstance(version_id, int) and not isinstance(version_id, bool) else None,
        content_hash=expected,
        interaction=dict(interaction),
        request_transform=adapter["request_transform"],
        response_transform=adapter["response_transform"],
        retry=_policy(retry),
        timeout_s=_number(retry, "timeout_s", 60, 0.0, MAX_TIMEOUT_S, exclusive_low=True),
        concurrency=int(_number(retry, "concurrency", 4, 1, MAX_CONCURRENCY)),
        degraded=adapter.get("degraded") is True,
        derived_probabilities=adapter.get("derived_probabilities") is True,
    )


def mapping_origin(bound: BoundAdapter) -> str:
    """native when it speaks a seeded protocol, wherever the key goes; onboarded when synthesised."""
    if bound.degraded:
        return "degraded"
    transforms = (bound.request_transform, bound.response_transform)
    return "native" if any(transforms == seed_transforms(name) for name in SEED_NAMES) else "onboarded"


def require_key(auth: Mapping[str, Any], environ: Mapping[str, str]) -> str | None:
    if auth["kind"] == "none":
        return None
    key = (environ.get(API_KEY_ENV) or "").strip()
    if not key:
        raise AdapterConfigError(f"auth.kind is {auth['kind']!r} but {API_KEY_ENV} is not set")
    return key


def place_key(auth: Mapping[str, Any], key: str | None, url: str, body: Any) -> tuple[str, dict[str, str], Any]:
    """Put the key where `auth` says, on a copy of the request. Returns (url, headers, body)."""
    kind = auth["kind"]
    if kind == "none" or key is None:
        return url, {}, body
    if kind == "bearer":
        prefix = auth.get("prefix")
        return url, {"Authorization": f"{'Bearer ' if prefix is None else prefix}{key}"}, body
    if kind == "header":
        return url, {auth["name"]: f"{auth.get('prefix') or ''}{key}"}, body
    if kind == "query":
        parts = urlsplit(url)
        query = urlencode([*parse_qsl(parts.query, keep_blank_values=True), (auth["name"], key)])
        return urlunsplit(parts._replace(query=query)), {}, body
    if not isinstance(body, Mapping):
        raise TransformError("the request transform must produce an object to carry a body_field key")
    placed = copy.deepcopy(dict(body))
    node = placed
    *parents, leaf = auth["path"].split(".")
    for part in parents:
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise TransformError(f"auth.path {auth['path']!r} crosses a non-object at {part!r}")
        node = child
    node[leaf] = f"{auth.get('prefix') or ''}{key}"
    return url, {}, placed


def _auth(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AdapterConfigError("request config auth must be an object")
    _only(value, AUTH_KEYS, "auth")
    kind = value.get("kind")
    if kind not in AUTH_KINDS:
        raise AdapterConfigError(f"auth.kind must be one of {', '.join(AUTH_KINDS)}")
    for key in ("name", "prefix", "path"):
        if value.get(key) is not None and not isinstance(value[key], str):
            raise AdapterConfigError(f"auth.{key} must be a string")
    if kind in ("header", "query") and not value.get("name"):
        raise AdapterConfigError(f"auth.kind {kind} needs auth.name")
    if kind == "body_field" and not value.get("path"):
        raise AdapterConfigError("auth.kind body_field needs auth.path")
    return {key: item for key, item in value.items() if item is not None}


def _policy(retry: Mapping[str, Any]) -> RetryPolicy:
    on_status = retry.get("on_status", DEFAULT_RETRY["on_status"])
    if not isinstance(on_status, list) or not all(isinstance(code, int) and 100 <= code <= 599 for code in on_status):
        raise AdapterConfigError("adapter retry.on_status must be a list of HTTP status codes")
    return RetryPolicy(
        attempts=int(_number(retry, "attempts", 3, 1, MAX_ATTEMPTS)),
        backoff_s=_number(retry, "backoff_s", 1.0, 0.0, 60.0),
        on_status=frozenset(on_status),
    )


def _number(source: Mapping[str, Any], key: str, default: float, low: float, high: float, exclusive_low: bool = False) -> float:
    value = source.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdapterConfigError(f"adapter retry.{key} must be a number")
    if value > high or value < low or (exclusive_low and value == low):
        raise AdapterConfigError(f"adapter retry.{key} must be within [{low:g}, {high:g}]")
    return float(value)


def _only(value: Mapping[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise AdapterConfigError(f"unsupported {where} keys: {', '.join(unknown)}")
