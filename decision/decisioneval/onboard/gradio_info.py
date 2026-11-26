"""Onboarding helpers for Gradio Spaces: recognise one, wake it, and read its endpoint info."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from decisioneval.remote.bound_adapter import place_key
from decisioneval.remote.gradio import CALL_PATH, space_base
from decisioneval.transport import ModelClient

WAKE_DEADLINE_S = 180.0
WAKE_POLL_S = 10.0


def _space_target(
    url: str, auth: Mapping[str, Any] | None, key: str | None
) -> tuple[str, str, dict[str, str]]:
    """Return `(base, query, headers)` with any non-body credential placed for a Space GET."""
    headers: dict[str, str] = {}
    if auth and auth.get("kind") != "body_field":
        url, headers, _ = place_key(auth, key, url, None)
    parts = urlsplit(url)
    base = space_base(urlunsplit(parts._replace(query="", fragment="")))
    query = f"?{parts.query}" if parts.query else ""
    return base, query, headers


def gradio_interaction(url: str, redacted: dict[str, Any] | None) -> dict[str, str] | None:
    """The gradio-call interaction when the URL or the pasted curl points at a Space, else None."""
    for candidate in (str((redacted or {}).get("url") or ""), url):
        match = CALL_PATH.search(candidate)
        if match:
            return {"pattern": "gradio-call", "api_name": f"/{match.group(1)}"}
    if (urlsplit(url).hostname or "").endswith(".hf.space"):
        return {"pattern": "gradio-call", "api_name": "/predict"}
    return None


async def wake(
    client: ModelClient,
    url: str,
    *,
    auth: Mapping[str, Any] | None = None,
    key: str | None = None,
    deadline_s: float = WAKE_DEADLINE_S,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> bool:
    """Poll the Space root until it stops answering 'building' or 503. True once it is up."""
    base, query, headers = _space_target(url, auth, key)
    started = clock()
    while True:
        reply = await client.send(f"{base}{query}", None, headers, method="GET")
        text = json.dumps(reply.payload, ensure_ascii=False).lower()
        # A running app embeds its config; only a status page is read for these words, not a Space's own title.
        status_page = "gradio_config" not in text and ("building" in text or "sleeping" in text)
        if reply.status < 500 and not status_page:
            return True
        if clock() - started >= deadline_s:
            return False
        await sleep(WAKE_POLL_S)


async def endpoint_info(
    client: ModelClient,
    url: str,
    *,
    auth: Mapping[str, Any] | None = None,
    key: str | None = None,
) -> Any:
    """The Space's named endpoints and their parameters, or None when it publishes none."""
    base, query, headers = _space_target(url, auth, key)
    for path in ("/gradio_api/info", "/info"):
        reply = await client.send(f"{base}{path}{query}", None, headers, method="GET")
        if 200 <= reply.status < 300 and isinstance(reply.payload, dict) and "_text" not in reply.payload:
            return reply.payload.get("named_endpoints", reply.payload)
    return None
