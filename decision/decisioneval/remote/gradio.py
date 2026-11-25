"""Deliver one request through the adapter's interaction pattern.

`gradio-call` is Gradio's two-step API: POST the inputs to `/gradio_api/call/<name>`
(Gradio 5) or `/call/<name>` (Gradio 4), get an event id, then GET the event
stream until `complete`, whose data is the reply. Both requests go through the
same guarded client, so the egress rules and the timeout hold for each.
"""

from __future__ import annotations

import json
import re
from urllib.parse import urlsplit, urlunsplit
from collections.abc import Mapping
from typing import Any

from decisioneval.spec.transport import Reply
from decisioneval.transport import ModelClient

PREFIXES = ("/gradio_api/call/", "/call/")
CALL_PATH = re.compile(r"/(?:gradio_api/)?call/([^/]+)/?$")


class GradioError(Exception):
    """The Space answered, but not with a usable result. Counted as an adapter error."""


def space_base(url: str) -> str:
    """The Space root, whether the URL was given as the root or as a call endpoint."""
    return CALL_PATH.sub("", url.rstrip("/")) if CALL_PATH.search(url) else url.rstrip("/")


async def deliver(
    client: ModelClient, interaction: Mapping[str, Any], url: str, body: Any, headers: Mapping[str, str]
) -> Reply:
    if interaction.get("pattern") != "gradio-call":
        return await client.send(url, body, headers)
    return await gradio_call(client, url, str(interaction.get("api_name") or "/predict"), body, headers)


async def gradio_call(client: ModelClient, url: str, api_name: str, body: Any, headers: Mapping[str, str]) -> Reply:
    # A query-string key belongs on every request, not inside the Space root.
    parts = urlsplit(url)
    base = space_base(urlunsplit(parts._replace(query="", fragment="")))
    name, query = api_name.strip("/"), f"?{parts.query}" if parts.query else ""
    for prefix in PREFIXES:
        started = await client.send(f"{base}{prefix}{name}{query}", body, headers)
        if started.status != 404:
            break
    if not 200 <= started.status < 300:
        return started
    event_id = started.payload.get("event_id") if isinstance(started.payload, dict) else None
    if not isinstance(event_id, str) or not event_id:
        raise GradioError(f"the Space gave no event id: {json.dumps(started.payload, ensure_ascii=False)}")
    stream = await client.send(f"{base}{prefix}{name}/{event_id}{query}", None, headers, method="GET")
    if not 200 <= stream.status < 300:
        return stream
    text = stream.payload.get("_text") if isinstance(stream.payload, dict) else None
    for event, data in parse_events(text or ""):
        if event == "complete":
            try:
                payload = json.loads(data)
            except json.JSONDecodeError as exc:
                raise GradioError(f"the complete event is not JSON: {data}") from exc
            return Reply(status=200, payload=payload, latency_ms=started.latency_ms + stream.latency_ms, headers=stream.headers)
        if event == "error":
            raise GradioError(f"the Space reported an error: {data}")
    raise GradioError("the event stream ended without a complete event")


def parse_events(text: str) -> list[tuple[str, str]]:
    """Server-sent events as (event, data); heartbeats and comments are dropped."""
    events: list[tuple[str, str]] = []
    event, data = "message", []
    for line in [*text.splitlines(), ""]:
        if not line:
            if data:
                events.append((event, "\n".join(data)))
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    return [(name, payload) for name, payload in events if name != "heartbeat"]
