"""The reply shape and transport errors shared by the async client and onboarding."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


class TransportError(Exception):
    """The model could not be reached. The run must stop."""


class RedirectRefused(TransportError):
    """The model answered with a redirect, which is never followed."""


@dataclass(frozen=True)
class Reply:
    status: int
    payload: Any
    latency_ms: float
    headers: dict[str, str]


def _parse(raw: bytes) -> Any:
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except ValueError:  # also an integer longer than Python will parse
        return {"_text": text}
