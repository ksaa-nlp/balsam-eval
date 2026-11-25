"""Pooled async HTTP for model calls: bounded concurrency, jittered retries, no redirects.

A non-2xx reply is returned, not raised, exactly like the spec transport, so the
caller decides what a status means. Only an unreachable model raises.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from decisioneval.egress import GuardedResolver, check_url
from decisioneval.spec.transport import RedirectRefused, Reply, TransportError, _parse

log = logging.getLogger("decisioneval.transport")

RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
MAX_RETRY_AFTER_S = 60.0


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 3
    backoff_s: float = 1.0
    max_backoff_s: float = 30.0
    on_status: frozenset[int] = field(default=RETRYABLE_STATUS_CODES)

    def delay(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, MAX_RETRY_AFTER_S)
        return random.uniform(0, min(self.max_backoff_s, self.backoff_s * (2 ** (attempt - 1))))


class ModelClient:
    """One pooled session per job, shared by every concurrent item.

    Only public addresses are reachable unless `allow_private`, which exists for
    local runs and tests and is never set in a build.
    """

    def __init__(
        self,
        *,
        concurrency: int = 4,
        timeout_s: float = 60.0,
        retry: RetryPolicy | None = None,
        resolver: aiohttp.abc.AbstractResolver | None = None,
        allow_private: bool = False,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.concurrency = max(1, concurrency)
        self.timeout_s = timeout_s
        self.retry = retry or RetryPolicy()
        self._sleep = sleep
        self._allow_private = allow_private
        self._resolver = None if allow_private else (resolver or GuardedResolver())
        self._sem = asyncio.Semaphore(self.concurrency)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> ModelClient:
        connector = aiohttp.TCPConnector(
            limit=self.concurrency, limit_per_host=self.concurrency, resolver=self._resolver
        )
        timeout = aiohttp.ClientTimeout(total=self.timeout_s, connect=min(10.0, self.timeout_s))
        self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def send(
        self, url: str, body: Any, headers: Mapping[str, str], method: str = "POST"
    ) -> Reply:
        """Send `body` as JSON. Retries retryable statuses and connection errors."""
        if self._session is None:
            raise RuntimeError("ModelClient must be used inside 'async with'")
        if not self._allow_private:
            check_url(url)
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        merged = {"Content-Type": "application/json", **headers}
        reply: Reply | None = None
        for attempt in range(1, self.retry.attempts + 1):
            retry_after = None
            try:
                reply = await self._once(method, url, data, merged)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt == self.retry.attempts:
                    raise TransportError(f"could not reach {url} after {attempt} attempt(s): {exc}") from exc
                log.warning("transport error on attempt %d: %s", attempt, exc)
            else:
                if reply.status not in self.retry.on_status or attempt == self.retry.attempts:
                    return reply
                retry_after = _seconds(reply.headers.get("retry-after"))
            await self._sleep(self.retry.delay(attempt, retry_after))
        assert reply is not None
        return reply

    async def _once(self, method: str, url: str, data: bytes | None, headers: Mapping[str, str]) -> Reply:
        assert self._session is not None
        async with self._sem:
            started = time.perf_counter()
            async with self._session.request(
                method, url, data=data, headers=headers, allow_redirects=False
            ) as response:
                raw = await response.read()
                latency_ms = (time.perf_counter() - started) * 1000.0
                # A redirect would carry the key to wherever Location points.
                if 300 <= response.status < 400:
                    raise RedirectRefused(
                        f"refused a {response.status} redirect from {url}; a model url must answer directly"
                    )
                return Reply(
                    status=response.status,
                    payload=_parse(raw),
                    latency_ms=latency_ms,
                    headers={key.lower(): value for key, value in response.headers.items()},
                )


def _seconds(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if seconds >= 0 else None
