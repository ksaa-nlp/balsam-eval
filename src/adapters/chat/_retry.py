"""Shared retry classification for provider API calls."""

import errno
from typing import Any

import requests


_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
_PROVIDER_RETRYABLE_STATUS_CODES = {
    "anthropic": {529},
    "azure": {409},
    "openai": {409},
}
_RETRYABLE_ERRNOS = {
    errno.ECONNABORTED,
    errno.ECONNREFUSED,
    errno.ECONNRESET,
    errno.EHOSTUNREACH,
    errno.ENETDOWN,
    errno.ENETUNREACH,
    errno.EPIPE,
    errno.ETIMEDOUT,
}


def _status_code(exc: BaseException) -> int | None:
    """Extract an HTTP status code from common SDK exception shapes."""
    for value in (
        getattr(exc, "status_code", None),
        getattr(exc, "status", None),
        getattr(getattr(exc, "response", None), "status_code", None),
        getattr(exc, "code", None),
    ):
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raw_value: Any = getattr(value, "value", None)
        if isinstance(raw_value, int):
            return raw_value
    return None


def is_retryable_error(exc: BaseException, provider: str | None = None) -> bool:
    """Return whether failure can plausibly succeed without changing request."""
    status = _status_code(exc)
    if status is not None:
        provider_statuses = _PROVIDER_RETRYABLE_STATUS_CODES.get(provider or "", set())
        return status in _RETRYABLE_STATUS_CODES or status in provider_statuses
    if isinstance(
        exc,
        (
            TimeoutError,
            ConnectionError,
            requests.Timeout,
            requests.ConnectionError,
        ),
    ):
        return True
    if isinstance(exc, OSError) and exc.errno in _RETRYABLE_ERRNOS:
        return True
    exception_names = {base.__name__ for base in type(exc).__mro__}
    if exception_names & {
        "APIConnectionError",
        "ConnectError",
        "ConnectionError",
        "ConnectTimeout",
        "NetworkError",
        "ReadTimeout",
        "Timeout",
        "TimeoutException",
    }:
        return True
    cause = exc.__cause__ or exc.__context__
    return (
        cause is not None
        and cause is not exc
        and is_retryable_error(cause, provider=provider)
    )
