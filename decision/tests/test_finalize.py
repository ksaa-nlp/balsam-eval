"""Finalize retries what may pass on a retry, stops on what never will, and posts once per outcome."""

import io
import json
import urllib.error

import pytest

from decisioneval.remote.finalize import MAX_ERROR_CHARS, FinalizeError, finalize


class Opener:
    def __init__(self, *outcomes):
        self.outcomes, self.requests = list(outcomes), []

    def __call__(self, request, timeout):
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return io.BytesIO(b"{}")


def http_error(code):
    return urllib.error.HTTPError("https://api/x", code, "status", {}, io.BytesIO(b"nope"))


def post(opener, sleeps, **kwargs):
    finalize("https://api/", "t", "9", opener=opener, sleep=sleeps.append, **kwargs)


def test_a_5xx_is_retried_with_a_growing_wait_until_it_succeeds():
    opener, sleeps = Opener(http_error(503), http_error(502), None), []
    post(opener, sleeps, succeeded=True)
    assert len(opener.requests) == 3 and sleeps == [2.0, 4.0]
    assert opener.requests[0].full_url == "https://api/evaluation-jobs/9/finalize"


def test_a_4xx_is_not_retried():
    opener, sleeps = Opener(http_error(400)), []
    with pytest.raises(FinalizeError, match="HTTP 400: nope"):
        post(opener, sleeps, succeeded=True)
    assert len(opener.requests) == 1 and sleeps == []


def test_network_errors_give_up_after_every_attempt():
    opener, sleeps = Opener(*[urllib.error.URLError("refused")] * 4), []
    with pytest.raises(FinalizeError, match="refused"):
        post(opener, sleeps, succeeded=False, error="boom")
    assert len(opener.requests) == 4


def test_the_payload_is_cut_and_keyed_by_job_and_outcome():
    first, second, sleeps = Opener(None), Opener(None), []
    post(first, sleeps, succeeded=False, error="x" * (MAX_ERROR_CHARS + 50), code="C")
    post(second, sleeps, succeeded=False, error="other")
    body = json.loads(first.requests[0].data)
    assert body == {"outcome": "failed", "error": "x" * MAX_ERROR_CHARS, "code": "C"}
    assert first.requests[0].headers["Idempotency-key"] == second.requests[0].headers["Idempotency-key"]
