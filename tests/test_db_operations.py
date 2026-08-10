import hashlib
from unittest.mock import Mock

import pytest
import requests

import src.db_operations as db
from src.db_operations import JobOutcome, _request_with_retry, finalize_job


def response(status_code=200, text="ok"):
    result = Mock(spec=requests.Response)
    result.status_code = status_code
    result.text = text
    return result


def test_request_returns_non_retryable_response_immediately(monkeypatch):
    request = Mock(return_value=response(500))
    sleep = Mock()
    monkeypatch.setattr(db.requests, "request", request)
    monkeypatch.setattr(db.time, "sleep", sleep)

    result = _request_with_retry("GET", "https://example")

    assert result.status_code == 500
    request.assert_called_once()
    sleep.assert_not_called()


def test_request_retries_transient_status_with_backoff_and_timeout(monkeypatch):
    request = Mock(side_effect=[response(503), response(429), response(200)])
    sleep = Mock()
    monkeypatch.setattr(db.requests, "request", request)
    monkeypatch.setattr(db.time, "sleep", sleep)

    result = _request_with_retry(
        "POST",
        "https://example",
        max_retries=3,
        initial_timeout=10,
        max_timeout=20,
    )

    assert result.status_code == 200
    assert [call.kwargs["timeout"] for call in request.call_args_list] == [10, 15, 20]
    assert [call.args[0] for call in sleep.call_args_list] == [1, 2]


def test_request_retries_transport_errors_then_raises(monkeypatch):
    request = Mock(side_effect=requests.Timeout("slow"))
    monkeypatch.setattr(db.requests, "request", request)
    monkeypatch.setattr(db.time, "sleep", Mock())

    with pytest.raises(requests.RequestException, match="after 2 attempts") as exc_info:
        _request_with_retry("GET", "https://example", max_retries=2)

    assert isinstance(exc_info.value.__cause__, requests.Timeout)


def test_finalize_job_sends_expected_url_headers_and_success_payload(monkeypatch):
    request = Mock(return_value=response())
    monkeypatch.setattr(db, "_request_with_retry", request)

    finalize_job(
        api_host="https://backend.example/",
        finalize_token="secret",
        job_id="42",
        outcome=JobOutcome.SUCCEEDED,
        error="ignored",
    )

    expected_key = hashlib.sha256(b"42:succeeded").hexdigest()
    request.assert_called_once_with(
        "POST",
        "https://backend.example/evaluation-jobs/42/finalize",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer secret",
            "Idempotency-Key": expected_key,
        },
        json_data={"outcome": "succeeded"},
    )


def test_finalize_job_includes_failed_error(monkeypatch):
    request = Mock(return_value=response())
    monkeypatch.setattr(db, "_request_with_retry", request)

    finalize_job(
        api_host="https://backend.example",
        finalize_token="secret",
        job_id="42",
        outcome=JobOutcome.FAILED,
        error="evaluation failed",
    )

    assert request.call_args.kwargs["json_data"] == {
        "outcome": "failed",
        "error": "evaluation failed",
    }


def test_finalize_job_raises_with_truncated_response_body(monkeypatch):
    monkeypatch.setattr(db, "_request_with_retry", Mock(return_value=response(400, "x" * 600)))

    with pytest.raises(RuntimeError) as exc_info:
        finalize_job(
            api_host="https://backend.example",
            finalize_token="secret",
            job_id="42",
            outcome=JobOutcome.FAILED,
        )

    assert "HTTP 400" in str(exc_info.value)
    assert "x" * 500 in str(exc_info.value)
    assert "x" * 501 not in str(exc_info.value)
