"""Score every question through the bound adapter version, in one straight async loop.

Per question: request transform, key placement, guarded send, response
transform, decision contract, scoring. No discovery, probe or LLM runs here.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

from decisioneval.evaluator import evaluate_question
from decisioneval.models import DecisionResult, ItemResult, Question
from decisioneval.remote.bound_adapter import BoundAdapter, place_key
from decisioneval.remote.gradio import GradioError, deliver
from decisioneval.remote.contract import ADAPTER_ERROR, MISSING, Checked, check
from decisioneval.spec.transport import TransportError
from decisioneval.transforms.engine import TransformError
from decisioneval.transforms.seeds import request_input
from decisioneval.transforms.worker import TransformWorker
from decisioneval.transport import ModelClient

BREAKER_LIMIT = 5
# A wrong key, a wrong URL or a lasting rate limit fails every item alike, so the run stops at once.
ABORTING_STATUS = frozenset({401, 403, 404, 408, 425, 429})
MALFORMED_RESPONSE = "malformed_response"


class RunAborted(Exception):
    """The model could not be reached, or kept refusing the request. Nothing is published."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class CircuitBroken(Exception):
    """The adapter kept producing output the contract refuses. Nothing is published."""


async def execute(
    bound: BoundAdapter,
    questions: Sequence[Question],
    *,
    key: str | None,
    client: ModelClient,
    worker: TransformWorker,
    on_item: Callable[[ItemResult], None] = lambda item: None,
) -> list[ItemResult]:
    """Score `questions` in order, handing each item to `on_item` as it finishes.

    Raises `RunAborted` or `CircuitBroken`.
    """
    results: dict[int, ItemResult] = {}
    gate = asyncio.Semaphore(bound.concurrency)
    state = {"streak": 0, "broken": False, "refused": 0}

    async def one(index: int, question: Question) -> None:
        async with gate:
            if state["broken"]:
                return
            item, breaker_error = await score_question(bound, question, key=key, client=client, worker=worker)
        results[index] = item
        on_item(item)
        status = item.raw_response.get("http_status") if isinstance(item.raw_response, dict) else None
        state["refused"] = state["refused"] + 1 if status else 0
        if state["refused"] >= BREAKER_LIMIT:
            # The same refusal again and again is the account or key (a 400 for a bad key or no credit), not the adapter.
            raise RunAborted(f"the model refused {BREAKER_LIMIT} requests in a row, the last with HTTP {status}", status=status)
        state["streak"] = state["streak"] + 1 if breaker_error else 0
        if state["streak"] >= BREAKER_LIMIT:
            state["broken"] = True

    try:
        async with asyncio.TaskGroup() as group:
            for index, question in enumerate(questions):
                group.create_task(one(index, question))
    except ExceptionGroup as failure:
        raise failure.exceptions[0]  # keeps its __cause__, which onboarding reads for the error code
    if state["broken"]:
        raise CircuitBroken(
            f"{BREAKER_LIMIT} consecutive answers broke the decision contract; the adapter needs re-onboarding"
        )
    return [results[index] for index in sorted(results)]


async def score_question(
    bound: BoundAdapter,
    question: Question,
    *,
    key: str | None,
    client: ModelClient,
    worker: TransformWorker,
) -> tuple[ItemResult, bool]:
    """One scored item, and whether it counts toward the circuit breaker."""
    data = request_input(
        model=bound.model,
        instruction=question.instruction,
        input_text=question.input,
        options=question.options,
        option_ids=question.option_ids,
    )
    try:
        body = worker.run(bound.request_transform, data)
        url, headers, body = place_key(bound.auth, key, bound.url, body)
    except TransformError as exc:
        return _scored(question, Checked(invalid_reason=ADAPTER_ERROR, distribution_problem=MISSING),
                       {"error": f"request transform: {exc}"}, bound, None), True

    try:
        reply = await deliver(client, bound.interaction, url, body, headers)
    except GradioError as exc:
        checked = Checked(invalid_reason=ADAPTER_ERROR, distribution_problem=MISSING)
        return _scored(question, checked, {"error": str(exc)}, bound, None), True
    except TransportError as exc:
        raise RunAborted(f"could not reach the model: {exc}") from exc
    if 400 <= reply.status < 600 and reply.status not in ABORTING_STATUS:
        # This item's request was refused (too long, unparseable, or it crashes the model): score it invalid
        # and let the refusal streak catch an outage, instead of failing a job no relaunch can finish.
        checked = Checked(invalid_reason=ADAPTER_ERROR, distribution_problem=MISSING)
        raw = {
            "reply": reply.payload,
            "error": f"request transform: the model refused this request with HTTP {reply.status}",
            "http_status": reply.status,
        }
        return _scored(question, checked, raw, bound, reply.latency_ms), False
    if not 200 <= reply.status < 300:
        raise RunAborted(f"the model answered HTTP {reply.status} after its retries", status=reply.status)

    raw: dict[str, Any] = {"reply": reply.payload}
    if isinstance(reply.payload, dict) and set(reply.payload) == {"_text"}:
        checked = Checked(invalid_reason=MALFORMED_RESPONSE, distribution_problem=MISSING)
        return _scored(question, checked, raw, bound, reply.latency_ms), False
    try:
        mapped = worker.run(bound.response_transform, response_input(reply.payload, question.option_ids, question.options))
    except TransformError as exc:
        checked = Checked(invalid_reason=ADAPTER_ERROR, distribution_problem=MISSING)
        return _scored(question, checked, {**raw, "error": f"response transform: {exc}"}, bound, reply.latency_ms), True
    checked = check(mapped, question.option_ids)
    return _scored(question, checked, {**raw, "mapped": mapped}, bound, reply.latency_ms), checked.breaker_error


def response_input(reply: Any, option_ids: Sequence[str], options: Sequence[str]) -> dict[str, Any]:
    """What a response transform sees: the parsed reply and the question's options. Never the key."""
    return {"reply": reply, "option_ids": list(option_ids), "options": list(options)}


def _scored(
    question: Question, checked: Checked, raw: dict[str, Any], bound: BoundAdapter, latency_ms: float | None
) -> ItemResult:
    reply = raw.get("reply")
    resolved = reply.get("model") if isinstance(reply, dict) and isinstance(reply.get("model"), str) else None
    decision = DecisionResult(
        predicted_option_id=checked.choice,
        probabilities=checked.probabilities,
        raw_response={
            **raw,
            "invalid_reason": checked.invalid_reason,
            "distribution_problem": checked.distribution_problem,
        },
        latency_ms=latency_ms,
        valid=checked.choice is not None,
        model_requested=bound.model or None,
        model_resolved=resolved,
    )
    return dataclasses.replace(
        evaluate_question(question, decision), distribution_problem=checked.distribution_problem
    )
