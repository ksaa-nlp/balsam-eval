"""Synthesize stage: build an adapter for an API no seeded protocol fits, from the pasted curl.

Order: a seeded protocol with the key where the curl puts it; the LLM for the
request transform; the probe compiled to JSONata for the response transform,
the LLM only when the probe can't settle it; then the canary suite. Repairs
are bounded and see visible canaries only. Held-out items never reach a prompt.
"""

from __future__ import annotations

import json
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from collections.abc import Awaitable, Callable
from typing import Any

from decisioneval.onboard.canary import VISIBLE, ORIGINAL, CanaryItem, judge
from decisioneval.onboard.compile import CompileError, compile_response
from decisioneval.onboard.gradio_info import endpoint_info, gradio_interaction, wake
from decisioneval.onboard.report import OnboardRequest, adapter_report, outcome, shape_of
from decisioneval.onboard.trial import (
    AUTH_REFUSED,
    DELIVERY_UNCONFIRMED,
    FULL_CASES,
    ONBOARDING_FAILED,
    abort_code,
    bind,
    run_trial,
    seed_body,
    seed_verdict,
)
from decisioneval.onboard.llm import LlmError, LlmProvider, extract_json
from decisioneval.remote.bound_adapter import AdapterConfigError, place_key
from decisioneval.remote.execute import RunAborted, response_input
from decisioneval.remote.gradio import GradioError, deliver
from decisioneval.spec.probe import Sample, probe, to_spec
from decisioneval.spec.transport import TransportError
from decisioneval.store import clip, is_secret_key, redact_secrets, scrub_key
from decisioneval.transforms.adapter import DEFAULT_RETRY, seed_adapter
from decisioneval.transforms.engine import ENGINE, ENGINE_VERSION, TransformError, compile_expression
from decisioneval.transforms.seeds import SEED_NAMES, request_input
from decisioneval.transforms.worker import TransformWorker
from decisioneval.transport import ModelClient

STAGE = "synthesize"
REPAIR_ROUNDS = 3
REQUEST_SIDE = "request transform:"  # how execute labels a request-side adapter error
SAMPLE_ITEMS = tuple(VISIBLE)
RULES = """JSONata rules (anything else is refused before it runs):
- no $eval, $now, $millis, $random or $match, and no regex literals;
- no partial application (`?`), and never bind a function to a name with :=;
- a function literal may only be the second argument of $map, $filter, $reduce, $sort or $each (or their only argument after ~>);
- never redefine a built-in function's name, as a variable or as a parameter;
- each expression is at most 4096 bytes."""
CONTRACTS = """Request transform input: {"model", "instruction", "input", "state", "options": [...], "option_ids": [...], "criteria": {id: text}}.
Its output is the JSON body to send. Never put a key or token in it: the platform inserts the key after the transform runs.
Response transform input: {"reply": <the parsed reply>, "option_ids": [...], "options": [...]}.
Its output is exactly {"status": "answer" | "refusal", "choice": <one of option_ids, or null>, "probabilities": {<option id>: <number>} or null}.
Copy probabilities exactly as the reply gives them: never compute, rescale, renormalise, softmax or fill them. If the reply has no per-option probabilities, use null."""


class SynthesisFailed(Exception):
    """No transform survived the bounded attempts."""


async def synthesize_stage(
    request: OnboardRequest,
    *,
    key: str | None,
    client: ModelClient,
    worker: TransformWorker,
    llm: LlmProvider | None,
    progress: Callable[[dict[str, Any]], Awaitable[None] | None] = lambda report: None,
) -> dict[str, Any]:
    auth = request.auth
    try:
        seeded = await _seeded(request, key=key, client=client, worker=worker)
    except RunAborted as exc:
        return _aborted(exc)
    if seeded is not None:
        return seeded
    if llm is None:
        return outcome(STAGE, "failed", error_code=ONBOARDING_FAILED,
                       error_detail="no seeded protocol fits this API and no mapping LLM is configured")
    interaction = gradio_interaction(request.url, request.redacted_request) or {"pattern": "sync-http-json"}
    space_info = None
    if interaction["pattern"] == "gradio-call":
        try:
            if not await wake(client, request.url):
                return outcome(STAGE, "failed", error_code=ONBOARDING_FAILED,
                               error_detail="the Space did not wake within 3 minutes")
            space_info = await endpoint_info(client, request.url)
        except TransportError as exc:
            return _aborted(RunAborted(f"could not reach the Space: {exc}"))

    feedback: list[str] = []  # why the last attempt failed, for the next LLM round
    request_failed = False
    request_transform: str | None = None
    samples: list[tuple[CanaryItem, Any, Any]] = []
    reported_progress = False
    for round_index in range(REPAIR_ROUNDS + 1):
        try:
            if request_transform is None:
                request_transform = _ask(llm, _prompt(request, feedback, samples=None, space=space_info, key=key), "request_transform", key)
                samples = []
            if not samples:  # the replies only change when the request transform does
                samples = await _sample(request, request_transform, interaction, key=key, client=client, worker=worker)
        except _Retry as retry:
            feedback, request_failed = [str(retry)], True
            request_transform = None
            continue
        except RunAborted as exc:
            return _aborted(exc)
        except SynthesisFailed as exc:
            return outcome(STAGE, "failed", error_code=ONBOARDING_FAILED, error_detail=str(exc))

        try:
            # A request-side failure says nothing about the replies, so the probe still gets the first try.
            response_feedback = [] if request_failed else feedback
            request_failed = False
            response_transform, origin = _response_transform(llm, request, samples, response_feedback, space_info, key)
        except _Retry as retry:
            feedback = [str(retry)]
            continue
        except SynthesisFailed as exc:
            return outcome(STAGE, "failed", error_code=ONBOARDING_FAILED, error_detail=str(exc))

        degraded, derived = _probability_flags(response_transform, samples, worker)
        adapter = {
            "interaction": interaction,
            "request_transform": request_transform,
            "response_transform": response_transform,
            "retry": dict(DEFAULT_RETRY),
            "engine": ENGINE,
            "engine_version": ENGINE_VERSION,
            "degraded": degraded,
            "derived_probabilities": derived,
        }
        try:
            bound = bind(request.url, request.model, auth, adapter)
        except AdapterConfigError as exc:
            feedback = [f"the transforms were refused: {exc}"]
            request_transform = None
            continue
        if not reported_progress:
            reported_progress = True
            pending = progress(outcome(STAGE, "canary_eval"))
            if pending is not None:
                await pending
        try:
            trial = await run_trial(bound, FULL_CASES, key=key, client=client, worker=worker)
        except RunAborted as exc:
            return _aborted(exc)
        verdict = judge(trial.answers, trial.errors())
        if verdict.verdict == "structure_fail":
            feedback = list(trial.errors(only=VISIBLE)) or [
                "some answers broke the contract on items you can't see; make the transforms handle every reply shape shown"
            ]
            if any(REQUEST_SIDE in error for error in trial.errors()):
                request_transform, request_failed = None, True
            continue
        if verdict.verdict == "delivery_fail":
            return outcome(STAGE, "failed", error_code=DELIVERY_UNCONFIRMED,
                           error_detail="the adapter's answers on held-out questions are no better than chance",
                           validations=[verdict.as_validation()])
        return outcome(
            STAGE, "active",
            adapter=adapter_report(adapter, auth, bound.content_hash, origin=origin if origin == "probe" or not round_index else "repair"),
            shape=shape_of(request.url, samples[0][1], samples[0][2]),
            validations=[verdict.as_validation()],
        )
    return outcome(STAGE, "failed", error_code=ONBOARDING_FAILED,
                   error_detail=f"no adapter passed after {REPAIR_ROUNDS} repairs: {clip('; '.join(feedback), 1500, key)}")


class _Retry(Exception):
    """This attempt failed in a way the next LLM round can fix; the message is its feedback."""


async def _seeded(request: OnboardRequest, *, key, client, worker) -> dict[str, Any] | None:
    for name in SEED_NAMES:
        bound = bind(request.url, request.model, request.auth, seed_adapter(name))
        try:
            checked = await seed_verdict(bound, key=key, client=client, worker=worker)
        except RunAborted as exc:
            if exc.status is not None and exc.status not in AUTH_REFUSED:
                continue
            raise
        if checked is None or checked[1].verdict != "pass":
            continue
        full, verdict = checked
        return outcome(STAGE, "active",
                       adapter=adapter_report(seed_adapter(name), request.auth, bound.content_hash, origin="seeded", seed_name=name),
                       shape=shape_of(request.url, seed_body(name, request.model, worker), full.mapped[0]["reply"]),
                       validations=[verdict.as_validation()])
    return None


async def _sample(
    request: OnboardRequest, transform: str, interaction: dict[str, Any], *, key, client, worker
) -> list[tuple[CanaryItem, Any, Any]]:
    """Send the visible items through `transform` and keep (item, body sent, reply)."""
    try:
        compile_expression(transform)
    except TransformError as exc:
        raise _Retry(f"the request transform was refused: {exc}") from exc
    samples = []
    for item in SAMPLE_ITEMS:
        question = item.as_question(ORIGINAL)
        data = request_input(model=request.model, instruction=question.instruction, input_text=question.input,
                             options=question.options, option_ids=question.option_ids)
        try:
            body = worker.run(transform, data)
            url, headers, sent = place_key(request.auth, key, request.url, body)
        except TransformError as exc:
            raise _Retry(f"the request transform failed on {item.id}: {exc}") from exc
        try:
            reply = await deliver(client, interaction, url, sent, headers)
        except GradioError as exc:
            raise _Retry(f"the Space failed on the body built for {item.id}: {exc}") from exc
        except TransportError as exc:
            raise RunAborted(f"could not reach the model: {exc}") from exc
        if reply.status in AUTH_REFUSED:
            raise RunAborted(f"the model refused the key (HTTP {reply.status})", status=reply.status)
        if not 200 <= reply.status < 300:
            raise _Retry(f"the API answered HTTP {reply.status} to the body built for {item.id}: {clip(json.dumps(reply.payload, ensure_ascii=False), 500, key)}")
        samples.append((item, body, reply.payload))
    return samples


def _response_transform(llm, request, samples, feedback, space, key) -> tuple[str, str]:
    if not feedback:
        try:
            document = to_spec("onboard", probe([
                Sample(raw=reply, option_ids=item.option_ids, options=item.options) for item, _, reply in samples
            ]))
            transform = compile_response(document["response"])
            compile_expression(transform)
            return transform, "probe"
        except (ValueError, CompileError, TransformError):
            pass  # ambiguous or unmappable: the LLM gets the replies
    transform = _ask(llm, _prompt(request, feedback, samples=samples, space=space, key=key), "response_transform", key)
    try:
        compile_expression(transform)
    except TransformError as exc:
        raise _Retry(f"the response transform was refused: {exc}") from exc
    return transform, "llm"


def _probability_flags(transform: str, samples, worker: TransformWorker) -> tuple[bool, bool]:
    """(degraded, derived): the transform never gives probabilities, or gives numbers no reply contains."""
    degraded, derived = True, False
    for item, _, reply in samples:
        try:
            mapped = worker.run(transform, response_input(reply, item.option_ids, item.options))
        except TransformError:
            return False, False  # the canary suite reports it as a structure failure
        probabilities = mapped.get("probabilities") if isinstance(mapped, dict) else None
        if isinstance(probabilities, dict):
            degraded = False
            # A transform may only copy a distribution; one it computed (one-hot, a softmax) is not the model's.
            derived = derived or not set(probabilities.values()) <= _numbers(reply)
    return degraded, derived


def _numbers(value: Any) -> set[float]:
    if isinstance(value, dict):
        return set().union(*(_numbers(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_numbers(item) for item in value))
    if isinstance(value, bool):
        return set()
    if isinstance(value, (int, float)):
        return {value}
    if isinstance(value, str):
        try:
            return {float(value)}
        except ValueError:
            return set()
    return set()


def _ask(llm: LlmProvider, prompt: str, field: str, key: str | None) -> str:
    try:
        # Replies and errors can echo the key; it must never reach the LLM.
        answer = extract_json(llm(scrub_key(prompt, key)))
    except LlmError as exc:
        raise SynthesisFailed(f"the mapping LLM failed: {exc}") from exc
    value = (answer or {}).get(field)
    if not isinstance(value, str) or not value.strip():
        raise _Retry(f"the last answer had no {field}; answer with a JSON object holding it")
    return value


def _prompt(request: OnboardRequest, feedback: list[str], *, samples, space: Any = None, key: str | None = None) -> str:
    example = SAMPLE_ITEMS[0].as_question(ORIGINAL)
    parts = [
        "You write the two JSONata transforms that let a benchmark talk to a decision-model API.",
        "A decision model reads a multiple-choice question and returns the option it picks, ideally with a probability per option.",
        CONTRACTS,
        RULES,
        "The user's working request for this API (its key replaced by <API_KEY>):",
        json.dumps(_without_credentials(request.redacted_request or {"url": request.url}), ensure_ascii=False, indent=2),
        "An example request transform input:",
        json.dumps(request_input(model=request.model, instruction=example.instruction, input_text=example.input,
                                 options=example.options, option_ids=example.option_ids), ensure_ascii=False),
    ]
    if space is not None:
        parts.append(
            "This API is a Gradio Space: the request body is {\"data\": [...]} with the endpoint's parameters in order, "
            "and the reply is the data array of its complete event. Its endpoints:"
        )
        parts.append(clip(json.dumps(space, ensure_ascii=False), 6000, key))
    if samples is not None:
        parts.append("Bodies your request transform built, and the API's replies:")
        parts.append(clip(json.dumps([{"option_ids": list(item.option_ids), "options": list(item.options),
                                  "sent": body, "reply": reply} for item, body, reply in samples], ensure_ascii=False), 20000, key))
        parts.append('Answer with only a JSON object: {"response_transform": "<JSONata>"}')
    else:
        parts.append('Answer with only a JSON object: {"request_transform": "<JSONata>"}')
    if feedback:
        parts.append("The last attempt failed:\n- " + "\n- ".join(feedback))
    return "\n\n".join(parts)


def _credential_name(name: str) -> bool:
    normal = name.strip().lower().replace("-", "_").replace(" ", "_")
    return is_secret_key(normal) or normal in ("key", "code", "sig", "signature") or normal.endswith("_key")


def _without_credentials(value: Any) -> Any:
    """Belt and braces over the backend's curl redaction: secret-looking fields, headers and query values go."""
    if isinstance(value, dict):
        return {
            name: "<API_KEY>" if isinstance(name, str) and _credential_name(name) else _without_credentials(item)
            for name, item in value.items()
        }
    if isinstance(value, list):
        return [_without_credentials(item) for item in value]
    if isinstance(value, str) and value.startswith(("http://", "https://")):
        parts = urlsplit(value)
        query = [(name, "<API_KEY>" if _credential_name(name) else item)
                 for name, item in parse_qsl(parts.query, keep_blank_values=True)]
        return urlunsplit(parts._replace(query=urlencode(query, safe="<>")))
    return redact_secrets(value)


def _aborted(exc: RunAborted) -> dict[str, Any]:
    return outcome(STAGE, "failed", error_code=abort_code(exc), error_detail=str(exc))
