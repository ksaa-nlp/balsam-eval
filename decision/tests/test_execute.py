"""The straight eval loop over a bound adapter, against a real local model server."""

from __future__ import annotations

import pytest
from aiohttp import web

from decisioneval.metrics.probability import MALFORMED, MISSING, probability_scores
from decisioneval.models import Question
from decisioneval.remote.bound_adapter import load_bound_adapter
from decisioneval.remote.execute import BREAKER_LIMIT, CircuitBroken, RunAborted, execute
from decisioneval.transforms.adapter import content_hash, seed_adapter
from decisioneval.transforms.worker import TransformWorker
from decisioneval.transport import ModelClient, RetryPolicy

OPTIONS = ("نعم", "لا", "ربما")


def questions(count: int = 6) -> list[Question]:
    return [
        Question(id=f"q{i}", dataset_name="pool", category="7", task="t", instruction="اختر",
                 input=f"سؤال {i}", options=OPTIONS, option_ids=("A", "B", "C"),
                 gold_answer=OPTIONS[i % 3], gold_option_id="ABC"[i % 3])
        for i in range(count)
    ]


def vector(pick: str) -> dict[str, float]:
    return {key: (0.8 if key == pick else 0.1) for key in "ABC"}


def system_one(body, pick):
    return {"answers": {"answer": {"type": "choice", "choice": pick, "probabilities": vector(pick)}}}


def openai(body, pick):
    rows = [{"value": key, "probability": value} for key, value in vector(pick).items()]
    return {"model": "gpt-6-luna", "answers": [{"type": "choice", "name": "answer", "choice": pick, "probabilities": rows}]}


def gold_pick(body) -> str:
    text = body.get("input") or body.get("state", "")
    return "ABC"[int(text.rsplit(" ", 1)[-1]) % 3]


@pytest.fixture
async def model_server():
    seen: list = []
    behaviour = {"reply": system_one, "status": 200, "refuse": None}

    async def handle(request):
        body = await request.json()
        seen.append((dict(request.headers), body))
        if behaviour["status"] != 200:
            return web.Response(status=behaviour["status"])
        if behaviour["refuse"] and behaviour["refuse"] in body.get("state", ""):
            return web.json_response({"error": "context too long"}, status=413)
        reply = behaviour["reply"](body, gold_pick(body))
        return web.Response(text=reply) if isinstance(reply, str) else web.json_response(reply)

    app = web.Application()
    app.router.add_post("/decide", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/decide", behaviour, seen
    finally:
        await runner.cleanup()


def bound(url, seed="systemone", auth=None, **changes):
    auth = auth or {"kind": "bearer"}
    adapter = {**seed_adapter(seed), **changes}
    adapter["content_hash"] = content_hash(adapter, auth)
    return load_bound_adapter({"url": url, "model": "m-1", "auth": auth, "adapter": adapter})


async def run(adapter, items, key="s3cret"):
    async with ModelClient(retry=RetryPolicy(attempts=2, backoff_s=0.0), allow_private=True) as client:
        with TransformWorker() as worker:
            return await execute(adapter, items, key=key, client=client, worker=worker)


@pytest.mark.parametrize("seed,reply", [("systemone", system_one), ("openai-decisions", openai)])
async def test_a_seeded_protocol_scores_every_question_in_order(model_server, seed, reply):
    url, behaviour, seen = model_server
    behaviour["reply"] = reply
    items = await run(bound(url, seed), questions())

    assert [item.item_id for item in items] == [f"q{i}" for i in range(6)]
    assert all(item.correct and not item.invalid for item in items)
    assert probability_scores(items)["probability_metrics"]["status"] == "available"
    assert all(headers["Authorization"] == "Bearer s3cret" for headers, _ in seen)


async def test_the_key_never_reaches_a_transform_or_a_stored_record(model_server):
    url, _, seen = model_server
    echo_input = '{"model": model, "state": state, "questions": {"answer": {"criteria": criteria}}, "everything": $}'
    items = await run(bound(url, request_transform=echo_input), questions(1))
    assert "s3cret" not in repr(seen[0][1])
    assert "s3cret" not in repr(items[0].raw_response)


async def test_refusals_are_invalid_and_never_trip_the_breaker(model_server):
    url, behaviour, _ = model_server
    behaviour["reply"] = lambda body, pick: {"answers": {"answer": {"type": "refusal"}}}
    items = await run(bound(url), questions(BREAKER_LIMIT * 2))

    assert all(item.invalid and not item.correct for item in items)
    assert {item.raw_response["invalid_reason"] for item in items} == {"refusal"}
    assert probability_scores(items)["probability_metrics"]["reason"] == MISSING


async def test_a_reply_that_is_not_json_is_a_malformed_response_not_an_adapter_error(model_server):
    url, behaviour, _ = model_server
    behaviour["reply"] = lambda body, pick: "<html>busy</html>"
    items = await run(bound(url), questions(BREAKER_LIMIT + 1))
    assert {item.raw_response["invalid_reason"] for item in items} == {"malformed_response"}


async def test_one_malformed_vector_nulls_calibration_as_malformed_but_keeps_the_choice(model_server):
    url, behaviour, _ = model_server
    behaviour["reply"] = lambda body, pick: {
        "answers": {"answer": {"choice": pick, "probabilities": {"A": 0.5, "B": 0.5, "C": 0.5}
                               if body["state"].endswith("0") else vector(pick)}}}
    items = await run(bound(url), questions())
    assert items[0].correct and items[0].probabilities is None
    assert probability_scores(items)["probability_metrics"]["reason"] == MALFORMED


async def test_consecutive_malformed_vectors_break_the_circuit(model_server):
    url, behaviour, _ = model_server
    behaviour["reply"] = lambda body, pick: {"answers": {"answer": {"choice": pick, "probabilities": {"A": 2}}}}
    with pytest.raises(CircuitBroken):
        await run(bound(url), questions(BREAKER_LIMIT + 3))


async def test_a_response_transform_that_errors_breaks_the_circuit(model_server):
    url, _, seen = model_server
    with pytest.raises(CircuitBroken):
        await run(bound(url, response_transform='$error("no")'), questions(20))
    assert len(seen) < 20


async def test_a_model_that_keeps_failing_aborts_the_run(model_server):
    url, behaviour, _ = model_server
    behaviour["status"] = 503
    with pytest.raises(RunAborted, match="503"):
        await run(bound(url), questions(2))


async def test_an_unreachable_model_aborts_the_run():
    with pytest.raises(RunAborted, match="could not reach"):
        await run(bound("http://127.0.0.1:9/decide"), questions(1))


async def test_one_item_the_model_refuses_with_a_4xx_is_invalid_and_the_run_finishes(model_server):
    url, behaviour, _ = model_server
    behaviour["refuse"] = "سؤال 2"
    items = await run(bound(url), questions())
    assert len(items) == 6 and items[2].invalid
    assert items[2].raw_response["invalid_reason"] == "adapter_error" and items[2].raw_response["http_status"] == 413
    assert all(item.correct for index, item in enumerate(items) if index != 2)


async def test_every_item_refused_with_a_4xx_aborts_as_the_account_not_the_adapter(model_server):
    url, behaviour, _ = model_server
    behaviour["status"] = 400  # what some APIs answer for a bad key or an empty balance
    with pytest.raises(RunAborted, match="refused 5 requests in a row") as caught:
        await run(bound(url), questions(BREAKER_LIMIT + 2))
    assert caught.value.status == 400


@pytest.mark.parametrize("status", [401, 403, 404, 429])
async def test_a_wrong_key_url_or_lasting_rate_limit_aborts(model_server, status):
    url, behaviour, _ = model_server
    behaviour["status"] = status
    with pytest.raises(RunAborted, match=str(status)):
        await run(bound(url), questions(2))
