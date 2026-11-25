"""The pooled async model client against a real local aiohttp server."""

import asyncio

import pytest
from aiohttp import web

from decisioneval.spec.transport import TransportError
from decisioneval.transport import ModelClient, RetryPolicy

ANSWER = {"answers": {"answer": {"choice": "A", "probabilities": {"A": 0.9, "B": 0.1}}}}


async def no_sleep(_seconds):
    return None


@pytest.fixture
async def server():
    state = {"hits": {}, "in_flight": 0, "peak": 0, "auth": []}

    async def handle(request):
        name = request.match_info["name"]
        state["hits"][name] = state["hits"].get(name, 0) + 1
        state["auth"].append(request.headers.get("Authorization"))
        if name == "flaky" and state["hits"][name] == 1:
            return web.Response(status=429, headers={"Retry-After": "0"})
        if name == "down":
            return web.Response(status=503)
        if name == "denied":
            return web.Response(status=401, text="bad key")
        if name == "bounce":
            raise web.HTTPFound("http://localhost:1/steal")
        if name == "slow":
            state["in_flight"] += 1
            state["peak"] = max(state["peak"], state["in_flight"])
            await asyncio.sleep(0.05)
            state["in_flight"] -= 1
        if name == "text":
            return web.Response(text="not json")
        return web.json_response({**ANSWER, "echo": await request.json()})

    app = web.Application()
    app.router.add_route("*", "/{name}", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", state
    finally:
        await runner.cleanup()


def client(**kwargs):
    kwargs.setdefault("retry", RetryPolicy(attempts=3, backoff_s=0.0))
    return ModelClient(sleep=no_sleep, allow_private=True, **kwargs)


async def test_a_json_body_round_trips_with_its_headers(server):
    base, state = server
    async with client() as model:
        reply = await model.send(f"{base}/ok", {"q": "سؤال"}, {"Authorization": "Bearer k"})
    assert reply.status == 200 and reply.payload["echo"] == {"q": "سؤال"}
    assert state["auth"] == ["Bearer k"]


async def test_a_retryable_status_is_retried_then_succeeds(server):
    base, state = server
    async with client() as model:
        reply = await model.send(f"{base}/flaky", {}, {})
    assert reply.status == 200 and state["hits"]["flaky"] == 2


async def test_a_non_retryable_status_is_returned_after_one_attempt(server):
    base, state = server
    async with client() as model:
        reply = await model.send(f"{base}/denied", {}, {})
    assert reply.status == 401 and state["hits"]["denied"] == 1


async def test_retries_stop_at_the_attempt_budget_and_return_the_last_reply(server):
    base, state = server
    async with client() as model:
        reply = await model.send(f"{base}/down", {}, {})
    assert reply.status == 503 and state["hits"]["down"] == 3


async def test_a_redirect_is_refused_not_followed(server):
    base, state = server
    async with client() as model:
        with pytest.raises(TransportError, match="refused a 302 redirect"):
            await model.send(f"{base}/bounce", {}, {"Authorization": "Bearer k"})
    assert state["hits"]["bounce"] == 1


async def test_a_body_that_is_not_json_is_kept_as_text(server):
    base, _ = server
    async with client() as model:
        reply = await model.send(f"{base}/text", {}, {})
    assert reply.payload == {"_text": "not json"}


async def test_an_unreachable_host_raises_after_the_retries():
    async with client(timeout_s=1.0) as model:
        with pytest.raises(TransportError, match="after 3 attempt"):
            await model.send("http://127.0.0.1:9/none", {}, {})


async def test_concurrency_never_exceeds_the_limit(server):
    base, state = server
    async with client(concurrency=2) as model:
        await asyncio.gather(*(model.send(f"{base}/slow", {}, {}) for _ in range(6)))
    assert state["peak"] == 2
