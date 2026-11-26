"""The gradio-call interaction against a fake Gradio Space."""

from __future__ import annotations

import json

import pytest
from aiohttp import web

from decisioneval.models import Question
from decisioneval.onboard.canary import ITEMS
from decisioneval.onboard.gradio_info import endpoint_info, gradio_interaction, wake
from decisioneval.onboard.report import OnboardRequest
from decisioneval.onboard.synthesize import synthesize_stage
from decisioneval.remote.bound_adapter import load_bound_adapter
from decisioneval.remote.execute import execute
from decisioneval.remote.gradio import GradioError, gradio_call, parse_events, space_base
from decisioneval.transforms.adapter import content_hash, seed_adapter
from decisioneval.transforms.worker import TransformWorker
from decisioneval.transport import ModelClient, RetryPolicy

REQUEST = '{"data": [state, $join(options, "\\n")]}'
GOLD = {item.question: item.gold_text for item in ITEMS}
PAIRS_RESPONSE = """(
  $rows := reply[0].options;
  $best := $reduce($rows, function($a, $b) { $a[1] >= $b[1] ? $a : $b });
  $i := $filter([0..($count($$.options) - 1)], function($i) { $$.options[$i] = $best[0] });
  {"status": "answer", "choice": $$.option_ids[$i], "probabilities": $merge($map($rows, function($r) {
    {$$.option_ids[$filter([0..($count($$.options) - 1)], function($i) { $$.options[$i] = $r[0] })]: $r[1]} }))}
)"""


@pytest.fixture
async def space():
    state = {"version": 4, "mode": "complete", "sleeping": 0, "root_sleeping": 0, "calls": [], "oracle": False, "events": {}}

    async def call(request):
        state["calls"].append(request.path)
        if state["sleeping"]:
            state["sleeping"] -= 1
            return web.Response(status=503, text="Space is building")
        body = await request.json()
        event_id = f"ev-{len(state['events']) + 1}"
        state["events"][event_id] = body["data"]
        return web.json_response({"event_id": event_id})

    async def stream(request):
        state["calls"].append(request.path)
        data = state["events"][request.match_info["event_id"]]
        options = data[1].split("\n")
        pick = options.index(GOLD[data[0].split("\n\n", 1)[-1]]) if state["oracle"] else 0
        rows = [[text, 0.7 if index == pick else round(0.3 / (len(options) - 1), 6)] for index, text in enumerate(options)]
        if state["mode"] == "error":
            payload = "event: error\ndata: null\n\n"
        elif state["mode"] == "silent":
            payload = "event: heartbeat\ndata: null\n\n"
        else:
            payload = (
                "event: heartbeat\ndata: null\n\n"
                "event: generating\ndata: []\n\n"
                f"event: complete\ndata: {json.dumps([{'options': rows}], ensure_ascii=False)}\n\n"
            )
        return web.Response(text=payload, content_type="text/event-stream")

    async def root(request):
        if state.get("require_bearer") and request.headers.get("Authorization") != f"Bearer {state['require_bearer']}":
            return web.Response(
                status=state.get("unauth_status", 404),
                text="<html><title>Hugging Face – The AI community building the future.</title><h1>404</h1></html>",
            )
        if state["root_sleeping"]:
            state["root_sleeping"] -= 1
            return web.Response(status=503, text="Space is building")
        return web.Response(status=state.get("root_status", 200), text=state.get("root_text", "<html>gradio</html>"))

    async def info(request):
        if state.get("require_bearer") and request.headers.get("Authorization") != f"Bearer {state['require_bearer']}":
            return web.Response(status=state.get("unauth_status", 404), text="not found")
        return web.json_response({"named_endpoints": {"/predict": {"parameters": [{"label": "question"}, {"label": "options"}]}}})

    app = web.Application()
    app.router.add_get("/", root)
    app.router.add_get("/gradio_api/info", info)
    for prefix in ("/gradio_api/call", "/call"):
        async def gated_call(request, prefix=prefix):
            if prefix == "/gradio_api/call" and state["version"] == 4:
                return web.Response(status=404)
            return await call(request)

        async def gated_stream(request, prefix=prefix):
            if prefix == "/gradio_api/call" and state["version"] == 4:
                return web.Response(status=404)
            return await stream(request)

        app.router.add_post(f"{prefix}/predict", gated_call)
        app.router.add_get(f"{prefix}/predict/{{event_id}}", gated_stream)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", state
    finally:
        await runner.cleanup()


def client():
    async def no_sleep(_):
        return None

    return ModelClient(retry=RetryPolicy(attempts=4, backoff_s=0.0), allow_private=True, sleep=no_sleep)


@pytest.mark.parametrize("version,prefix", [(4, "/call/"), (5, "/gradio_api/call/")])
async def test_a_call_reads_the_event_stream_until_complete(space, version, prefix):
    base, state = space
    state["version"] = version
    async with client() as model:
        reply = await gradio_call(model, f"{base}/call/predict", "/predict", {"data": ["س", "نعم\nلا"]}, {})
    assert reply.status == 200 and reply.payload == [{"options": [["نعم", 0.7], ["لا", 0.3]]}]
    assert state["calls"][-2:] == [f"{prefix}predict", f"{prefix}predict/ev-1"]


async def test_a_sleeping_space_is_retried_until_it_wakes(space):
    base, state = space
    state["sleeping"] = 2
    async with client() as model:
        reply = await gradio_call(model, base, "/predict", {"data": ["س", "a\nb"]}, {})
    assert reply.status == 200


async def test_a_stream_without_complete_is_a_gradio_error(space):
    base, state = space
    state["mode"] = "silent"
    async with client() as model:
        with pytest.raises(GradioError, match="without a complete"):
            await gradio_call(model, base, "/predict", {"data": ["س", "a\nb"]}, {})


async def test_an_error_event_is_a_5xx_reply_not_a_broken_adapter(space):
    base, state = space
    state["mode"] = "error"
    async with client() as model:
        reply = await gradio_call(model, base, "/predict", {"data": ["س", "a\nb"]}, {})
    assert reply.status == 502 and "reported an error" in reply.payload["error"]


def test_events_are_parsed_and_heartbeats_dropped():
    text = "event: heartbeat\ndata: null\n\nevent: complete\ndata: [1,\ndata: 2]\n\n"
    assert parse_events(text) == [("complete", "[1,\n2]")]


@pytest.mark.parametrize("url", ["https://x.hf.space", "https://x.hf.space/", "https://x.hf.space/call/predict",
                                 "https://x.hf.space/gradio_api/call/predict"])
def test_the_space_root_is_found_from_any_given_url(url):
    assert space_base(url) == "https://x.hf.space"


async def test_an_eval_runs_through_a_gradio_adapter(space):
    base, state = space
    auth = {"kind": "none"}
    response = '''(
      $rows := reply[0].options;
      $best := $reduce($rows, function($a, $b) { $a[1] >= $b[1] ? $a : $b });
      $i := $filter([0..($count($$.options) - 1)], function($i) { $$.options[$i] = $best[0] });
      {"status": "answer", "choice": $$.option_ids[$i], "probabilities": $merge($map($rows, function($r) {
        {$$.option_ids[$filter([0..($count($$.options) - 1)], function($i) { $$.options[$i] = $r[0] })]: $r[1]} }))}
    )'''
    adapter = {**seed_adapter("systemone"), "interaction": {"pattern": "gradio-call", "api_name": "/predict"},
               "request_transform": REQUEST, "response_transform": response}
    adapter["content_hash"] = content_hash(adapter, auth)
    bound = load_bound_adapter({"url": f"{base}/call/predict", "model": "", "auth": auth, "adapter": adapter})
    question = Question(id=1, dataset_name="d", category="c", task="t", instruction="اختر", input="سؤال",
                        options=("نعم", "لا", "ربما"), option_ids=("A", "B", "C"), gold_answer="نعم", gold_option_id="A")
    async with client() as model:
        with TransformWorker() as worker:
            [item] = await execute(bound, [question], key=None, client=model, worker=worker)
    assert item.correct and item.probabilities == {"A": 0.7, "B": 0.15, "C": 0.15}


async def test_a_gradio_error_mid_eval_is_invalid_and_retried_on_resume(space):
    base, state = space
    state["mode"] = "error"
    auth = {"kind": "none"}
    adapter = {**seed_adapter("systemone"), "interaction": {"pattern": "gradio-call", "api_name": "/predict"},
               "request_transform": REQUEST}
    adapter["content_hash"] = content_hash(adapter, auth)
    bound = load_bound_adapter({"url": base, "model": "", "auth": auth, "adapter": adapter})
    question = Question(id=1, dataset_name="d", category="c", task="t", instruction="", input="q",
                        options=("x", "y"), option_ids=("A", "B"), gold_answer="x", gold_option_id="A")
    async with client() as model:
        with TransformWorker() as worker:
            [item] = await execute(bound, [question], key=None, client=model, worker=worker)
    assert item.invalid and item.raw_response["http_status"] == 502  # http_status keeps it out of the checkpoint


# -- onboarding a Space ------------------------------------------------------


@pytest.mark.parametrize(
    "url,redacted,expected",
    [
        ("https://x.hf.space", None, {"pattern": "gradio-call", "api_name": "/predict"}),
        ("https://x.example/call/classify", None, {"pattern": "gradio-call", "api_name": "/classify"}),
        ("https://x.example/", {"url": "https://x.example/gradio_api/call/decide"}, {"pattern": "gradio-call", "api_name": "/decide"}),
        ("https://api.example/v1/decide", None, None),
    ],
)
def test_a_space_is_recognised_from_its_url_or_curl(url, redacted, expected):
    assert gradio_interaction(url, redacted) == expected


async def test_wake_polls_a_building_space_until_it_is_up(space):
    base, state = space
    state["root_sleeping"] = 2

    async def no_sleep(_):
        return None

    async with client() as model:
        assert await wake(model, f"{base}/call/predict", sleep=no_sleep)
        state["root_sleeping"] = 99
        ticks = iter(range(1000))
        assert not await wake(model, base, sleep=no_sleep, clock=lambda: next(ticks) * 100.0)


async def test_a_running_space_titled_sleeping_is_awake(space):
    base, state = space
    state["root_text"] = "<html><title>Sleeping-posture classifier</title><script>window.gradio_config = {}</script></html>"
    async with client() as model:
        assert await wake(model, base, sleep=lambda _: pytest.fail("polled a running Space"))
    state["root_text"] = "<html>This Space is sleeping due to inactivity</html>"
    ticks = iter(range(1000))

    async def no_sleep(_):
        return None

    async with client() as model:
        assert not await wake(model, base, sleep=no_sleep, clock=lambda: next(ticks) * 100.0)


class ScriptedLlm:
    def __init__(self, *answers):
        self.answers, self.prompts = list(answers), []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return json.dumps(self.answers.pop(0))


async def test_synthesis_onboards_a_space_as_a_gradio_call_adapter(space):
    base, state = space
    state["oracle"], state["root_sleeping"] = True, 1
    llm = ScriptedLlm({"request_transform": REQUEST}, {"response_transform": PAIRS_RESPONSE})
    request = OnboardRequest(url=f"{base}/call/predict", model="", auth={"kind": "none"},
                             redacted_request={"url": f"{base}/call/predict", "body": {"data": ["q", "a"]}})
    async with client() as model:
        with TransformWorker() as worker:
            report = await synthesize_stage(request, key=None, client=model, worker=worker, llm=llm)

    assert report["outcome"] == "active", report
    assert report["adapter"]["interaction"] == {"pattern": "gradio-call", "api_name": "/predict"}
    assert report["validations"][0]["verdict"] == "pass"
    assert '"question"' in llm.prompts[0]  # the Space's endpoint info reached the LLM


@pytest.mark.parametrize("path", ["", "/call/predict", "/gradio_api/call/predict"])
async def test_a_query_key_rides_on_both_requests(space, path):
    base, state = space
    seen = []
    original = ModelClient.send

    async def spy(self, url, body, headers, method="POST"):
        seen.append(url)
        return await original(self, url, body, headers, method)

    async with client() as model:
        model.send = spy.__get__(model)
        reply = await gradio_call(model, f"{base}{path}?key=K", "/predict", {"data": ["س", "a\nb"]}, {})
    assert reply.status == 200
    assert seen[-2:] == [f"{base}/call/predict?key=K", f"{base}/call/predict/ev-1?key=K"]


async def test_wake_and_endpoint_info_authenticate_private_spaces(space):
    base, state = space
    state["require_bearer"] = "hf_secret"
    state["oracle"] = True
    auth = {"kind": "bearer"}
    async with client() as model:
        assert await wake(model, f"{base}/call/predict", auth=auth, key="hf_secret",
                          sleep=lambda _: pytest.fail("polled an awake private Space"))
        info = await endpoint_info(model, f"{base}/call/predict", auth=auth, key="hf_secret")
        assert "/predict" in info

        llm = ScriptedLlm({"request_transform": REQUEST}, {"response_transform": PAIRS_RESPONSE})
        request = OnboardRequest(url=f"{base}/call/predict", model="", auth=auth,
                                 redacted_request={"url": f"{base}/call/predict", "body": {"data": ["q", "a"]}})
        with TransformWorker() as worker:
            report = await synthesize_stage(request, key="hf_secret", client=model, worker=worker, llm=llm)
    assert report["outcome"] == "active", report
    assert '"question"' in llm.prompts[0]


async def test_wake_skips_body_field_auth_on_get(space):
    base, _ = space
    async with client() as model:
        assert await wake(model, base, auth={"kind": "body_field", "path": "api_key"}, key="k",
                          sleep=lambda _: pytest.fail("polled an awake Space"))
