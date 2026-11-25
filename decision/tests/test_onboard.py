"""Onboarding stages end to end against local model servers and a scripted LLM."""

from __future__ import annotations

import base64
import json

import pytest
from aiohttp import web

from decisioneval.onboard import __main__ as onboard_main
from decisioneval.onboard.__main__ import run_stage
from decisioneval.onboard.canary import HELD_OUT, ITEMS
from decisioneval.onboard.report import OnboardRequest, load_request, redact
from decisioneval.transforms.adapter import content_hash, seed_adapter, seed_hashes

GOLD = {item.question: item.gold_text for item in ITEMS}
CUSTOM_REQUEST = '{"prompt": input, "choices": options}'
PICK_RESPONSE = '($p := reply.out.pick; {"status": "answer", "choice": $$.option_ids[$p], "probabilities": null})'


def oracle_pick(question: str, ids: list[str], texts: list[str], mode: str) -> int:
    if mode == "middle":
        return len(ids) // 2
    return texts.index(GOLD[question])


def vector(ids, pick):
    rest = round(0.2 / (len(ids) - 1), 6)
    values = [rest] * len(ids)
    values[pick] = round(1 - rest * (len(ids) - 1), 6)
    return values


@pytest.fixture
async def model():
    state = {"protocol": "systemone", "mode": "gold", "seen": []}

    async def handle(request):
        if state["protocol"] == "unauthorized":
            return web.Response(status=401)
        body = await request.json()
        state["seen"].append((dict(request.headers), body))
        protocol = state["protocol"]
        if protocol == "systemone":
            if not isinstance(body.get("questions"), dict):
                return web.json_response({"error": "bad body"}, status=400)
            criteria = body["questions"]["answer"]["criteria"]
            ids, texts = list(criteria), list(criteria.values())
            question = body["state"].split("\n\n", 1)[-1]
            pick = oracle_pick(question, ids, texts, state["mode"])
            return web.json_response({"answers": {"answer": {"type": "choice", "choice": ids[pick],
                                                             "probabilities": dict(zip(ids, vector(ids, pick)))}}})
        if protocol == "openai":
            if not isinstance(body.get("questions"), list):
                return web.json_response({"error": "bad body"}, status=400)
            choices = body["questions"][0]["choices"]
            ids, texts = [c["value"] for c in choices], [c["description"] for c in choices]
            pick = oracle_pick(body["input"], ids, texts, state["mode"])
            rows = [{"value": i, "probability": p} for i, p in zip(ids, vector(ids, pick))]
            return web.json_response({"answers": [{"type": "choice", "name": "answer", "choice": ids[pick], "probabilities": rows}]})
        if protocol == "custom":
            if set(body) != {"prompt", "choices"}:
                # Real APIs often echo what they were sent; the key must still never reach a prompt.
                return web.json_response({"error": "expected prompt and choices",
                                          "you_sent": request.headers.get("Authorization")}, status=422)
            texts = body["choices"]
            ids = [str(i) for i in range(len(texts))]
            pick = oracle_pick(body["prompt"], ids, texts, state["mode"])
            return web.json_response({"out": {"pick": pick, "p": {str(i): v for i, v in enumerate(vector(ids, pick))}}})
        return web.json_response({"status": "ok", "note": "see the dashboard"})

    app = web.Application()
    app.router.add_post("/v1/decide", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/v1/decide", state
    finally:
        await runner.cleanup()


def onboard_request(url, kind="bearer", **extra) -> OnboardRequest:
    return OnboardRequest(url=url, model="m-1", auth={"kind": kind}, **extra)


class ScriptedLlm:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return json.dumps(self.answers.pop(0) if self.answers else {})


# -- probe stage ------------------------------------------------------------


async def test_a_system_one_endpoint_binds_the_seeded_version(model):
    url, _ = model
    report = await run_stage("probe", onboard_request(url), key="k", allow_private=True)
    assert report["outcome"] == "active"
    assert report["adapter"]["seed_name"] == "systemone" and report["adapter"]["origin"] == "seeded"
    assert report["adapter"]["content_hash"] == seed_hashes()["systemone/bearer"]
    assert report["validations"][0]["split"] == "held_out" and report["validations"][0]["verdict"] == "pass"


async def test_a_seeded_shape_that_answers_blind_is_not_bound(model):
    url, state = model
    state["mode"] = "middle"
    report = await run_stage("probe", onboard_request(url), key="k", allow_private=True)
    assert report["outcome"] == "awaiting_curl" and report["adapter"] is None


async def test_a_custom_bearer_prefix_is_sent_and_bound(model):
    url, state = model
    request = OnboardRequest(url=url, model="m-1", auth={"kind": "bearer", "prefix": "Token "})
    report = await run_stage("probe", request, key="k", allow_private=True)
    assert report["outcome"] == "active" and report["adapter"]["auth"] == {"kind": "bearer", "prefix": "Token "}
    assert state["seen"][0][0]["Authorization"] == "Token k"
    assert report["shape"]["host"] == "127.0.0.1"
    assert report["shape"]["request_keys"] == ["model", "questions", "state"]


async def test_an_openai_endpoint_skips_system_one_and_binds_the_openai_seed(model):
    url, state = model
    state["protocol"] = "openai"
    report = await run_stage("probe", onboard_request(url), key="k", allow_private=True)
    assert report["outcome"] == "active"
    assert report["adapter"]["content_hash"] == seed_hashes()["openai-decisions/bearer"]


async def test_a_keyless_system_one_endpoint_binds_the_keyless_seed(model):
    url, state = model
    report = await run_stage("probe", onboard_request(url, kind="none"), key=None, allow_private=True)
    assert report["adapter"]["content_hash"] == seed_hashes()["systemone/none"]
    assert all("Authorization" not in headers for headers, _ in state["seen"])


async def test_an_unknown_api_asks_for_a_curl_without_any_llm(model):
    url, state = model
    state["protocol"] = "opaque"
    report = await run_stage("probe", onboard_request(url), key="k", allow_private=True)
    assert report["outcome"] == "awaiting_curl" and report["adapter"] is None


async def test_a_refused_key_fails_the_probe(model):
    url, state = model
    state["protocol"] = "unauthorized"
    report = await run_stage("probe", onboard_request(url), key="k", allow_private=True)
    assert report["outcome"] == "failed" and "refused the key" in report["error_detail"]


async def test_a_private_address_is_refused_by_default(model):
    url, state = model
    report = await run_stage("probe", onboard_request(url), key="k")
    assert report["error_code"] == "DECISION_URL_NOT_PUBLIC" and state["seen"] == []


async def test_a_known_version_for_the_host_is_verified_with_the_full_suite(model):
    url, state = model
    state["protocol"] = "custom"
    custom = {**seed_adapter("systemone"), "request_transform": CUSTOM_REQUEST,
              "response_transform": PICK_RESPONSE}
    candidate = {"version_id": 9, "auth": {"kind": "bearer"}, "adapter": custom}
    report = await run_stage("probe", onboard_request(url, candidates=(candidate,)), key="k", allow_private=True)
    assert report["outcome"] == "active"
    assert report["adapter"]["content_hash"] == content_hash(custom, {"kind": "bearer"})
    assert report["validations"][0]["split"] == "held_out" and report["validations"][0]["verdict"] == "pass"


async def test_a_known_version_that_answers_blind_is_not_reused(model):
    url, state = model
    state["protocol"], state["mode"] = "custom", "middle"
    custom = {**seed_adapter("systemone"), "request_transform": CUSTOM_REQUEST,
              "response_transform": PICK_RESPONSE}
    report = await run_stage("probe", onboard_request(url, candidates=({"auth": {"kind": "bearer"}, "adapter": custom},)),
                             key="k", allow_private=True)
    assert report["outcome"] == "awaiting_curl"


# -- synthesize stage ---------------------------------------------------------


REDACTED = {"method": "POST", "url": "…", "headers": {"Authorization": "Bearer <API_KEY>"},
            "body": {"prompt": "ما عاصمة الأردن؟", "choices": ["عمّان", "إربد"]}}


async def test_synthesis_asks_the_llm_for_the_request_and_compiles_the_response(model):
    url, state = model
    state["protocol"] = "custom"
    llm = ScriptedLlm({"request_transform": CUSTOM_REQUEST})
    progress = []
    report = await run_stage("synthesize", onboard_request(url, redacted_request=REDACTED), key="s3cret-key-123",
                             llm=llm, progress=progress.append, allow_private=True)

    assert report["outcome"] == "active", report
    assert report["adapter"]["origin"] == "probe" and report["adapter"]["degraded"] is False
    assert report["shape"]["request_keys"] == ["choices", "prompt"] and report["shape"]["response_keys"] == ["out"]
    assert [entry["outcome"] for entry in progress] == ["canary_eval"]
    assert len(llm.prompts) == 1


async def test_held_out_items_and_the_key_never_reach_a_prompt(model):
    url, state = model
    state["protocol"] = "custom"
    llm = ScriptedLlm({"request_transform": '{"prompt": input}'}, {"request_transform": CUSTOM_REQUEST})
    await run_stage("synthesize", onboard_request(url, redacted_request=REDACTED), key="s3cret-key-123", llm=llm, allow_private=True)

    assert len(llm.prompts) == 2 and "422" in llm.prompts[1]  # the first body was refused, and the LLM was told
    for prompt in llm.prompts:
        assert "s3cret-key-123" not in prompt
        assert not any(item.question in prompt for item in HELD_OUT)


async def test_a_transform_that_garbles_the_question_is_not_activated(model):
    url, state = model
    state["protocol"], state["mode"] = "custom", "middle"
    report = await run_stage("synthesize", onboard_request(url, redacted_request=REDACTED), key="k",
                             llm=ScriptedLlm({"request_transform": CUSTOM_REQUEST}), allow_private=True)
    assert report["outcome"] == "failed" and report["error_code"] == "DECISION_DELIVERY_UNCONFIRMED"
    assert report["validations"][0]["verdict"] == "delivery_fail"


async def test_synthesis_gives_up_after_three_repairs(model):
    url, state = model
    state["protocol"] = "opaque"
    llm = ScriptedLlm(*([{"request_transform": CUSTOM_REQUEST}] + [{"response_transform": '{"status": "answer", "choice": reply.status}'}] * 5))
    report = await run_stage("synthesize", onboard_request(url, redacted_request=REDACTED), key="k", llm=llm, allow_private=True)
    assert report["outcome"] == "failed" and "after 3 repairs" in report["error_detail"]
    assert len(llm.prompts) == 5  # one request, then one response per round


async def test_a_seeded_protocol_behind_a_header_key_is_found_without_the_llm(model):
    url, state = model
    llm = ScriptedLlm()
    report = await run_stage("synthesize", OnboardRequest(url=url, model="m", auth={"kind": "header", "name": "x-api-key"},
                                                          redacted_request=REDACTED), key="k", llm=llm, allow_private=True)
    assert report["outcome"] == "active" and report["adapter"]["seed_name"] == "systemone"
    assert report["adapter"]["auth"] == {"kind": "header", "name": "x-api-key"}
    assert llm.prompts == [] and state["seen"][0][0]["x-api-key"] == "k"


async def test_without_an_llm_synthesis_fails_with_a_reason(model):
    url, state = model
    state["protocol"] = "custom"
    report = await run_stage("synthesize", onboard_request(url, redacted_request=REDACTED), key="k", allow_private=True)
    assert report["outcome"] == "failed" and "no mapping LLM" in report["error_detail"]


# -- the job ------------------------------------------------------------------


def test_the_report_never_carries_the_key():
    assert b"s3cret-key-123" not in redact({"error_detail": "bad token s3cret-key-123", "nested": ["Bearer s3cret-key-123"]}, "s3cret-key-123")


def test_a_broken_job_description_is_reported_as_failed(monkeypatch):
    sent = []
    monkeypatch.setattr(onboard_main, "post_report", lambda *args, **kwargs: sent.append(args[3]))
    code = onboard_main.main({"ONBOARDING_RUN_ID": "4", "API_HOST": "https://api.example", "ONBOARD_TOKEN": "t",
                              "ONBOARD_STAGE": "probe", "ONBOARD_REQUEST_B64": "%%%"})
    assert code == 1 and sent[0]["outcome"] == "failed" and "configuration error" in sent[0]["error_detail"]


def test_the_request_is_read_from_base64():
    encoded = base64.b64encode(json.dumps({"url": "https://m.example/v1", "model": "x",
                                           "auth": {"kind": "bearer", "name": None}, "candidates": []}).encode()).decode()
    assert load_request(encoded) == OnboardRequest(url="https://m.example/v1", model="x", auth={"kind": "bearer"})


async def test_a_request_transform_that_breaks_on_held_out_canaries_is_regenerated(model):
    url, state = model
    state["protocol"] = "custom"
    # Builds every visible sample, then fails on one held-out question during the canary run.
    fragile = '$contains(input, "التمر") ? $error("bad item") : {"prompt": input, "choices": options}'
    llm = ScriptedLlm({"request_transform": fragile}, {"request_transform": CUSTOM_REQUEST})
    report = await run_stage("synthesize", onboard_request(url, redacted_request=REDACTED), key="k", llm=llm, allow_private=True)
    assert report["outcome"] == "active", report
    assert len(llm.prompts) == 2 and '"request_transform"' in llm.prompts[1]
    assert "bad item" not in llm.prompts[1]  # the failure was on a held-out item, so only a generic hint


def test_a_lost_progress_report_does_not_end_the_run(monkeypatch):
    from decisioneval.onboard.report import ReportError

    sent = []

    async def fake_stage(stage, request, *, key, llm=None, progress=None, **_):
        progress({"stage": stage, "outcome": "canary_eval"})
        return {"stage": stage, "outcome": "active", "error_code": None}

    def post(api_host, token, run_id, report, **kwargs):
        if report["outcome"] == "canary_eval":
            raise ReportError("backend busy")
        sent.append(report)

    monkeypatch.setattr(onboard_main, "run_stage", fake_stage)
    monkeypatch.setattr(onboard_main, "post_report", post)
    request = base64.b64encode(json.dumps({"url": "https://m.example/v1", "model": "m", "auth": {"kind": "none"}}).encode()).decode()
    code = onboard_main.main({"ONBOARDING_RUN_ID": "5", "API_HOST": "https://api.example", "ONBOARD_TOKEN": "t",
                              "ONBOARD_STAGE": "synthesize", "ONBOARD_REQUEST_B64": request})
    assert code == 0 and [report["outcome"] for report in sent] == ["active"]


def test_a_key_the_curl_parser_missed_still_never_reaches_a_prompt():
    from decisioneval.onboard.synthesize import _prompt

    leaky = {"url": "https://h.example/d?key=AIzaSyLEAKED1234567&mode=x", "headers": {"X-Auth-Token": "tok-LEAKED-2"},
             "body": {"subscription_key": "LEAKED-3", "prompt": "q"}}
    prompt = _prompt(OnboardRequest(url="https://h.example/d", model="m", auth={"kind": "none"}, redacted_request=leaky),
                     [], samples=None)
    assert "LEAKED" not in prompt and "mode=x" in prompt


def test_a_long_error_detail_is_cut_only_after_the_key_is_gone():
    from decisioneval.onboard.report import post_report

    key = "sk-live-0123456789"
    sent = []

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def opener(request, timeout):
        sent.append(request.data)
        return Reply()

    detail = "y" * 1995 + key
    post_report("https://api.example", "t", "1", {"stage": "probe", "outcome": "failed", "error_detail": detail},
                key=key, opener=opener)
    body = json.loads(sent[0])
    assert "sk-live" not in sent[0].decode() and len(body["error_detail"]) == 2000
