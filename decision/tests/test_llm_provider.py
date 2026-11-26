"""The synthesis LLM: provider resolution, request shapes and reply handling, with no network."""

import base64
import json

import pytest

from decisioneval.onboard import __main__ as onboard_main
from decisioneval.onboard.llm import (
    AnthropicProvider,
    LlmConfigError,
    LlmError,
    OpenAICompatibleProvider,
    _chat_text_of,
    _text_of,
    extract_json,
    provider_from_env,
)


class Reply:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


@pytest.fixture
def capture(monkeypatch):
    sent = {}

    def install(payload):
        def fake_urlopen(request, timeout):
            sent["url"] = request.full_url
            sent["body"] = json.loads(request.data)
            sent["headers"] = dict(request.header_items())
            return Reply(payload)

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        return sent

    return install


def test_the_anthropic_request_names_opus_5_5_without_sampling_parameters(capture):
    sent = capture({"stop_reason": "end_turn", "content": [
        {"type": "thinking", "thinking": ""}, {"type": "text", "text": "{}"}]})
    assert AnthropicProvider(api_key="k")("prompt") == "{}"
    body = sent["body"]
    assert sent["url"] == "https://api.anthropic.com/v1/messages"
    assert body["model"] == "claude-opus-5-5" and "temperature" not in body and "thinking" not in body
    assert body["max_tokens"] >= 16000 and body["output_config"] == {"effort": "high"}
    assert body["fallbacks"] == "default"
    assert sent["headers"]["Anthropic-beta"] == "server-side-fallback-2026-07-01"
    assert sent["headers"]["X-api-key"] == "k"


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_a_refused_or_truncated_anthropic_reply_is_an_error(stop):
    with pytest.raises(LlmError):
        _text_of({"stop_reason": stop, "content": [{"type": "text", "text": "{"}]})


def test_the_openai_request_is_chat_completions_with_a_bearer_key(capture):
    sent = capture({"choices": [{"finish_reason": "stop", "message": {"content": " {} "}}]})
    provider = OpenAICompatibleProvider(api_key="k", model="m", base_url="https://llm.example/v1/")
    assert provider("prompt") == "{}"
    assert sent["url"] == "https://llm.example/v1/chat/completions"
    assert sent["headers"]["Authorization"] == "Bearer k"
    assert sent["body"] == {
        "model": "m",
        "max_completion_tokens": 16000,
        "messages": [{"role": "user", "content": "prompt"}],
    }


@pytest.mark.parametrize("payload", [
    {"choices": [{"finish_reason": "length", "message": {"content": "{"}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": None, "refusal": "no"}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": "  "}}]},
    {"choices": []},
    {"error": "x"},
    [],
])
def test_a_truncated_refused_or_empty_openai_reply_is_an_error(payload):
    with pytest.raises(LlmError):
        _chat_text_of(payload)


def test_no_key_means_no_provider():
    assert provider_from_env({"DECISION_SPEC_LLM_PROVIDER": "openai"}) is None


def test_an_anthropic_key_alone_resolves_to_anthropic_so_the_backend_secret_keeps_working():
    provider = provider_from_env({"DECISION_SPEC_LLM_API_KEY": "sk-ant-abc"})
    assert isinstance(provider, AnthropicProvider) and provider.model == "claude-opus-5-5"


def test_any_other_key_alone_resolves_to_openai_compatible():
    provider = provider_from_env({"DECISION_SPEC_LLM_API_KEY": "sk-proj-abc"})
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.model == "gpt-5.1" and provider.base_url == "https://api.openai.com/v1"


def test_an_explicit_provider_model_and_base_url_win_over_the_key_prefix():
    provider = provider_from_env({
        "DECISION_SPEC_LLM_API_KEY": "sk-ant-abc",
        "DECISION_SPEC_LLM_PROVIDER": " OpenAI ",
        "DECISION_SPEC_LLM_MODEL": "llama-3.3-70b",
        "DECISION_SPEC_LLM_BASE_URL": "https://api.groq.com/openai/v1",
    })
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.model == "llama-3.3-70b" and provider.base_url == "https://api.groq.com/openai/v1"


def test_an_explicit_anthropic_provider_honours_the_base_url(capture):
    sent = capture({"stop_reason": "end_turn", "content": [{"type": "text", "text": "ok"}]})
    provider = provider_from_env({
        "DECISION_SPEC_LLM_API_KEY": "plain-key",
        "DECISION_SPEC_LLM_PROVIDER": "anthropic",
        "DECISION_SPEC_LLM_BASE_URL": "https://proxy.example/",
    })
    assert provider("p") == "ok"
    assert sent["url"] == "https://proxy.example/v1/messages"


def test_an_unknown_provider_is_a_configuration_error():
    with pytest.raises(LlmConfigError, match="anthropic, openai"):
        provider_from_env({"DECISION_SPEC_LLM_API_KEY": "k", "DECISION_SPEC_LLM_PROVIDER": "gemini"})


def test_onboarding_reports_an_unknown_provider_as_a_configuration_error(monkeypatch):
    reports = []
    monkeypatch.setattr(onboard_main, "post_report", lambda host, token, run, report, **kwargs: reports.append(report))
    request = base64.b64encode(json.dumps({"url": "https://m.example/v1", "model": "m", "auth": {"kind": "none"}}).encode()).decode()
    code = onboard_main.main({
        "ONBOARDING_RUN_ID": "r", "API_HOST": "https://api.example", "ONBOARD_TOKEN": "t",
        "ONBOARD_STAGE": "synthesize", "ONBOARD_REQUEST_B64": request,
        "DECISION_SPEC_LLM_API_KEY": "k", "DECISION_SPEC_LLM_PROVIDER": "gemini",
    })
    assert code == 1 and reports[-1]["outcome"] == "failed"
    assert reports[-1]["error_detail"].startswith("configuration error: DECISION_SPEC_LLM_PROVIDER")


@pytest.mark.parametrize(("text", "expected"), [
    ('Here: {"response": {"a": 1}} done', {"a": 1}),
    ('{"a": 1}', {"a": 1}),
    ("no json", None),
    ("[1, 2]", None),
    ("{broken", None),
])
def test_extract_json_reads_the_outermost_object(text, expected):
    assert extract_json(text) == expected
