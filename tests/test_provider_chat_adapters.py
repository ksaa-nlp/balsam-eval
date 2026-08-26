import json
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from src.adapters.chat import aixplain, azure_openai, huggingface_chat
from src.adapters.chat._provider_utils import generation_options, parse_messages


def req(prompt, kwargs=None):
    return SimpleNamespace(args=(prompt, kwargs or {}))


def completion(text):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def test_shared_prompt_and_generation_parsing():
    messages = [{"role": "system", "content": "brief"}, {"role": "user", "content": "hi"}]
    assert parse_messages(json.dumps(messages)) == messages
    assert parse_messages("plain") == [{"role": "user", "content": "plain"}]
    assert generation_options(
        {"until": "END", "max_gen_toks": 7, "do_sample": False},
        temperature=0.8,
        max_tokens=99,
    ) == {"temperature": 0.0, "max_tokens": 7, "stop": ["END"]}


def test_azure_constructor_and_generate_retry(monkeypatch):
    client = MagicMock()
    create = client.chat.completions.create
    create.side_effect = [OSError("busy"), completion("answer")]
    sdk = MagicMock(return_value=client)
    monkeypatch.setattr(azure_openai, "AzureOpenAI", sdk)
    monkeypatch.setattr(azure_openai.time, "sleep", MagicMock())

    model = azure_openai.AzureOpenAIChatLM(
        deployment="deployment",
        endpoint="https://resource.openai.azure.com/",
        api_version="2025-01-01-preview",
        api_key="key",
        max_retries=2,
        retry_timeout=0,
    )
    result = model.generate_until(
        [req('[{"role":"user","content":"hello"}]', {"until": ["STOP"], "max_gen_toks": 8})],
        disable_tqdm=True,
    )

    assert result == ["answer"]
    sdk.assert_called_once_with(
        api_key="key",
        azure_endpoint="https://resource.openai.azure.com",
        api_version="2025-01-01-preview",
        timeout=120.0,
        max_retries=0,
    )
    assert create.call_count == 2
    assert create.call_args.kwargs == {
        "model": "deployment",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.0,
        "max_tokens": 8,
        "stop": ["STOP"],
    }


def test_azure_uses_environment_and_requires_configuration(monkeypatch):
    monkeypatch.setattr(azure_openai, "AzureOpenAI", MagicMock())
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "env-deployment")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://azure.example")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret")
    model = azure_openai.AzureOpenAIChatLM()
    assert model.model_name == "env-deployment"

    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT")
    monkeypatch.delenv("MODEL", raising=False)
    with pytest.raises(ValueError, match="deployment"):
        azure_openai.AzureOpenAIChatLM()


def test_azure_supports_reasoning_token_parameter(monkeypatch):
    client = MagicMock()
    client.chat.completions.create.return_value = completion("answer")
    monkeypatch.setattr(azure_openai, "AzureOpenAI", MagicMock(return_value=client))
    model = azure_openai.AzureOpenAIChatLM(
        deployment="o3-mini", endpoint="https://azure.example", api_key="key"
    )

    assert model.generate_until(
        [req("hello", {"max_completion_tokens": 64})], disable_tqdm=True
    ) == ["answer"]
    options = client.chat.completions.create.call_args.kwargs
    assert options["max_completion_tokens"] == 64
    assert "max_tokens" not in options


def test_huggingface_client_chat_completion_and_retry(monkeypatch):
    client = MagicMock()
    client.chat_completion.side_effect = [completion(""), completion("result")]
    sdk = MagicMock(return_value=client)
    monkeypatch.setattr(huggingface_chat, "InferenceClient", sdk)
    monkeypatch.setattr(huggingface_chat.time, "sleep", MagicMock())

    model = huggingface_chat.HuggingFaceChatLM(
        model="org/model", api_key="hf_key", provider="together", max_retries=2, retry_timeout=0
    )
    assert model.generate_until([req("hello", {"top_p": 0.9})], disable_tqdm=True) == ["result"]
    sdk.assert_called_once_with(token="hf_key", timeout=120.0, provider="together")
    assert client.chat_completion.call_args.kwargs["model"] == "org/model"
    assert client.chat_completion.call_args.kwargs["messages"] == [
        {"role": "user", "content": "hello"}
    ]


def test_huggingface_endpoint_and_optional_dependency(monkeypatch):
    sdk = MagicMock()
    monkeypatch.setattr(huggingface_chat, "InferenceClient", sdk)
    huggingface_chat.HuggingFaceChatLM(
        model="m", base_url="https://endpoint.example", api_key="key"
    )
    sdk.assert_called_once_with(
        token="key", timeout=120.0, base_url="https://endpoint.example"
    )
    monkeypatch.setattr(huggingface_chat, "InferenceClient", None)
    with pytest.raises(ImportError, match="optional dependency"):
        huggingface_chat.HuggingFaceChatLM(model="m")


def response(payload):
    item = MagicMock()
    item.json.return_value = payload
    return item


def test_aixplain_sync_request_and_response_shapes(monkeypatch):
    post = MagicMock(return_value=response({
        "status": "SUCCESS",
        "completed": True,
        "details": [{"message": {"role": "assistant", "content": "answer"}}],
    }))
    monkeypatch.setattr(aixplain.http_requests, "post", post)
    model = aixplain.AiXplainChatLM(model="id/with slash", api_key="key", max_retries=1)

    assert model.generate_until([req("hello", {"temperature": 0.2})], disable_tqdm=True) == ["answer"]
    assert model.url == "https://models.aixplain.com/api/v2/execute/id%2Fwith%20slash"
    assert post.call_args.kwargs["headers"]["x-api-key"] == "key"
    assert post.call_args.kwargs["json"] == {
        "text": [{"role": "user", "content": "hello"}],
        "temperature": 0.2,
        "max_tokens": 4096,
    }


def test_aixplain_polls_async_result(monkeypatch):
    monkeypatch.setattr(aixplain.http_requests, "post", MagicMock(return_value=response({
        "status": "IN_PROGRESS", "completed": False,
        "data": "https://platform-api.aixplain.com/result"
    })))
    get = MagicMock(side_effect=[
        response({"completed": False}),
        response({"status": "SUCCESS", "completed": True, "data": {"output": "done"}}),
    ])
    monkeypatch.setattr(aixplain.http_requests, "get", get)
    sleep = MagicMock()
    monkeypatch.setattr(aixplain.time, "sleep", sleep)
    model = aixplain.AiXplainChatLM(
        model="id", api_key="key", retry_timeout=0, max_poll_attempts=2
    )

    assert model.generate_until([req("hello")], disable_tqdm=True) == ["done"]
    assert get.call_args_list == [
        call("https://platform-api.aixplain.com/result", headers=model.headers, timeout=120.0),
        call("https://platform-api.aixplain.com/result", headers=model.headers, timeout=120.0),
    ]
    sleep.assert_called_once_with(0)


@pytest.mark.parametrize(
    "url",
    [
        "http://platform-api.aixplain.com/result",
        "https://attacker.example/result",
        "https://key@platform-api.aixplain.com/result",
    ],
)
def test_aixplain_rejects_untrusted_poll_url(url):
    model = aixplain.AiXplainChatLM(model="id", api_key="key")

    with pytest.raises(ValueError, match="untrusted polling URL"):
        model._poll(url)


@pytest.mark.parametrize(
    "cls",
    [azure_openai.AzureOpenAIChatLM, huggingface_chat.HuggingFaceChatLM, aixplain.AiXplainChatLM],
)
def test_provider_adapters_explicitly_reject_likelihood(cls):
    model = object.__new__(cls)
    with pytest.raises(NotImplementedError, match="loglikelihood"):
        model.loglikelihood([object()])
    with pytest.raises(NotImplementedError, match="loglikelihood"):
        model.loglikelihood_rolling([object()])


@pytest.mark.parametrize(
    "cls",
    [azure_openai.AzureOpenAIChatLM, huggingface_chat.HuggingFaceChatLM, aixplain.AiXplainChatLM],
)
def test_provider_adapters_preserve_chat_messages_in_template(cls):
    model = object.__new__(cls)
    history = [{"role": "user", "content": "hello"}]
    assert json.loads(model.apply_chat_template(history)) == history
