import base64
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.adapters.chat import anthropic, cohere, gemini, groq, local, openai


OPENAI_STYLE_MODULES = [openai, local, cohere]

CHAT_ADAPTERS = [
    (openai, openai.OpenAIAudioLM, openai.OpenAIChatCompletion),
    (local, local.LocalAudioLM, local.LocalChatCompletion),
    (cohere, cohere.CohereAudioLM, cohere.LocalChatCompletion),
    (anthropic, anthropic.AnthropicAudioLM, anthropic.AnthropicChat),
]


def audio(value=0.25):
    return {"array": np.array([0.0, value], dtype=np.float64), "sampling_rate": 8000}


def request(prompt="hello", kwargs=None, audio_items=None):
    aux = {} if audio_items is None else {"audio": audio_items}
    return SimpleNamespace(args=(prompt, kwargs or {}, aux))


@pytest.mark.parametrize("module", OPENAI_STYLE_MODULES)
def test_openai_style_audio_helpers_emit_wav_and_payload(module):
    encoded = module._audio_dicts_to_base64_wav([audio()])[0]
    assert base64.b64decode(encoded).startswith(b"RIFF")
    assert module._build_openai_audio_parts([audio()]) == [
        {"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}}
    ]


@pytest.mark.parametrize("module", OPENAI_STYLE_MODULES + [anthropic])
@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ('[{"role":"user","content":"hi"}]', [{"role": "user", "content": "hi"}]),
        ("plain", [{"role": "user", "content": "plain"}]),
        (42, [{"role": "user", "content": "42"}]),
    ],
)
def test_parse_chat_prompt_handles_json_plain_and_objects(module, prompt, expected):
    assert module._parse_chat_prompt(prompt) == expected


@pytest.mark.parametrize("module", OPENAI_STYLE_MODULES + [anthropic])
def test_parse_chat_prompt_handles_prompt_attribute_success_and_failure(module):
    valid = SimpleNamespace(prompt='[{"role":"assistant","content":"ok"}]')
    invalid = SimpleNamespace(prompt="not json")
    assert module._parse_chat_prompt(valid) == [{"role": "assistant", "content": "ok"}]
    assert module._parse_chat_prompt(invalid) == [{"role": "user", "content": "not json"}]


@pytest.mark.parametrize("module", OPENAI_STYLE_MODULES)
def test_audio_injection_targets_last_user_without_mutating_input(module):
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": 7, "type": "message"},
    ]
    part = {"type": "input_audio", "input_audio": {"data": "x", "format": "wav"}}
    result = module._inject_audio_into_messages(messages, [part])
    assert messages[-1]["content"] == 7
    assert result[-1] == {
        "role": "user",
        "content": [part, {"type": "text", "text": "7"}],
    }


@pytest.mark.parametrize("module", OPENAI_STYLE_MODULES)
def test_audio_injection_falls_back_to_last_message_and_stringifies_content(module):
    messages = [{"role": "assistant", "content": {"answer": 1}, "type": "message"}]
    audio_part = {"type": "audio-test"}
    inject = getattr(
        module,
        "_inject_audio_into_anthropic_messages",
        getattr(module, "_inject_audio_into_messages", None),
    )
    result = inject(messages, [audio_part])
    assert messages[0]["type"] == "message"
    assert result == [{
        "role": "assistant",
        "content": [audio_part, {"type": "text", "text": "{'answer': 1}"}],
    }]


@pytest.mark.parametrize("module", OPENAI_STYLE_MODULES + [anthropic])
def test_has_audio_requires_auxiliary_audio_key(module):
    assert module._has_audio([request(audio_items=[audio()])])
    assert not module._has_audio([SimpleNamespace(args=("x", {})), object()])


def test_anthropic_payload_extracts_system_and_filters_stops():
    payload = anthropic._messages_to_anthropic_payload(
        [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        ],
        model="claude-test",
        max_tokens=12,
        temperature=0.2,
        stop_sequences=["", "END", "  "],
    )
    assert payload == {
        "model": "claude-test",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        ],
        "max_tokens": 12,
        "temperature": 0.2,
        "stop_sequences": ["END"],
        "system": "rules",
    }


@pytest.mark.parametrize(
    ("module", "cls"),
    [(openai, openai.OpenAIAudioLM), (local, local.LocalAudioLM)],
)
def test_openai_style_audio_generation_builds_chat_request(module, cls, monkeypatch):
    model = object.__new__(cls)
    model.model = "model"
    model.model_call = MagicMock(return_value={"response": True})
    model.parse_generations = MagicMock(return_value=["generated"])
    monkeypatch.setattr(module, "_build_openai_audio_parts", lambda _: [{"type": "input_audio"}])

    result = model.generate_until(
        [request('[{"role":"user","content":"say"}]', {"temperature": 0.3}, [audio()])],
        disable_tqdm=True,
    )

    assert result == ["generated"]
    call = model.model_call.call_args.kwargs
    sent = json.loads(call["messages"][0].prompt)
    assert sent[0]["content"] == [{"type": "input_audio"}, {"type": "text", "text": "say"}]
    assert call["gen_kwargs"] == {"temperature": 0.3}


def test_local_generation_converts_client_errors_to_empty(monkeypatch):
    model = object.__new__(local.LocalAudioLM)
    model.model = "model"
    model.model_call = MagicMock(side_effect=RuntimeError("offline"))
    monkeypatch.setattr(local, "_build_openai_audio_parts", lambda _: [])
    assert model.generate_until([request(audio_items=[audio()])], disable_tqdm=True) == [""]


def test_cohere_rejects_audio_and_uses_native_wire_format():
    model = object.__new__(cohere.CohereAudioLM)
    model.model = "command"

    with pytest.raises(NotImplementedError, match="cohere-asr"):
        model.generate_until([request(audio_items=[audio()])], disable_tqdm=True)

    payload = model._create_payload(
        [{"role": "user", "content": "hello"}],
        generate=True,
        gen_kwargs={"max_gen_toks": 8, "until": "END", "top_p": 0.7},
    )
    assert payload == {
        "model": "command",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8,
        "temperature": 0,
        "stop_sequences": ["END"],
        "p": 0.7,
    }
    assert model.parse_generations({
        "message": {"content": [
            {"type": "text", "text": "first"},
            {"type": "text", "text": " second"},
        ]}
    }) == ["first second"]


def test_cohere_delegates_image_requests_to_multimodal_parent(monkeypatch):
    parent_generate = MagicMock(return_value=["vision result"])
    monkeypatch.setattr(cohere.LocalChatCompletion, "generate_until", parent_generate)
    model = object.__new__(cohere.CohereAudioLM)
    image_request = SimpleNamespace(args=("describe", {}, {"visual": [object()]}))

    assert model.generate_until([image_request], disable_tqdm=True) == ["vision result"]
    parent_generate.assert_called_once_with([image_request], disable_tqdm=True)


def test_cohere_payload_preserves_native_image_blocks():
    model = object.__new__(cohere.CohereAudioLM)
    model.model = "command-a-vision-07-2025"
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ],
    }]

    assert model._create_payload(messages, generate=True)["messages"] == messages


def test_anthropic_audio_generation_is_rejected():
    with pytest.raises(NotImplementedError, match="does not support audio"):
        object.__new__(anthropic.AnthropicAudioLM).generate_until(
            [request("hello", audio_items=[audio()])], disable_tqdm=True
        )


@pytest.mark.parametrize("module", OPENAI_STYLE_MODULES)
def test_runtime_protocol_methods_are_explicit_stubs(module):
    runtime = module._OpenAIChatRuntime if module is openai else module._LocalChatRuntime
    with pytest.raises(NotImplementedError):
        runtime.model_call(None)
    with pytest.raises(NotImplementedError):
        runtime.parse_generations(None, object())


@pytest.mark.parametrize(
    ("module", "cls", "parent", "expected_url"),
    [
        (openai, openai.OpenAIAudioLM, openai.OpenAIChatCompletion,
         "https://api.openai.com/v1/chat/completions"),
        (local, local.LocalAudioLM, local.LocalChatCompletion, None),
        (cohere, cohere.CohereAudioLM, cohere.LocalChatCompletion,
         "https://api.cohere.com/v2/chat"),
        (anthropic, anthropic.AnthropicAudioLM, anthropic.AnthropicChat,
         "https://api.anthropic.com/v1/messages"),
    ],
)
def test_adapter_constructors_resolve_defaults_and_forward_to_parent(
    module, cls, parent, expected_url, monkeypatch
):
    monkeypatch.delenv("BASE_URL", raising=False)
    calls = []

    def fake_parent_init(self, **kwargs):
        calls.append(kwargs)
        self.model = kwargs["model"]
        self.base_url = kwargs["base_url"]

    monkeypatch.setattr(parent, "__init__", fake_parent_init)
    model = cls(model="test-model", tokenizer_backend="none")
    assert model.model == "test-model"
    assert calls[0]["base_url"] == expected_url
    assert calls[0]["tokenizer_backend"] == "none"
    if module is not anthropic:
        assert calls[0]["tokenized_requests"] is False


@pytest.mark.parametrize("module,cls,parent", CHAT_ADAPTERS)
def test_generation_returns_empty_list_without_parent_or_network(
    module, cls, parent, monkeypatch
):
    parent_generate = MagicMock(side_effect=AssertionError("must not delegate"))
    monkeypatch.setattr(parent, "generate_until", parent_generate)
    assert object.__new__(cls).generate_until([], disable_tqdm=True) == []
    parent_generate.assert_not_called()


@pytest.mark.parametrize("module,cls,parent", CHAT_ADAPTERS)
def test_text_generation_delegates_to_parent(module, cls, parent, monkeypatch):
    calls = []

    def fake_generate(self, requests, disable_tqdm=False):
        calls.append((requests, disable_tqdm))
        return ["parent result"]

    monkeypatch.setattr(parent, "generate_until", fake_generate)
    requests = [request("text only")]
    assert object.__new__(cls).generate_until(requests, disable_tqdm=True) == ["parent result"]
    assert calls == [(requests, True)]


def test_cohere_auth_prefers_provider_key(monkeypatch):
    monkeypatch.setenv("CO_API_KEY", "cohere-key")
    monkeypatch.setenv("API_KEY", "fallback")
    model = object.__new__(cohere.CohereAudioLM)
    assert model.api_key == "cohere-key"
    assert model.header == {"Authorization": "Bearer cohere-key"}


def test_cohere_auth_requires_api_key(monkeypatch):
    monkeypatch.delenv("CO_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    with pytest.raises(ValueError, match="No API key found"):
        _ = object.__new__(cohere.CohereAudioLM).api_key


def test_groq_constructor_validates_key_and_normalizes_url(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No API key"):
        groq.GroqLM(model="m")

    client = MagicMock()
    monkeypatch.setattr(groq, "Groq", client)
    model = groq.GroqLM(api_key="key", base_url="https://host/openai/v1/chat/completions", model="m")
    client.assert_called_once_with(api_key="key", base_url="https://host")
    assert model.get_model_name() == "m"


def test_groq_properties_report_configured_model_limits():
    model = object.__new__(groq.GroqLM)
    model._tokenizer_name = "tokenizer"
    assert model.tokenizer_name == "tokenizer"
    assert model.max_sequence_length == 32768
    assert model.batch_size == 8


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("hello", {"role": "user", "content": "hello"}),
        ({"role": "assistant", "content": [{"type": "text", "text": "a"}, "b"], "name": "x"},
         {"role": "assistant", "content": "a b"}),
        (7, {"role": "user", "content": "7"}),
    ],
)
def test_groq_clean_message(message, expected):
    assert groq.GroqLM._clean_message(message) == expected


def test_groq_clean_message_handles_generic_text_blocks_and_dict_content():
    assert groq.GroqLM._clean_message({
        "content": [{"text": "generic"}, {"type": "image"}]
    }) == {"role": "user", "content": "generic"}
    assert groq.GroqLM._clean_message({
        "role": "system", "content": {"rule": "brief"}
    }) == {"role": "system", "content": "{'rule': 'brief'}"}


def test_groq_generation_rejects_unsupported_audio():
    model = object.__new__(groq.GroqLM)
    model.model_name = "audio-model"
    model._make_request_with_retry = MagicMock(return_value="transcript")
    with pytest.raises(NotImplementedError, match="openai-asr"):
        model.generate_until([request("transcribe", {"until": "END"}, [audio()])])
    model._make_request_with_retry.assert_not_called()


@pytest.mark.parametrize(
    ("instance", "expected"),
    [
        (("tuple prompt", "STOP"), ("tuple prompt", ["STOP"], None)),
        ({"prompt": "dict prompt", "until": "END"}, ("dict prompt", ["END"], None)),
        (17, ("17", [], None)),
    ],
)
def test_groq_extract_instance_data_fallback_formats(instance, expected):
    model = object.__new__(groq.GroqLM)
    assert model._extract_instance_data(instance) == expected


def test_groq_request_retries_empty_response_then_succeeds(monkeypatch):
    model = object.__new__(groq.GroqLM)
    model.model_name = "model"
    model.temperature = 0.2
    model.max_tokens = 50
    model.max_retries = 2
    model.retry_timeout = 0
    create = MagicMock(side_effect=[
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""))]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]),
    ])
    model.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    sleep = MagicMock()
    monkeypatch.setattr(groq.time, "sleep", sleep)

    assert model._make_request_with_retry(
        [{"role": "user", "content": "hi"}], stop=["END"], max_tokens=7
    ) == "answer"
    assert create.call_count == 2
    assert create.call_args.kwargs["max_tokens"] == 7
    assert create.call_args.kwargs["stop"] == ["END"]
    sleep.assert_called_once_with(0)


def test_groq_request_wraps_final_sdk_error(monkeypatch):
    model = object.__new__(groq.GroqLM)
    model.model_name = "model"
    model.temperature = 0
    model.max_tokens = 50
    model.max_retries = 2
    model.retry_timeout = 0
    create = MagicMock(side_effect=OSError("offline"))
    model.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(groq.time, "sleep", MagicMock())

    with pytest.raises(RuntimeError, match="Groq API call failed after 2 retries") as exc:
        model._make_request_with_retry([{"role": "user", "content": "hi"}])
    assert isinstance(exc.value.__cause__, OSError)
    assert create.call_count == 2


def test_groq_request_reports_exhausted_empty_responses(monkeypatch):
    model = object.__new__(groq.GroqLM)
    model.model_name = "model"
    model.temperature = 0
    model.max_tokens = 5
    model.max_retries = 1
    model.retry_timeout = 0
    create = MagicMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
    ))
    model.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(groq.time, "sleep", MagicMock())
    with pytest.raises(RuntimeError, match="Groq API call failed after 1 retries") as exc:
        model._make_request_with_retry([{"role": "user", "content": "hello"}])
    assert "returned empty response" in str(exc.value.__cause__)


def test_groq_generation_handles_empty_and_text_requests():
    model = object.__new__(groq.GroqLM)
    model.model_name = "model"
    model._make_request_with_retry = MagicMock(return_value="answer")

    assert model.generate_until([
        {"prompt": ""},
        {"prompt": "plain", "until": []},
    ]) == ["", "answer"]
    model._make_request_with_retry.assert_called_once_with(
        messages=[{"role": "user", "content": "plain"}], stop=None
    )


def test_groq_greedy_likelihood_and_chat_template_utilities(monkeypatch):
    model = object.__new__(groq.GroqLM)
    generate = MagicMock(return_value=["result"])
    monkeypatch.setattr(model, "generate_until", generate)
    requests = [object(), object()]

    assert model.greedy_until(requests) == ["result"]
    generate.assert_called_once_with(requests)
    assert model.loglikelihood(requests) == [(0.0, True), (0.0, True)]
    assert model.loglikelihood_rolling(requests) == [0.0, 0.0]
    assert model.apply_chat_template([]) == ""
    assert model.apply_chat_template([
        {"role": "system", "content": "rules"},
        {"content": "question"},
    ]) == "system: rules\nuser: question"


def test_gemini_constructor_and_generation_use_mocked_sdk(monkeypatch):
    client = MagicMock()
    client.models.generate_content.return_value = SimpleNamespace(text="result")
    sdk = MagicMock(return_value=client)
    monkeypatch.setattr(gemini.genai, "Client", sdk)
    model = gemini.GeminiLM(model_name="gemini-test", api_key="key", max_retries=1)

    assert model.generate_until([request("prompt", {"until": ["END"]})]) == ["result"]
    sdk.assert_called_once_with(api_key="key", http_options={"timeout": 120_000})
    call = client.models.generate_content.call_args.kwargs
    assert call["model"] == "models/gemini-test"
    assert call["contents"] == "prompt"
    assert call["config"].stop_sequences == ["END"]


def test_gemini_constructor_uses_environment_defaults(monkeypatch):
    client = MagicMock()
    sdk = MagicMock(return_value=client)
    monkeypatch.setattr(gemini.genai, "Client", sdk)
    monkeypatch.setenv("MODEL", "gemini-env")
    monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
    model = gemini.GeminiLM()
    assert model.model_name == "models/gemini-env"
    sdk.assert_called_once_with(api_key="env-key", http_options={"timeout": 120_000})


def test_gemini_constructor_requires_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No API key provided"):
        gemini.GeminiLM(model_name="gemini-test")


def test_gemini_properties_report_model_limits():
    model = object.__new__(gemini.GeminiLM)
    model._tokenizer_name = "tokenizer"
    assert model.tokenizer_name == "tokenizer"
    assert model.max_sequence_length == 32000
    assert model.batch_size == 8


def test_gemini_vertex_url_and_text_helpers(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(gemini.genai, "Client", MagicMock(return_value=client))
    url = "https://us-central1-aiplatform.googleapis.com/v1/projects/proj/locations/us-central1/publishers/google/models/gemini-2"
    model = gemini.GeminiLM(base_url=url)
    assert model.model_name == "gemini-2"
    assert model.tokenize("one, two!") == ["one", "two"]
    assert model.detokenize(["one", "two"]) == "one two"
    assert model.apply_chat_template([{"role": "user", "content": "Hi"}]) == "User: Hi\n\nAssistant:"
    with pytest.raises(ValueError, match="Could not parse"):
        gemini.GeminiLM._parse_vertex_url("https://invalid")


@pytest.mark.parametrize(
    ("instance", "expected"),
    [
        (("tuple prompt", "STOP"), ("tuple prompt", ["STOP"], None)),
        ({"prompt": "dict prompt", "until": "END"}, ("dict prompt", ["END"], None)),
        (17, ("17", [], None)),
    ],
)
def test_gemini_extract_instance_data_fallback_formats(instance, expected):
    model = object.__new__(gemini.GeminiLM)
    assert model._extract_instance_data(instance) == expected


def test_gemini_extract_instance_data_normalizes_string_until():
    model = object.__new__(gemini.GeminiLM)
    instance = SimpleNamespace(args=(SimpleNamespace(prompt="prompt"), {"until": "STOP"}, {}))
    assert model._extract_instance_data(instance) == ("prompt", ["STOP"], None)


def test_gemini_audio_conversion_creates_sdk_parts(monkeypatch):
    from_bytes = MagicMock(return_value="audio-part")
    monkeypatch.setattr(gemini.types.Part, "from_bytes", from_bytes)
    model = object.__new__(gemini.GeminiLM)

    assert model._audio_dicts_to_parts([audio()]) == ["audio-part"]
    call = from_bytes.call_args.kwargs
    assert call["data"].startswith(b"RIFF")
    assert call["mime_type"] == "audio/wav"


def test_gemini_generation_handles_empty_prompt_and_audio_contents(monkeypatch):
    model = object.__new__(gemini.GeminiLM)
    model.model_name = "models/test"
    model.max_retries = 1
    model.retry_timeout = 0
    model._audio_dicts_to_parts = MagicMock(return_value=["audio-part"])
    model._gen_config = MagicMock(return_value="config")
    generate = MagicMock(return_value=SimpleNamespace(text="transcript"))
    model.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))

    assert model.generate_until([
        {"prompt": ""},
        request("instruction", audio_items=[audio()]),
    ]) == ["", "transcript"]
    generate.assert_called_once_with(
        model="models/test",
        contents=["audio-part", "instruction"],
        config="config",
    )


def test_gemini_generation_retries_empty_and_sdk_errors(monkeypatch):
    model = object.__new__(gemini.GeminiLM)
    model.model_name = "models/test"
    model.max_retries = 3
    model.retry_timeout = 0
    model._gen_config = MagicMock(return_value="config")
    generate = MagicMock(side_effect=[
        SimpleNamespace(text=""),
        OSError("offline"),
        SimpleNamespace(text="answer"),
    ])
    model.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    sleep = MagicMock()
    monkeypatch.setattr(gemini.time, "sleep", sleep)

    assert model.generate_until(["prompt"]) == ["answer"]
    assert generate.call_count == 3
    assert sleep.call_args_list[0].args == (0,)
    assert sleep.call_count == 2


def test_gemini_generation_wraps_exhausted_empty_responses(monkeypatch):
    model = object.__new__(gemini.GeminiLM)
    model.model_name = "models/test"
    model.max_retries = 2
    model.retry_timeout = 0
    model._gen_config = MagicMock(return_value="config")
    generate = MagicMock(return_value=SimpleNamespace(text=""))
    model.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    monkeypatch.setattr(gemini.time, "sleep", MagicMock())

    with pytest.raises(RuntimeError, match="Generation failed for idx=0 after 2 retries"):
        model.generate_until(["prompt"])


def test_gemini_generation_wraps_final_sdk_error(monkeypatch):
    model = object.__new__(gemini.GeminiLM)
    model.model_name = "models/test"
    model.max_retries = 1
    model.retry_timeout = 0
    model._gen_config = MagicMock(return_value="config")
    model.client = SimpleNamespace(models=SimpleNamespace(
        generate_content=MagicMock(side_effect=OSError("offline"))
    ))
    with pytest.raises(RuntimeError, match="Generation failed for idx=0 after 1 retries") as exc:
        model.generate_until(["prompt"])
    assert isinstance(exc.value.__cause__, OSError)


def test_gemini_likelihood_token_count_and_string_template_fallbacks():
    model = object.__new__(gemini.GeminiLM)
    model.model_name = "models/test"
    count = MagicMock(side_effect=[SimpleNamespace(total_tokens=4), OSError("offline")])
    model.client = SimpleNamespace(models=SimpleNamespace(count_tokens=count))
    requests = [object(), object()]

    assert model.loglikelihood(requests) == [(0.0, True), (0.0, True)]
    assert model.loglikelihood_rolling(requests) == [0.0, 0.0]
    assert model.token_count(["api count", "fallback has three"]) == [4, 3]
    assert model.apply_chat_template("already formatted") == "already formatted"
    assert model.apply_chat_template(
        [{"role": "user", "content": "hello"}], add_generation_prompt=False
    ) == "User: hello"


def test_gemini_count_tokens_maps_missing_count_to_zero():
    model = object.__new__(gemini.GeminiLM)
    model.model_name = "models/test"
    model.client = SimpleNamespace(models=SimpleNamespace(
        count_tokens=MagicMock(return_value=SimpleNamespace(total_tokens=None))
    ))
    assert model.token_count(["text"]) == [0]


def test_gemini_create_completion_sends_overrides(monkeypatch):
    model = object.__new__(gemini.GeminiLM)
    model.model_name = "models/test"
    model.temperature = 0
    model.max_tokens = 100
    model.top_p = 0.9
    model.top_k = 20
    model.max_retries = 1
    model.retry_timeout = 0
    generate = MagicMock(return_value=SimpleNamespace(text="answer"))
    model.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))

    assert model.create_completion(
        "prompt", temperature=0.4, max_tokens=8, stop="END"
    ) == "answer"
    call = generate.call_args.kwargs
    assert call["model"] == "models/test"
    assert call["contents"] == "prompt"
    assert call["config"].temperature == 0.4
    assert call["config"].max_output_tokens == 8
    assert call["config"].stop_sequences == ["END"]


def test_gemini_create_completion_retries_empty_then_succeeds(monkeypatch):
    model = object.__new__(gemini.GeminiLM)
    model.model_name = "models/test"
    model.temperature = 0
    model.max_tokens = 100
    model.top_p = 0.9
    model.top_k = 20
    model.max_retries = 2
    model.retry_timeout = 0
    generate = MagicMock(side_effect=[
        SimpleNamespace(text=""), SimpleNamespace(text="answer")
    ])
    model.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    sleep = MagicMock()
    monkeypatch.setattr(gemini.time, "sleep", sleep)

    assert model.create_completion("prompt") == "answer"
    sleep.assert_called_once_with(0)


def test_gemini_create_completion_reports_exhausted_empty_response():
    model = object.__new__(gemini.GeminiLM)
    model.model_name = "models/test"
    model.temperature = 0
    model.max_tokens = 100
    model.top_p = 0.9
    model.top_k = 20
    model.max_retries = 1
    model.retry_timeout = 0
    model.client = SimpleNamespace(models=SimpleNamespace(
        generate_content=MagicMock(return_value=SimpleNamespace(text=""))
    ))
    with pytest.raises(RuntimeError, match="Gemini completion failed after 1 retries") as exc:
        model.create_completion("prompt")
    assert "attempts returned empty response" in str(exc.value.__cause__)


def test_gemini_create_completion_wraps_final_sdk_error(monkeypatch):
    model = object.__new__(gemini.GeminiLM)
    model.model_name = "models/test"
    model.temperature = 0
    model.max_tokens = 100
    model.top_p = 0.9
    model.top_k = 20
    model.max_retries = 2
    model.retry_timeout = 0
    generate = MagicMock(side_effect=OSError("offline"))
    model.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    monkeypatch.setattr(gemini.time, "sleep", MagicMock())

    with pytest.raises(RuntimeError, match="Gemini completion failed after 2 retries") as exc:
        model.create_completion("prompt")
    assert isinstance(exc.value.__cause__, OSError)


@pytest.mark.parametrize("cls", [openai.OpenAIAudioLM, local.LocalAudioLM, cohere.CohereAudioLM, anthropic.AnthropicAudioLM])
def test_chat_loglikelihood_stubs_preserve_cardinality(cls):
    model = object.__new__(cls)
    requests = [object(), object()]
    assert model.loglikelihood(requests) == [(0.0, True), (0.0, True)]
    assert model.loglikelihood_rolling(requests) == [0.0, 0.0]
