from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.adapters.asr import (
    azure_stt,
    google_stt,
    huggingface_asr,
    ibm_stt,
    nemo_asr,
    openai_asr,
    qwen_asr,
)


def _model(cls, client, *, retries=2):
    model = object.__new__(cls)
    model.client = client
    model.model_name = "test-model"
    model.language = None
    model.temperature = 0.0
    model.max_retries = retries
    model.retry_timeout = 2
    return model


def test_azure_retries_unknown_status_then_returns_no_match(monkeypatch):
    responses = []
    for payload in [
        {"RecognitionStatus": "InitialSilenceTimeout"},
        {"RecognitionStatus": "NoMatch"},
    ]:
        response = MagicMock()
        response.json.return_value = payload
        responses.append(response)

    post = MagicMock(side_effect=responses)
    sleep = MagicMock()
    monkeypatch.setattr(azure_stt.http_requests, "post", post)
    monkeypatch.setattr(azure_stt.time, "sleep", sleep)
    model = object.__new__(azure_stt.AzureSTTLM)
    model.endpoint_url = "https://azure.test/stt"
    model.language = "en-US"
    model.api_key = "key"
    model.max_retries = 2
    model.retry_timeout = 2

    assert model._transcribe_audio(b"wav", 16000) == ""
    assert post.call_count == 2
    sleep.assert_called_once_with(2)


def test_azure_success_without_hypothesis_is_retried_then_fails(monkeypatch):
    response = MagicMock()
    response.json.return_value = {"RecognitionStatus": "Success", "NBest": []}
    monkeypatch.setattr(azure_stt.http_requests, "post", MagicMock(return_value=response))
    sleep = MagicMock()
    monkeypatch.setattr(azure_stt.time, "sleep", sleep)
    model = object.__new__(azure_stt.AzureSTTLM)
    model.endpoint_url = "https://azure.test/stt"
    model.language = "en-US"
    model.api_key = "key"
    model.max_retries = 2
    model.retry_timeout = 3

    with pytest.raises(RuntimeError, match="failed after retries"):
        model._transcribe_audio(b"wav", 8000)
    sleep.assert_called_once_with(3)


def test_azure_retries_http_exception(monkeypatch):
    no_match = MagicMock()
    no_match.json.return_value = {"RecognitionStatus": "NoMatch"}
    monkeypatch.setattr(
        azure_stt.http_requests,
        "post",
        MagicMock(side_effect=[OSError("offline"), no_match]),
    )
    sleep = MagicMock()
    monkeypatch.setattr(azure_stt.time, "sleep", sleep)
    model = object.__new__(azure_stt.AzureSTTLM)
    model.endpoint_url = "https://azure.test/stt"
    model.language = "en-US"
    model.api_key = "key"
    model.max_retries = 2
    model.retry_timeout = 4

    assert model._transcribe_audio(b"wav", 16000) == ""
    sleep.assert_called_once_with(4)


def test_google_empty_then_exception_exhausts_offline(monkeypatch):
    speech = MagicMock()
    speech.RecognitionConfig.AudioEncoding.LINEAR16 = "LINEAR16"
    monkeypatch.setattr(google_stt, "speech", speech)
    sleep = MagicMock()
    monkeypatch.setattr(google_stt.time, "sleep", sleep)
    client = MagicMock()
    client.recognize.side_effect = [
        SimpleNamespace(results=[SimpleNamespace(alternatives=[])]),
        RuntimeError("offline"),
    ]
    model = _model(google_stt.GoogleSTTLM, client)

    assert model._transcribe_audio(b"wav", 16000) == ""
    sleep.assert_called_once_with(2)


def test_google_detects_sdk_disappearing_after_construction(monkeypatch):
    monkeypatch.setattr(google_stt, "speech", None)
    model = _model(google_stt.GoogleSTTLM, MagicMock(), retries=1)

    with pytest.raises(RuntimeError, match="became unavailable"):
        model._transcribe_audio(b"wav", 16000)


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (" raw string ", "raw string"),
        (SimpleNamespace(text=" object text "), "object text"),
    ],
)
def test_huggingface_accepts_sdk_result_protocols(result, expected):
    client = MagicMock()
    client.automatic_speech_recognition.return_value = result
    model = _model(huggingface_asr.HuggingFaceASRLM, client, retries=1)

    assert model._transcribe_audio(b"wav") == expected
    client.automatic_speech_recognition.assert_called_once_with(b"wav")


def test_huggingface_retries_empty_and_sdk_error(monkeypatch):
    client = MagicMock()
    client.automatic_speech_recognition.side_effect = [{}, OSError("offline")]
    sleep = MagicMock()
    monkeypatch.setattr(huggingface_asr.time, "sleep", sleep)
    model = _model(huggingface_asr.HuggingFaceASRLM, client)

    assert model._transcribe_audio(b"wav") == ""
    sleep.assert_called_once_with(2)


def test_ibm_retries_empty_results_and_protocol_error(monkeypatch):
    first = MagicMock()
    first.get_result.return_value = {
        "results": [{"alternatives": []}, {"alternatives": [{}]}]
    }
    client = MagicMock()
    client.recognize.side_effect = [first, RuntimeError("offline")]
    sleep = MagicMock()
    monkeypatch.setattr(ibm_stt.time, "sleep", sleep)
    model = _model(ibm_stt.IBMSTTLM, client)

    assert model._transcribe_audio(b"wav") == ""
    sleep.assert_called_once_with(2)


@pytest.mark.parametrize(
    ("module", "cls"),
    [
        (openai_asr, openai_asr.OpenAIWhisperLM),
        (nemo_asr, nemo_asr.NeMoASRLM),
    ],
)
def test_transcription_protocols_accept_string_and_text_object(module, cls):
    create = MagicMock(
        side_effect=[" string result ", SimpleNamespace(text=" object result ")]
    )
    client = SimpleNamespace(
        audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create))
    )
    model = _model(cls, client, retries=1)

    assert model._transcribe_audio(b"wav") == "string result"
    assert model._transcribe_audio(b"wav") == "object result"
    assert create.call_args_list[0].kwargs["file"].name == "audio.wav"


@pytest.mark.parametrize(
    ("module", "cls", "message"),
    [
        (openai_asr, openai_asr.OpenAIWhisperLM, "returned empty"),
        (nemo_asr, nemo_asr.NeMoASRLM, "returned empty"),
    ],
)
def test_transcription_empty_retries_raise_distinct_error(module, cls, message, monkeypatch):
    create = MagicMock(side_effect=[" ", SimpleNamespace(text="")])
    client = SimpleNamespace(
        audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create))
    )
    sleep = MagicMock()
    monkeypatch.setattr(module.time, "sleep", sleep)
    model = _model(cls, client)

    with pytest.raises(RuntimeError, match=message):
        model._transcribe_audio(b"wav")
    sleep.assert_called_once_with(2)


@pytest.mark.parametrize(
    ("module", "cls", "message"),
    [
        (openai_asr, openai_asr.OpenAIWhisperLM, "failed after retries"),
        (nemo_asr, nemo_asr.NeMoASRLM, "failed after retries"),
    ],
)
def test_transcription_sdk_errors_retry_and_preserve_cause(module, cls, message, monkeypatch):
    create = MagicMock(side_effect=[OSError("first"), ValueError("last")])
    client = SimpleNamespace(
        audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create))
    )
    sleep = MagicMock()
    monkeypatch.setattr(module.time, "sleep", sleep)
    model = _model(cls, client)

    with pytest.raises(RuntimeError, match=message) as exc_info:
        model._transcribe_audio(b"wav")
    assert isinstance(exc_info.value.__cause__, ValueError)
    sleep.assert_called_once_with(2)


def test_qwen_retries_invalid_choice_then_succeeds(monkeypatch):
    create = MagicMock(
        side_effect=[
            SimpleNamespace(choices=[]),
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=" words "))]
            ),
        ]
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    sleep = MagicMock()
    monkeypatch.setattr(qwen_asr.time, "sleep", sleep)
    model = _model(qwen_asr.QwenASRLM, client)
    model._audio_dict_to_base64_wav = MagicMock(return_value="encoded")

    assert model._transcribe_audio({}) == "words"
    sleep.assert_called_once_with(2)


def test_qwen_empty_retries_raise_empty_error(monkeypatch):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=" "))]
    )
    create = MagicMock(return_value=response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    sleep = MagicMock()
    monkeypatch.setattr(qwen_asr.time, "sleep", sleep)
    model = _model(qwen_asr.QwenASRLM, client)
    model._audio_dict_to_base64_wav = MagicMock(return_value="encoded")

    with pytest.raises(RuntimeError, match="returned empty"):
        model._transcribe_audio({})
    sleep.assert_called_once_with(2)


def test_qwen_invalid_protocol_preserves_final_error():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
    )
    create = MagicMock(return_value=response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    model = _model(qwen_asr.QwenASRLM, client, retries=1)
    model._audio_dict_to_base64_wav = MagicMock(return_value="encoded")

    with pytest.raises(RuntimeError, match="failed after retries") as exc_info:
        model._transcribe_audio({})
    assert isinstance(exc_info.value.__cause__, ValueError)


@pytest.mark.parametrize(
    ("module", "cls", "attribute", "message"),
    [
        (google_stt, google_stt.GoogleSTTLM, "speech", "google-cloud-speech"),
        (huggingface_asr, huggingface_asr.HuggingFaceASRLM, "InferenceClient", "huggingface-hub"),
        (ibm_stt, ibm_stt.IBMSTTLM, "SpeechToTextV1", "ibm-watson"),
    ],
)
def test_optional_sdk_constructor_errors(module, cls, attribute, message, monkeypatch):
    monkeypatch.setattr(module, attribute, None)

    with pytest.raises(ImportError, match=message):
        cls()


@pytest.mark.parametrize(
    ("cls", "kwargs", "message"),
    [
        (azure_stt.AzureSTTLM, {}, "No API key"),
        (openai_asr.OpenAIWhisperLM, {}, "No API key"),
        (nemo_asr.NeMoASRLM, {}, "No API key"),
        (qwen_asr.QwenASRLM, {}, "No API key"),
        (huggingface_asr.HuggingFaceASRLM, {}, "No API token"),
        (ibm_stt.IBMSTTLM, {"api_key": "key"}, "No service URL"),
    ],
)
def test_missing_constructor_configuration(cls, kwargs, message, monkeypatch):
    for name in (
        "API_KEY",
        "OPENAI_API_KEY",
        "NVIDIA_API_KEY",
        "AZURE_SPEECH_KEY",
        "HF_TOKEN",
        "BASE_URL",
        "IBM_STT_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    if cls is ibm_stt.IBMSTTLM:
        monkeypatch.setattr(ibm_stt, "IAMAuthenticator", MagicMock())
        monkeypatch.setattr(ibm_stt, "SpeechToTextV1", MagicMock())

    with pytest.raises(ValueError, match=message):
        cls(**kwargs)


@pytest.mark.parametrize(
    ("module", "cls", "kwargs"),
    [
        (azure_stt, azure_stt.AzureSTTLM, {"api_key": "key"}),
        (google_stt, google_stt.GoogleSTTLM, {}),
        (huggingface_asr, huggingface_asr.HuggingFaceASRLM, {"api_key": "key"}),
        (ibm_stt, ibm_stt.IBMSTTLM, {"api_key": "key", "base_url": "https://ibm"}),
    ],
)
def test_remaining_constructors_validate_retry_values(module, cls, kwargs, monkeypatch):
    if module is google_stt:
        monkeypatch.setattr(module, "speech", MagicMock())
    elif module is huggingface_asr:
        monkeypatch.setattr(module, "InferenceClient", MagicMock())
    elif module is ibm_stt:
        monkeypatch.setattr(module, "IAMAuthenticator", MagicMock())
        monkeypatch.setattr(module, "SpeechToTextV1", MagicMock())

    with pytest.raises(ValueError, match="positive integer"):
        cls(max_retries=True, **kwargs)
    with pytest.raises(ValueError, match="finite and non-negative"):
        cls(retry_timeout=float("inf"), **kwargs)
