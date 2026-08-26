import builtins
import runpy
import sys
from types import SimpleNamespace
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
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
from lm_eval.api import registry as lm_registry


ASR_CLASSES = [
    openai_asr.OpenAIWhisperLM,
    qwen_asr.QwenASRLM,
    google_stt.GoogleSTTLM,
    azure_stt.AzureSTTLM,
    ibm_stt.IBMSTTLM,
    huggingface_asr.HuggingFaceASRLM,
    nemo_asr.NeMoASRLM,
]


def audio(value=0.2):
    return {"array": np.array([0.0, value], dtype=np.float64), "sampling_rate": 16000}


def request(items=None):
    return SimpleNamespace(args=("prompt", {}, {} if items is None else {"audio": items}))


@pytest.mark.parametrize("cls", ASR_CLASSES)
def test_extract_audio_and_lm_properties(cls):
    model = object.__new__(cls)
    model._tokenizer_name = "tokenizer"
    assert model._extract_audio(request([audio()]))[0]["sampling_rate"] == 16000
    assert model._extract_audio(request()) is None
    assert model.tokenizer_name == "tokenizer"
    assert model.max_sequence_length == 0
    assert model.batch_size == 1


@pytest.mark.parametrize("cls", ASR_CLASSES)
def test_loglikelihood_stubs(cls):
    model = object.__new__(cls)
    requests = [object(), object()]
    assert model.loglikelihood(requests) == [(0.0, True), (0.0, True)]
    assert model.loglikelihood_rolling(requests) == [0.0, 0.0]


@pytest.mark.parametrize(
    "converter",
    [
        openai_asr.OpenAIWhisperLM._audio_dict_to_wav_bytes,
        ibm_stt.IBMSTTLM._audio_dict_to_wav_bytes,
        huggingface_asr.HuggingFaceASRLM._audio_dict_to_wav_bytes,
        nemo_asr.NeMoASRLM._audio_dict_to_wav_bytes,
    ],
)
def test_audio_conversion_produces_wav(converter):
    assert converter(audio()).startswith(b"RIFF")


@pytest.mark.parametrize("cls", [google_stt.GoogleSTTLM, azure_stt.AzureSTTLM])
def test_audio_conversion_returns_sample_rate(cls):
    wav, rate = cls._audio_dict_to_wav_bytes(audio())
    assert wav.startswith(b"RIFF")
    assert rate == 16000


def test_qwen_audio_conversion_produces_base64_wav():
    import base64

    encoded = qwen_asr.QwenASRLM._audio_dict_to_base64_wav(audio())
    assert base64.b64decode(encoded).startswith(b"RIFF")


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://host", "https://host/v1"),
        ("https://host/v1/audio/transcriptions", "https://host/v1"),
        ("https://host/v1/", "https://host/v1"),
    ],
)
def test_openai_asr_normalizes_base_urls(url, expected):
    assert openai_asr.OpenAIWhisperLM._normalize_base_url(url) == expected


@pytest.mark.parametrize("module,cls,kwargs", [
    (openai_asr, openai_asr.OpenAIWhisperLM, {"api_key": "key"}),
    (qwen_asr, qwen_asr.QwenASRLM, {"api_key": "key", "base_url": "https://host"}),
    (nemo_asr, nemo_asr.NeMoASRLM, {"api_key": "key"}),
])
def test_openai_sdk_constructors_are_offline_and_validate_retry(module, cls, kwargs, monkeypatch):
    client = MagicMock()
    sdk = MagicMock(return_value=client)
    monkeypatch.setattr(module, "OpenAI", sdk)
    model = cls(model="asr-model", retry_timeout=0, **kwargs)
    assert model.client is client
    with pytest.raises(ValueError, match="positive integer"):
        cls(model="m", max_retries=0, **kwargs)
    with pytest.raises(ValueError, match="finite and non-negative"):
        cls(model="m", retry_timeout=float("nan"), **kwargs)


def test_openai_constructor_normalizes_supplied_endpoint(monkeypatch):
    sdk = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(openai_asr, "OpenAI", sdk)
    openai_asr.OpenAIWhisperLM(
        model="m",
        api_key="key",
        base_url="https://host/v1/audio/transcriptions/",
    )
    sdk.assert_called_once_with(api_key="key", base_url="https://host/v1")


def test_qwen_constructor_requires_base_url(monkeypatch):
    monkeypatch.delenv("BASE_URL", raising=False)
    with pytest.raises(ValueError, match="No base URL provided"):
        qwen_asr.QwenASRLM(model="m", api_key="key")


def test_nemo_constructor_appends_v1_to_custom_base(monkeypatch):
    sdk = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(nemo_asr, "OpenAI", sdk)
    nemo_asr.NeMoASRLM(model="m", api_key="key", base_url="https://nim/")
    sdk.assert_called_once_with(api_key="key", base_url="https://nim/v1")


def test_azure_constructor_strips_custom_base_url():
    model = azure_stt.AzureSTTLM(api_key="key", base_url="https://azure/custom/")
    assert model.endpoint_url == "https://azure/custom"


def test_ibm_constructor_requires_api_key_before_service_url(monkeypatch):
    monkeypatch.setattr(ibm_stt, "IAMAuthenticator", MagicMock())
    monkeypatch.setattr(ibm_stt, "SpeechToTextV1", MagicMock())
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("IBM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No API key provided"):
        ibm_stt.IBMSTTLM(base_url="https://ibm")


def test_openai_transcription_sends_representative_payload():
    model = object.__new__(openai_asr.OpenAIWhisperLM)
    model.model_name = "whisper"
    model.language = "ar"
    model.temperature = 0.1
    model.max_retries = 1
    model.retry_timeout = 0
    create = MagicMock(return_value=SimpleNamespace(text="  transcript  "))
    model.client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
    assert model._transcribe_audio(b"wav") == "transcript"
    payload = create.call_args.kwargs
    assert payload["model"] == "whisper"
    assert payload["language"] == "ar"
    assert payload["file"].name == "audio.wav"


def test_nemo_transcription_includes_language():
    model = object.__new__(nemo_asr.NeMoASRLM)
    model.model_name = "nemo"
    model.language = "ar"
    model.temperature = 0
    model.max_retries = 1
    model.retry_timeout = 0
    create = MagicMock(return_value=" transcript ")
    model.client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
    assert model._transcribe_audio(b"wav") == "transcript"
    assert create.call_args.kwargs["language"] == "ar"


def test_qwen_transcription_sends_raw_audio_base64_and_language_prompt():
    model = object.__new__(qwen_asr.QwenASRLM)
    model.model_name = "qwen"
    model.language = "Arabic"
    model.max_retries = 1
    model.retry_timeout = 0
    create = MagicMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=" words "))]
    ))
    model.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert model._transcribe_audio(audio()) == "words"
    content = create.call_args.kwargs["messages"][0]["content"]
    assert not content[0]["input_audio"]["data"].startswith("data:")
    assert content[1] == {"type": "text", "text": "Transcribe this audio in Arabic."}


def test_azure_constructor_and_transcription_payload(monkeypatch):
    monkeypatch.setenv("AZURE_SPEECH_REGION", "west")
    model = azure_stt.AzureSTTLM(api_key="key", retry_timeout=0, max_retries=1)
    response = MagicMock()
    response.json.return_value = {"RecognitionStatus": "Success", "NBest": [{"Display": " hello "}]}
    post = MagicMock(return_value=response)
    monkeypatch.setattr(azure_stt.http_requests, "post", post)
    assert model._transcribe_audio(b"wav", 8000) == "hello"
    call = post.call_args
    assert call.kwargs["params"] == {"language": "ar-SA", "format": "detailed"}
    assert call.kwargs["headers"]["Ocp-Apim-Subscription-Key"] == "key"
    assert "samplerate=8000" in call.kwargs["headers"]["Content-Type"]


def test_google_constructor_and_transcription_are_fully_mocked(monkeypatch):
    speech = MagicMock()
    client = MagicMock()
    speech.SpeechClient.return_value = client
    speech.RecognitionConfig.AudioEncoding.LINEAR16 = "linear16"
    client.recognize.return_value = SimpleNamespace(results=[
        SimpleNamespace(alternatives=[SimpleNamespace(transcript="one")]),
        SimpleNamespace(alternatives=[SimpleNamespace(transcript="two")]),
    ])
    monkeypatch.setattr(google_stt, "speech", speech)
    model = google_stt.GoogleSTTLM(model="latest", language="en-US", max_retries=1)
    assert model._transcribe_audio(b"wav", 16000) == "one two"
    config = speech.RecognitionConfig.call_args.kwargs
    assert config["language_code"] == "en-US"
    assert config["model"] == "latest"


def test_google_transcription_sleeps_after_retryable_sdk_error(monkeypatch):
    speech = MagicMock()
    speech.RecognitionConfig.AudioEncoding.LINEAR16 = "linear16"
    model = object.__new__(google_stt.GoogleSTTLM)
    model.language = "en-US"
    model.model_name = "default"
    model.max_retries = 2
    model.retry_timeout = 3
    model.client = MagicMock()
    model.client.recognize.side_effect = [OSError("offline"), SimpleNamespace(results=[])]
    sleep = MagicMock()
    monkeypatch.setattr(google_stt, "speech", speech)
    monkeypatch.setattr(google_stt.time, "sleep", sleep)
    assert model._transcribe_audio(b"wav", 16000) == ""
    sleep.assert_called_once_with(3)


def test_huggingface_constructor_and_result_shapes(monkeypatch):
    client = MagicMock()
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(huggingface_asr, "InferenceClient", factory)
    model = huggingface_asr.HuggingFaceASRLM(
        model="hf/model", api_key="token", base_url="https://endpoint/", language="ar", max_retries=1
    )
    client.automatic_speech_recognition.return_value = {"text": "  text  "}
    assert model._transcribe_audio(b"wav") == "text"
    factory.assert_called_once_with(model="https://endpoint", token="token")
    client.automatic_speech_recognition.assert_called_once_with(
        b"wav", extra_body={"language": "ar"}
    )


def test_huggingface_transcription_sleeps_after_retryable_error(monkeypatch):
    model = object.__new__(huggingface_asr.HuggingFaceASRLM)
    model.language = None
    model.max_retries = 2
    model.retry_timeout = 2
    model.client = MagicMock()
    model.client.automatic_speech_recognition.side_effect = [
        OSError("offline"), {"text": "ok"}
    ]
    sleep = MagicMock()
    monkeypatch.setattr(huggingface_asr.time, "sleep", sleep)
    assert model._transcribe_audio(b"wav") == "ok"
    sleep.assert_called_once_with(2)


def test_ibm_constructor_and_transcription_are_fully_mocked(monkeypatch):
    authenticator = MagicMock(return_value="auth")
    client = MagicMock()
    factory = MagicMock(return_value=client)
    client.recognize.return_value.get_result.return_value = {
        "results": [{"alternatives": [{"transcript": " hello "}]}]
    }
    monkeypatch.setattr(ibm_stt, "IAMAuthenticator", authenticator)
    monkeypatch.setattr(ibm_stt, "SpeechToTextV1", factory)
    model = ibm_stt.IBMSTTLM(api_key="key", base_url="https://ibm/", max_retries=1)
    assert model._transcribe_audio(b"wav") == "hello"
    authenticator.assert_called_once_with("key")
    client.set_service_url.assert_called_once_with("https://ibm")


def test_ibm_transcription_sleeps_after_retryable_error(monkeypatch):
    model = object.__new__(ibm_stt.IBMSTTLM)
    model.model_name = "model"
    model.max_retries = 2
    model.retry_timeout = 4
    succeeded = MagicMock()
    succeeded.get_result.return_value = {"results": []}
    model.client = MagicMock()
    model.client.recognize.side_effect = [OSError("offline"), succeeded]
    sleep = MagicMock()
    monkeypatch.setattr(ibm_stt.time, "sleep", sleep)
    assert model._transcribe_audio(b"wav") == ""
    sleep.assert_called_once_with(4)


@pytest.mark.parametrize(
    ("module", "blocked_import", "optional_name"),
    [
        (google_stt, "google.cloud", "speech"),
        (huggingface_asr, "huggingface_hub", "InferenceClient"),
    ],
)
def test_optional_import_failure_paths_are_isolated(
    module, blocked_import, optional_name, monkeypatch
):
    original_import = builtins.__import__

    def isolated_import(name, *args, **kwargs):
        if name == blocked_import:
            raise ImportError("optional dependency unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", isolated_import)
    monkeypatch.setattr(lm_registry, "register_model", lambda _name: lambda cls: cls)
    namespace = runpy.run_path(module.__file__, run_name=f"isolated_{optional_name}")
    assert namespace[optional_name] is None


def test_ibm_optional_import_success_isolated_with_fake_sdks(monkeypatch):
    cloud_package = ModuleType("ibm_cloud_sdk_core")
    authenticators = ModuleType("ibm_cloud_sdk_core.authenticators")
    authenticators.IAMAuthenticator = type("IAMAuthenticator", (), {})
    watson = ModuleType("ibm_watson")
    watson.SpeechToTextV1 = type("SpeechToTextV1", (), {})
    monkeypatch.setitem(sys.modules, "ibm_cloud_sdk_core", cloud_package)
    monkeypatch.setitem(sys.modules, "ibm_cloud_sdk_core.authenticators", authenticators)
    monkeypatch.setitem(sys.modules, "ibm_watson", watson)
    monkeypatch.setattr(lm_registry, "register_model", lambda _name: lambda cls: cls)
    namespace = runpy.run_path(ibm_stt.__file__, run_name="isolated_ibm_success")
    assert namespace["IAMAuthenticator"] is authenticators.IAMAuthenticator
    assert namespace["SpeechToTextV1"] is watson.SpeechToTextV1


@pytest.mark.parametrize("cls", ASR_CLASSES)
def test_generation_joins_multiple_audio_transcriptions_and_handles_missing_audio(cls):
    model = object.__new__(cls)
    model.model_name = "model"
    if cls in (google_stt.GoogleSTTLM, azure_stt.AzureSTTLM):
        model._audio_dict_to_wav_bytes = MagicMock(return_value=(b"wav", 16000))
        model._transcribe_audio = MagicMock(side_effect=["first", "second"])
    elif cls is qwen_asr.QwenASRLM:
        model._transcribe_audio = MagicMock(side_effect=["first", "second"])
    else:
        model._audio_dict_to_wav_bytes = MagicMock(return_value=b"wav")
        model._transcribe_audio = MagicMock(side_effect=["first", "second"])
    assert model.generate_until([request([audio(), audio()]), request()]) == ["first second", ""]
