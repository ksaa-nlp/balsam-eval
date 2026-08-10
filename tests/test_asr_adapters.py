from types import SimpleNamespace
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


def test_qwen_transcription_sends_audio_data_uri_and_language_prompt():
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
    assert content[0]["input_audio"]["data"].startswith("data:audio/wav;base64,")
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
    assert "language=ar-SA" in call.args[0]
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


def test_huggingface_constructor_and_result_shapes(monkeypatch):
    client = MagicMock()
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(huggingface_asr, "InferenceClient", factory)
    model = huggingface_asr.HuggingFaceASRLM(
        model="hf/model", api_key="token", base_url="https://endpoint/", language="ar", max_retries=1
    )
    client.automatic_speech_recognition.return_value = {"text": "  text  "}
    assert model._transcribe_audio(b"wav") == "text"
    factory.assert_called_once_with(model="hf/model", token="token", api_url="https://endpoint")
    client.automatic_speech_recognition.assert_called_once_with(
        b"wav", extra_body={"language": "ar"}
    )


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
