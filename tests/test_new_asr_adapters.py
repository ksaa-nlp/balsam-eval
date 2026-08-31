from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import requests

from src.adapters.asr import (
    assemblyai_stt,
    cohere_asr,
    deepgram_stt,
    elevenlabs_stt,
    gladia_stt,
    revai_stt,
    speechmatics_stt,
)


CLASSES = [
    cohere_asr.CohereASRLM,
    deepgram_stt.DeepgramSTTLM,
    speechmatics_stt.SpeechmaticsSTTLM,
    assemblyai_stt.AssemblyAISTTLM,
    elevenlabs_stt.ElevenLabsSTTLM,
    gladia_stt.GladiaSTTLM,
    revai_stt.RevAISTTLM,
]


class Response:
    def __init__(self, payload=None, *, text="", status=200, headers=None):
        self.payload = payload or {}
        self.text = text
        self.status_code = status
        self.headers = headers or {}

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}", response=self)


def session(*responses):
    mocked = MagicMock()
    mocked.request.side_effect = responses
    return mocked


def audio():
    return {"array": np.array([0.0, 0.2]), "sampling_rate": 16000}


def request(items=None):
    auxiliary = {} if items is None else {"audio": items}
    return SimpleNamespace(args=("prompt", {}, auxiliary))


@pytest.mark.parametrize("cls", CLASSES)
def test_common_contract_wav_generation_and_validation(cls):
    model = cls(api_key="key", max_retries=1)
    assert model._audio_dict_to_wav_bytes(audio()).startswith(b"RIFF")
    assert model.tokenizer_name == model.model_name
    assert model.max_sequence_length == 0
    assert model.batch_size == 1
    assert model.loglikelihood([1]) == [(0.0, True)]
    assert model.loglikelihood_rolling([1]) == [0.0]

    model._transcribe_audio = MagicMock(side_effect=["one", "two"])
    assert model.generate_until([request([audio(), audio()]), request()]) == ["one two", ""]

    with pytest.raises(ValueError, match="positive integer"):
        cls(api_key="key", max_retries=0)
    with pytest.raises(ValueError, match="poll_timeout"):
        cls(api_key="key", poll_timeout=0)


@pytest.mark.parametrize(
    ("cls", "env_name"),
    [
        (cohere_asr.CohereASRLM, "COHERE_API_KEY"),
        (deepgram_stt.DeepgramSTTLM, "DEEPGRAM_API_KEY"),
        (speechmatics_stt.SpeechmaticsSTTLM, "SPEECHMATICS_API_KEY"),
        (assemblyai_stt.AssemblyAISTTLM, "ASSEMBLYAI_API_KEY"),
        (elevenlabs_stt.ElevenLabsSTTLM, "ELEVENLABS_API_KEY"),
        (gladia_stt.GladiaSTTLM, "GLADIA_API_KEY"),
        (revai_stt.RevAISTTLM, "REVAI_API_KEY"),
    ],
)
def test_provider_api_key_required(cls, env_name, monkeypatch):
    monkeypatch.delenv(env_name, raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    with pytest.raises(ValueError, match="No API key"):
        cls()


@pytest.mark.parametrize(
    ("cls", "url_env", "default_model", "default_language", "default_url"),
    [
        (
            cohere_asr.CohereASRLM,
            "COHERE_ASR_URL",
            "cohere-transcribe-03-2026",
            "ar",
            "https://api.cohere.com/v2/audio/transcriptions",
        ),
        (
            deepgram_stt.DeepgramSTTLM,
            "DEEPGRAM_STT_URL",
            "nova-3",
            "ar",
            "https://api.deepgram.com/v1/listen",
        ),
        (
            speechmatics_stt.SpeechmaticsSTTLM,
            "SPEECHMATICS_STT_URL",
            "standard",
            "ar",
            "https://asr.api.speechmatics.com/v2",
        ),
        (
            assemblyai_stt.AssemblyAISTTLM,
            "ASSEMBLYAI_STT_URL",
            "universal-3-5-pro",
            "ar",
            "https://api.assemblyai.com/v2",
        ),
        (
            elevenlabs_stt.ElevenLabsSTTLM,
            "ELEVENLABS_STT_URL",
            "scribe_v2",
            "ar",
            "https://api.elevenlabs.io/v1/speech-to-text",
        ),
        (
            gladia_stt.GladiaSTTLM,
            "GLADIA_STT_URL",
            "solaria-1",
            "ar",
            "https://api.gladia.io/v2",
        ),
        (
            revai_stt.RevAISTTLM,
            "REVAI_STT_URL",
            "machine",
            "ar",
            "https://api.rev.ai/speechtotext/v1",
        ),
    ],
)
def test_provider_defaults(
    cls, url_env, default_model, default_language, default_url, monkeypatch
):
    monkeypatch.delenv("MODEL", raising=False)
    monkeypatch.delenv("ASR_LANGUAGE", raising=False)
    monkeypatch.delenv(url_env, raising=False)
    model = cls(api_key="key")
    assert model.model_name == default_model
    assert model.language == default_language
    assert model.base_url == default_url


def test_cohere_payload_and_empty_retry(monkeypatch):
    model = cohere_asr.CohereASRLM(
        api_key="key", model="cohere-transcribe-03-2026",
        max_retries=2, retry_timeout=3,
    )
    model.session = session(Response({"text": ""}), Response({"text": " hello "}))
    sleep = MagicMock()
    monkeypatch.setattr("src.adapters.asr._http.time.sleep", sleep)

    assert model._transcribe_audio(b"wav") == "hello"
    first = model.session.request.call_args_list[0]
    assert first.kwargs["headers"]["Authorization"] == "Bearer key"
    assert first.kwargs["data"] == {
        "model": "cohere-transcribe-03-2026",
        "language": "ar",
    }
    assert first.kwargs["files"]["file"] == ("audio.wav", b"wav", "audio/wav")
    sleep.assert_called_once_with(3)


def test_deepgram_payload_and_result():
    payload = {
        "results": {
            "channels": [
                {"alternatives": [{"transcript": " first "}]},
                {"alternatives": [{"transcript": "second"}]},
            ]
        }
    }
    model = deepgram_stt.DeepgramSTTLM(
        api_key="key", model="nova-3", language="ar", max_retries=1
    )
    model.session = session(Response(payload))

    assert model._transcribe_audio(b"wav") == "first second"
    call = model.session.request.call_args
    assert call.kwargs["headers"]["Authorization"] == "Token key"
    assert call.kwargs["params"]["model"] == "nova-3"
    assert call.kwargs["params"]["language"] == "ar"
    assert call.kwargs["data"] == b"wav"


def test_elevenlabs_payload_result_and_default_language():
    model = elevenlabs_stt.ElevenLabsSTTLM(
        api_key="key", model="scribe_v2", max_retries=1
    )
    model.session = session(Response({"text": " words "}))

    assert model._transcribe_audio(b"wav") == "words"
    call = model.session.request.call_args
    assert call.kwargs["headers"] == {"xi-api-key": "key"}
    assert call.kwargs["data"] == {"model_id": "scribe_v2", "language_code": "ar"}


def test_assemblyai_upload_submit_poll_and_result(monkeypatch):
    model = assemblyai_stt.AssemblyAISTTLM(
        api_key="key", model="universal-3-5-pro", language="ar",
        max_retries=1, poll_interval=0,
    )
    model.session = session(
        Response({"upload_url": "https://upload"}),
        Response({"id": "job"}),
        Response({"status": "queued"}),
        Response({"status": "completed", "text": " result "}),
    )
    monkeypatch.setattr("src.adapters.asr._http.time.sleep", MagicMock())

    assert model._transcribe_audio(b"wav") == "result"
    calls = model.session.request.call_args_list
    assert calls[0].args[:2] == ("POST", "https://api.assemblyai.com/v2/upload")
    assert calls[1].kwargs["json"] == {
        "audio_url": "https://upload",
        "speech_models": ["universal-3-5-pro"],
        "language_code": "ar",
    }
    assert calls[-1].args[1].endswith("/transcript/job")


def test_gladia_upload_submit_poll_and_result():
    model = gladia_stt.GladiaSTTLM(
        api_key="key", model="solaria-1", language="ar",
        max_retries=1, poll_interval=0,
    )
    model.session = session(
        Response({"audio_url": "https://audio"}),
        Response({"id": "job"}),
        Response(
            {
                "status": "done",
                "result": {"transcription": {"full_transcript": " result "}},
            }
        ),
    )

    assert model._transcribe_audio(b"wav") == "result"
    calls = model.session.request.call_args_list
    assert calls[0].kwargs["headers"] == {"x-gladia-key": "key"}
    assert calls[1].kwargs["json"] == {
        "audio_url": "https://audio",
        "model": "solaria-1",
        "language_config": {"languages": ["ar"]},
    }


def test_revai_submit_poll_and_plaintext_result():
    model = revai_stt.RevAISTTLM(api_key="key", language="ar", max_retries=1)
    model.session = session(
        Response({"id": "job"}),
        Response({"status": "transcribed"}),
        Response(text=" result "),
    )

    assert model._transcribe_audio(b"wav") == "result"
    calls = model.session.request.call_args_list
    assert calls[0].kwargs["files"]["media"] == ("audio.wav", b"wav", "audio/wav")
    assert '"language": "ar"' in calls[0].kwargs["data"]["options"]
    assert calls[-1].kwargs["headers"]["Accept"] == "text/plain"


def test_speechmatics_submit_poll_and_json_transcript():
    model = speechmatics_stt.SpeechmaticsSTTLM(
        api_key="key", model="enhanced", language="ar", max_retries=1
    )
    model.session = session(
        Response({"id": "job"}),
        Response({"job": {"status": "done"}}),
        Response(
            {
                "results": [
                    {"alternatives": [{"content": "Hello"}]},
                    {"alternatives": [{"content": "."}]},
                ]
            }
        ),
    )

    assert model._transcribe_audio(b"wav") == "Hello."
    config = model.session.request.call_args_list[0].kwargs["data"]["config"]
    assert '"language": "ar"' in config
    assert '"operating_point": "enhanced"' in config


def test_http_retries_retryable_status_and_preserves_nonretryable_error(monkeypatch):
    model = cohere_asr.CohereASRLM(api_key="key", max_retries=2, retry_timeout=1)
    model.session = session(
        Response(status=429, headers={"Retry-After": "4"}),
        Response({"text": "ok"}),
    )
    sleep = MagicMock()
    monkeypatch.setattr("src.adapters.asr._http.time.sleep", sleep)
    assert model._transcribe_audio(b"wav") == "ok"
    sleep.assert_called_once_with(4)

    model.session = session(Response(status=401))
    with pytest.raises(RuntimeError, match="HTTP request failed:") as error:
        model._transcribe_audio(b"wav")
    assert isinstance(error.value.__cause__, requests.HTTPError)
    assert model.session.request.call_count == 1


def test_poll_failure_and_timeout(monkeypatch):
    model = assemblyai_stt.AssemblyAISTTLM(
        api_key="key", max_retries=1, poll_timeout=1, poll_interval=0
    )
    with pytest.raises(RuntimeError, match="bad audio"):
        model._poll(
            lambda: {"status": "error", "error": "bad audio"},
            lambda body: body["status"], pending={"queued"},
            succeeded={"completed"}, failed={"error"}, provider="AssemblyAI",
        )

    monotonic = MagicMock(side_effect=[0, 2])
    monkeypatch.setattr("src.adapters.asr._http.time.monotonic", monotonic)
    with pytest.raises(TimeoutError, match="timed out"):
        model._poll(
            lambda: {"status": "queued"}, lambda body: body["status"],
            pending={"queued"}, succeeded={"completed"}, failed={"error"},
            provider="AssemblyAI",
        )
