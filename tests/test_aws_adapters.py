import builtins
import io
import json
import runpy
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import numpy as np
import pytest

from lm_eval.api import registry as lm_registry
from src.adapters.asr import aws_transcribe
from src.adapters.chat import aws_bedrock, aws_sagemaker


def request(prompt="hello", kwargs=None, audio_items=None):
    auxiliary = {} if audio_items is None else {"audio": audio_items}
    return SimpleNamespace(args=(prompt, kwargs or {}, auxiliary))


def audio():
    return {"array": np.array([0.0, 0.2]), "sampling_rate": 16000}


def test_aws_transcribe_constructor_uses_standard_client_configuration(monkeypatch):
    clients = {"transcribe": MagicMock(), "s3": MagicMock()}
    factory = MagicMock(side_effect=lambda service, **_kwargs: clients[service])
    monkeypatch.setattr(aws_transcribe, "boto3", SimpleNamespace(client=factory))
    monkeypatch.setenv("AWS_REGION", "me-south-1")
    monkeypatch.setenv("AWS_TRANSCRIBE_S3_BUCKET", "audio-bucket")
    monkeypatch.setenv("AWS_TRANSCRIBE_ENDPOINT_URL", "https://transcribe.local")
    monkeypatch.setenv("AWS_S3_ENDPOINT_URL", "https://s3.local")

    model = aws_transcribe.AWSTranscribeLM()

    assert model.bucket == "audio-bucket"
    assert factory.call_args_list == [
        call("transcribe", region_name="me-south-1", endpoint_url="https://transcribe.local"),
        call("s3", region_name="me-south-1", endpoint_url="https://s3.local"),
    ]


def test_aws_transcribe_requires_bucket_and_valid_polling(monkeypatch):
    monkeypatch.setattr(
        aws_transcribe, "boto3", SimpleNamespace(client=MagicMock(return_value=MagicMock()))
    )
    monkeypatch.delenv("AWS_TRANSCRIBE_S3_BUCKET", raising=False)
    with pytest.raises(ValueError, match="S3 bucket"):
        aws_transcribe.AWSTranscribeLM()
    with pytest.raises(ValueError, match="poll_interval"):
        aws_transcribe.AWSTranscribeLM(bucket="bucket", poll_interval=-1)


def test_aws_transcribe_uploads_polls_reads_and_cleans_up(monkeypatch):
    model = object.__new__(aws_transcribe.AWSTranscribeLM)
    model.bucket = "bucket"
    model.s3_prefix = "tmp"
    model.language = "ar-SA"
    model.poll_interval = 0
    model.job_timeout = 10
    model.s3_client = MagicMock()
    model.s3_client.get_object.return_value = {
        "Body": io.BytesIO(json.dumps({
            "results": {"transcripts": [{"transcript": "  words  "}]}
        }).encode())
    }
    model.transcribe_client = MagicMock()
    model.transcribe_client.get_transcription_job.side_effect = [
        {"TranscriptionJob": {"TranscriptionJobStatus": "IN_PROGRESS"}},
        {"TranscriptionJob": {"TranscriptionJobStatus": "COMPLETED"}},
    ]
    monkeypatch.setattr(aws_transcribe.uuid, "uuid4", lambda: SimpleNamespace(hex="jobid"))
    monkeypatch.setattr(aws_transcribe.time, "sleep", MagicMock())

    assert model._transcribe_audio(b"wav", 16000) == "words"
    model.s3_client.put_object.assert_called_once_with(
        Bucket="bucket", Key="tmp/balsam-eval-jobid.wav", Body=b"wav", ContentType="audio/wav"
    )
    start = model.transcribe_client.start_transcription_job.call_args.kwargs
    assert start["Media"] == {"MediaFileUri": "s3://bucket/tmp/balsam-eval-jobid.wav"}
    assert start["OutputKey"] == "tmp/balsam-eval-jobid.json"
    assert start["LanguageCode"] == "ar-SA"
    model.transcribe_client.delete_transcription_job.assert_called_once_with(
        TranscriptionJobName="balsam-eval-jobid"
    )
    assert model.s3_client.delete_object.call_args_list == [
        call(Bucket="bucket", Key="tmp/balsam-eval-jobid.wav"),
        call(Bucket="bucket", Key="tmp/balsam-eval-jobid.json"),
    ]


def test_aws_transcribe_failed_job_still_cleans_resources(monkeypatch):
    model = object.__new__(aws_transcribe.AWSTranscribeLM)
    model.bucket = "bucket"
    model.s3_prefix = ""
    model.language = "en-US"
    model.poll_interval = 0
    model.job_timeout = 10
    model.s3_client = MagicMock()
    model.s3_client.delete_object.side_effect = OSError("cleanup unavailable")
    model.transcribe_client = MagicMock()
    model.transcribe_client.get_transcription_job.return_value = {
        "TranscriptionJob": {
            "TranscriptionJobStatus": "FAILED",
            "FailureReason": "bad media",
        }
    }
    monkeypatch.setattr(aws_transcribe.uuid, "uuid4", lambda: SimpleNamespace(hex="failed"))

    with pytest.raises(RuntimeError, match="bad media"):
        model._transcribe_audio(b"wav", 8000)
    model.transcribe_client.delete_transcription_job.assert_called_once()
    assert model.s3_client.delete_object.call_count == 2


def test_aws_transcribe_lm_contract_and_wav_generation():
    model = object.__new__(aws_transcribe.AWSTranscribeLM)
    model._tokenizer_name = "transcribe"
    model._transcribe_audio = MagicMock(side_effect=["one", "two"])

    assert model.tokenizer_name == "transcribe"
    assert model.max_sequence_length == 0
    assert model.batch_size == 1
    assert model.generate_until([request(audio_items=[audio(), audio()]), request()]) == [
        "one two",
        "",
    ]
    wav, rate = model._audio_dict_to_wav_bytes(audio())
    assert wav.startswith(b"RIFF") and rate == 16000
    assert model.loglikelihood([1]) == [(0.0, True)]
    assert model.loglikelihood_rolling([1]) == [0.0]


def test_bedrock_constructor_and_converse_payload(monkeypatch):
    client = MagicMock()
    client.converse.return_value = {
        "output": {"message": {"content": [{"text": " answer "}]}}
    }
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(aws_bedrock, "boto3", SimpleNamespace(client=factory))
    monkeypatch.setenv("BEDROCK_MODEL_ID", "provider.model-v1")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    model = aws_bedrock.BedrockChatLM(top_p=0.9)
    result = model.generate_until([request(
        '[{"role":"system","content":"brief"},{"role":"user","content":"hi"}]',
        {"max_gen_toks": 12, "temperature": 0.2, "until": "END"},
    )])

    assert result == ["answer"]
    factory.assert_called_once_with(
        "bedrock-runtime", region_name="us-east-1", endpoint_url=None
    )
    payload = client.converse.call_args.kwargs
    assert payload["modelId"] == "provider.model-v1"
    assert payload["system"] == [{"text": "brief"}]
    assert payload["messages"] == [{"role": "user", "content": [{"text": "hi"}]}]
    assert payload["inferenceConfig"] == {
        "maxTokens": 12,
        "temperature": 0.2,
        "topP": 0.9,
        "stopSequences": ["END"],
    }


def test_sagemaker_invokes_chat_payload_and_extracts_configured_path(monkeypatch):
    client = MagicMock()
    client.invoke_endpoint.return_value = {
        "Body": io.BytesIO(b'{"choices":[{"message":{"content":" result "}}]}')
    }
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(aws_sagemaker, "boto3", SimpleNamespace(client=factory))
    model = aws_sagemaker.SageMakerChatLM(
        endpoint_name="chat-endpoint", response_path="choices.0.message.content"
    )

    assert model.generate_until([request("hello", {"max_tokens": 9, "stop": ["END"]})]) == [
        "result"
    ]
    invocation = client.invoke_endpoint.call_args.kwargs
    assert invocation["EndpointName"] == "chat-endpoint"
    assert json.loads(invocation["Body"]) == {
        "messages": [{"role": "user", "content": "hello"}],
        "parameters": {"max_new_tokens": 9, "temperature": 0.0, "stop": ["END"]},
    }


def test_sagemaker_default_response_path_and_lm_contract(monkeypatch):
    client = MagicMock()
    client.invoke_endpoint.return_value = {
        "Body": io.BytesIO(b'[{"generated_text":"ok"}]')
    }
    monkeypatch.setattr(
        aws_sagemaker, "boto3", SimpleNamespace(client=MagicMock(return_value=client))
    )
    monkeypatch.setenv("SAGEMAKER_ENDPOINT_NAME", "endpoint")
    model = aws_sagemaker.SageMakerChatLM()

    assert model.generate_until(["prompt"]) == ["ok"]
    assert model.tokenizer_name == "endpoint"
    assert model.loglikelihood([1, 2]) == [(0.0, True), (0.0, True)]
    assert model.loglikelihood_rolling([1, 2]) == [0.0, 0.0]


@pytest.mark.parametrize("module", [aws_transcribe, aws_bedrock, aws_sagemaker])
def test_boto3_optional_import_failure_is_isolated(module, monkeypatch):
    original_import = builtins.__import__

    def isolated_import(name, *args, **kwargs):
        if name == "boto3":
            raise ImportError("optional dependency unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", isolated_import)
    monkeypatch.setattr(lm_registry, "register_model", lambda *_names: lambda cls: cls)
    namespace = runpy.run_path(module.__file__, run_name=f"isolated_{module.__name__}")
    assert namespace["boto3"] is None


@pytest.mark.parametrize(
    ("module", "cls", "kwargs", "message"),
    [
        (aws_transcribe, aws_transcribe.AWSTranscribeLM, {"bucket": "bucket"}, "pip install boto3"),
        (aws_bedrock, aws_bedrock.BedrockChatLM, {"model": "model"}, "pip install boto3"),
        (
            aws_sagemaker,
            aws_sagemaker.SageMakerChatLM,
            {"endpoint_name": "endpoint"},
            "pip install boto3",
        ),
    ],
)
def test_missing_boto3_has_clear_install_error(module, cls, kwargs, message, monkeypatch):
    monkeypatch.setattr(module, "boto3", None)
    with pytest.raises(ImportError, match=message):
        cls(**kwargs)
