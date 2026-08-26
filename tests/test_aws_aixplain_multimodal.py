import base64
import io
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from src.adapters.chat import aixplain, aws_bedrock, aws_sagemaker


def request(prompt="hello", kwargs=None, **auxiliary):
    return SimpleNamespace(args=(prompt, kwargs or {}, auxiliary))


def image():
    return Image.new("RGB", (2, 1), "red")


def audio():
    return {"array": np.array([0.0, 0.25], dtype=np.float32), "sampling_rate": 16000}


def test_bedrock_converse_adds_native_images_and_preserves_roles_and_system():
    model = object.__new__(aws_bedrock.BedrockChatLM)
    model.model_name = "vision-model"
    model.max_tokens = 10
    model.temperature = 0.0
    model.top_p = None
    model.client = MagicMock()
    model.client.converse.return_value = {
        "output": {"message": {"content": [{"text": "seen"}]}}
    }
    prompt = json.dumps([
        {"role": "system", "content": "be precise"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "describe"},
    ])

    assert model.generate_until([request(prompt, visual=[image()])]) == ["seen"]

    payload = model.client.converse.call_args.kwargs
    assert payload["system"] == [{"text": "be precise"}]
    assert [message["role"] for message in payload["messages"]] == [
        "user",
        "assistant",
        "user",
    ]
    assert payload["messages"][0]["content"] == [{"text": "first"}]
    assert payload["messages"][1]["content"] == [{"text": "reply"}]
    final_content = payload["messages"][2]["content"]
    assert final_content[0] == {"text": "describe"}
    assert final_content[1]["image"]["format"] == "png"
    assert final_content[1]["image"]["source"]["bytes"].startswith(b"\x89PNG")


def test_bedrock_rejects_audio_before_calling_converse():
    model = object.__new__(aws_bedrock.BedrockChatLM)
    model.client = MagicMock()

    with pytest.raises(NotImplementedError, match="Converse.*audio"):
        model.generate_until([request(audio=[audio()])])
    model.client.converse.assert_not_called()


def test_sagemaker_passes_combined_openai_multimodal_blocks():
    model = object.__new__(aws_sagemaker.SageMakerChatLM)
    model.endpoint_name = "multimodal-endpoint"
    model.messages_key = "messages"
    model.response_path = "output"
    model.content_type = "application/json"
    model.accept = "application/json"
    model.max_tokens = 10
    model.temperature = 0.0
    model.client = MagicMock()
    model.client.invoke_endpoint.return_value = {
        "Body": io.BytesIO(b'{"output":"ok"}')
    }

    assert model.generate_until([request("inspect", visual=[image()], audio=[audio()])]) == [
        "ok"
    ]

    payload = json.loads(model.client.invoke_endpoint.call_args.kwargs["Body"])
    content = payload["messages"][0]["content"]
    assert [part["type"] for part in content] == ["image_url", "input_audio", "text"]
    image_url = content[0]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")
    assert base64.b64decode(image_url.split(",", 1)[1]).startswith(b"\x89PNG")
    assert base64.b64decode(content[1]["input_audio"]["data"]).startswith(b"RIFF")
    assert content[2] == {"type": "text", "text": "inspect"}


def test_sagemaker_text_payload_remains_unchanged():
    messages = [{"role": "user", "content": "hello"}]
    assert aws_sagemaker._inject_multimodal(messages, {}) is messages


def aixplain_model():
    model = object.__new__(aixplain.AiXplainChatLM)
    model.model_name = "vision-model"
    model.url = "https://models.aixplain.com/api/v2/execute/vision-model"
    model.headers = {"x-api-key": "secret", "Content-Type": "application/json"}
    model.temperature = 0.0
    model.max_tokens = 20
    model.timeout = 5
    model.max_retries = 1
    model.retry_timeout = 0
    model.max_poll_attempts = 1
    model._poll_host = "models.aixplain.com"
    return model


def test_aixplain_uses_documented_data_message_schema_for_images(monkeypatch):
    response = MagicMock()
    response.json.return_value = {"completed": True, "data": "answer"}
    post = MagicMock(return_value=response)
    monkeypatch.setattr(aixplain.http_requests, "post", post)
    model = aixplain_model()

    assert model.generate_until([request("describe", visual=[image()])], disable_tqdm=True) == [
        "answer"
    ]

    payload = post.call_args.kwargs["json"]
    assert "text" not in payload
    assert payload["data"][0]["role"] == "user"
    content = payload["data"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1] == {"type": "text", "text": "describe"}


def test_aixplain_text_payload_is_unchanged_and_audio_is_explicit(monkeypatch):
    response = MagicMock()
    response.json.return_value = {"completed": True, "data": "answer"}
    post = MagicMock(return_value=response)
    monkeypatch.setattr(aixplain.http_requests, "post", post)
    model = aixplain_model()

    assert model.generate_until([request("hello")], disable_tqdm=True) == ["answer"]
    assert post.call_args.kwargs["json"]["text"] == [
        {"role": "user", "content": "hello"}
    ]

    post.reset_mock()
    with pytest.raises(NotImplementedError, match="no reliable.*schema"):
        model.generate_until([request("listen", audio=[audio()])], disable_tqdm=True)
    post.assert_not_called()
