import base64
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from src.adapters.chat import anthropic, local, openai


def _audio():
    return {"array": np.array([0.0, 0.25]), "sampling_rate": 8000}


def _request(prompt, *, visual=None, audio=None):
    aux = {}
    if visual is not None:
        aux["visual"] = visual
    if audio is not None:
        aux["audio"] = audio
    return SimpleNamespace(args=(prompt, {}, aux))


@pytest.mark.parametrize(
    ("module", "cls"),
    [(openai, openai.OpenAIAudioLM), (local, local.LocalAudioLM)],
)
def test_openai_style_combines_image_audio_and_text(module, cls):
    model = object.__new__(cls)
    model.model = "model"
    model.model_call = MagicMock(return_value={})
    model.parse_generations = MagicMock(return_value=["answer"])

    result = model.generate_until(
        [_request("describe", visual=[Image.new("RGB", (1, 1))], audio=[_audio()])],
        disable_tqdm=True,
    )

    assert result == ["answer"]
    messages = json.loads(model.model_call.call_args.kwargs["messages"][0].prompt)
    content = messages[-1]["content"]
    assert [part["type"] for part in content] == ["image_url", "input_audio", "text"]
    image_url = content[0]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")
    assert base64.b64decode(image_url.split(",", 1)[1]).startswith(b"\x89PNG")
    audio_data = content[1]["input_audio"]["data"]
    assert not audio_data.startswith("data:")
    assert base64.b64decode(audio_data).startswith(b"RIFF")
    assert content[2] == {"type": "text", "text": "describe"}


@pytest.mark.parametrize("module", [openai, local])
def test_openai_style_empty_messages_accept_media(module):
    part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}}
    assert module._inject_media_into_messages([], [part]) == [
        {"role": "user", "content": [part]}
    ]


def test_anthropic_image_generation_uses_native_block(monkeypatch):
    model = object.__new__(anthropic.AnthropicAudioLM)
    model.model = "claude"
    model.base_url = "https://api.anthropic.com/v1/messages"
    model.header = {}
    model.verify_certificate = True
    model.timeout = 5
    model._max_gen_toks = 16
    response = MagicMock()
    response.json.return_value = {"content": [{"type": "text", "text": "answer"}]}
    post = MagicMock(return_value=response)
    monkeypatch.setattr(anthropic.http_requests, "post", post)

    assert model.generate_until(
        [_request("describe", visual=[Image.new("RGB", (1, 1))])],
        disable_tqdm=True,
    ) == ["answer"]
    content = post.call_args.kwargs["json"]["messages"][0]["content"]
    assert [part["type"] for part in content] == ["image", "text"]
    assert content[0]["source"]["media_type"] == "image/png"
    assert base64.b64decode(content[0]["source"]["data"]).startswith(b"\x89PNG")


def test_anthropic_empty_messages_accept_image():
    part = {"type": "image", "source": {}}
    assert anthropic._inject_images_into_anthropic_messages([], [part]) == [
        {"role": "user", "content": [part]}
    ]


def test_anthropic_rejects_audio_before_network(monkeypatch):
    parent = MagicMock()
    post = MagicMock()
    monkeypatch.setattr(anthropic.AnthropicChat, "generate_until", parent)
    monkeypatch.setattr(anthropic.http_requests, "post", post)

    with pytest.raises(NotImplementedError, match="does not support audio input"):
        object.__new__(anthropic.AnthropicAudioLM).generate_until(
            [_request("listen", audio=[_audio()])], disable_tqdm=True
        )
    parent.assert_not_called()
    post.assert_not_called()


@pytest.mark.parametrize(
    ("cls", "parent"),
    [
        (openai.OpenAIAudioLM, openai.OpenAIChatCompletion),
        (local.LocalAudioLM, local.LocalChatCompletion),
        (anthropic.AnthropicAudioLM, anthropic.AnthropicChat),
    ],
)
def test_empty_auxiliary_media_preserves_text_delegation(cls, parent, monkeypatch):
    generate = MagicMock(return_value=["text answer"])
    monkeypatch.setattr(parent, "generate_until", generate)
    request = _request("text", visual=[], audio=[])

    assert object.__new__(cls).generate_until([request], disable_tqdm=True) == [
        "text answer"
    ]
    generate.assert_called_once_with([request], disable_tqdm=True)
