import base64
from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from src.adapters.chat import azure_openai, huggingface_chat


def request(prompt, *, visual=None, audio=None):
    auxiliary = {}
    if visual is not None:
        auxiliary["visual"] = visual
    if audio is not None:
        auxiliary["audio"] = audio
    return SimpleNamespace(args=(prompt, {"max_gen_toks": 8}, auxiliary))


def completion(text="answer"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def image():
    return Image.new("RGB", (2, 1), color=(12, 34, 56))


def audio():
    return {
        "array": np.array([0.0, 0.25, -0.25], dtype=np.float64),
        "sampling_rate": 8000,
    }


def test_azure_combines_image_audio_and_text_in_last_user_turn(monkeypatch):
    client = MagicMock()
    client.chat.completions.create.return_value = completion()
    monkeypatch.setattr(azure_openai, "AzureOpenAI", MagicMock(return_value=client))
    model = azure_openai.AzureOpenAIChatLM(
        deployment="vision-audio", endpoint="https://azure.example", api_key="key"
    )
    prompt = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "describe and transcribe"},
    ]
    original = deepcopy(prompt)

    assert model.generate_until(
        [request(prompt, visual=[image()], audio=[audio()])], disable_tqdm=True
    ) == ["answer"]

    assert prompt == original
    sent = client.chat.completions.create.call_args.kwargs["messages"]
    assert sent[0:2] == prompt[0:2]
    image_part, audio_part, text_part = sent[-1]["content"]
    assert image_part["type"] == "image_url"
    image_url = image_part["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")
    decoded_image = base64.b64decode(image_url.partition(",")[2])
    assert Image.open(BytesIO(decoded_image)).size == (2, 1)
    assert audio_part["type"] == "input_audio"
    assert audio_part["input_audio"]["format"] == "wav"
    assert base64.b64decode(audio_part["input_audio"]["data"]).startswith(b"RIFF")
    assert text_part == {"type": "text", "text": "describe and transcribe"}


def test_azure_multimodal_prompt_without_user_adds_user_turn(monkeypatch):
    client = MagicMock()
    client.chat.completions.create.return_value = completion()
    monkeypatch.setattr(azure_openai, "AzureOpenAI", MagicMock(return_value=client))
    model = azure_openai.AzureOpenAIChatLM(
        deployment="vision", endpoint="https://azure.example", api_key="key"
    )

    model.generate_until(
        [request([{"role": "system", "content": "rules"}], visual=[image()])],
        disable_tqdm=True,
    )

    sent = client.chat.completions.create.call_args.kwargs["messages"]
    assert sent[0] == {"role": "system", "content": "rules"}
    assert sent[1]["role"] == "user"
    assert sent[1]["content"][0]["type"] == "image_url"


def test_huggingface_sends_image_data_url_and_preserves_text(monkeypatch):
    client = MagicMock()
    client.chat_completion.return_value = completion()
    monkeypatch.setattr(
        huggingface_chat, "InferenceClient", MagicMock(return_value=client)
    )
    model = huggingface_chat.HuggingFaceChatLM(model="org/vision", api_key="key")

    assert model.generate_until(
        [request("describe", visual=[image()])], disable_tqdm=True
    ) == ["answer"]

    content = client.chat_completion.call_args.kwargs["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1] == {"type": "text", "text": "describe"}


def test_huggingface_explicitly_rejects_audio_including_combined_input(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(
        huggingface_chat, "InferenceClient", MagicMock(return_value=client)
    )
    model = huggingface_chat.HuggingFaceChatLM(model="org/model", api_key="key")

    with pytest.raises(NotImplementedError, match="input_audio"):
        model.generate_until(
            [request("prompt", visual=[image()], audio=[audio()])],
            disable_tqdm=True,
        )
    client.chat_completion.assert_not_called()


@pytest.mark.parametrize(
    ("module", "model_class", "client_method"),
    [
        (azure_openai, azure_openai.AzureOpenAIChatLM, "azure"),
        (huggingface_chat, huggingface_chat.HuggingFaceChatLM, "hf"),
    ],
)
def test_text_only_request_keeps_string_content(
    module, model_class, client_method, monkeypatch
):
    client = MagicMock()
    if client_method == "azure":
        client.chat.completions.create.return_value = completion()
        monkeypatch.setattr(module, "AzureOpenAI", MagicMock(return_value=client))
        model = model_class(
            deployment="text", endpoint="https://azure.example", api_key="key"
        )
        call = client.chat.completions.create
    else:
        client.chat_completion.return_value = completion()
        monkeypatch.setattr(module, "InferenceClient", MagicMock(return_value=client))
        model = model_class(model="org/text", api_key="key")
        call = client.chat_completion

    assert model.generate_until([request("plain")], disable_tqdm=True) == ["answer"]
    assert call.call_args.kwargs["messages"] == [
        {"role": "user", "content": "plain"}
    ]
