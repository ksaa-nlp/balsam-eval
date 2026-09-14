import base64
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from src.adapters.chat import gemini, groq


def request(prompt, kwargs=None, **auxiliary):
    return SimpleNamespace(args=(prompt, kwargs or {}, auxiliary))


def audio():
    return {"array": np.zeros(8, dtype=np.float32), "sampling_rate": 8000}


def test_gemini_sends_image_and_audio_while_preserving_chat_roles(monkeypatch):
    model = object.__new__(gemini.GeminiLM)
    model.model_name = "models/gemini-test"
    model.temperature = 0
    model.max_tokens = 100
    model.top_p = 0.9
    model.top_k = 20
    model.max_retries = 1
    model.retry_timeout = 0
    generate = MagicMock(return_value=SimpleNamespace(text="answer"))
    model.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    monkeypatch.setattr(model, "_audio_dicts_to_parts", MagicMock(return_value=["audio-part"]))
    monkeypatch.setattr(model, "_visuals_to_parts", MagicMock(return_value=["image-part"]))
    prompt = json.dumps([
        {"role": "system", "content": "Be brief"},
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "Describe inputs"},
    ])

    assert model.generate_until([
        request(
            prompt,
            {"until": "END", "temperature": 0.4, "max_gen_toks": 12},
            visual=[Image.new("RGB", (2, 2), "red")],
            audio=[audio()],
        )
    ]) == ["answer"]

    call = generate.call_args.kwargs
    assert [content.role for content in call["contents"]] == ["user", "model", "user"]
    assert call["contents"][-1].parts[:2] == ["image-part", "audio-part"]
    assert call["config"].system_instruction == "Be brief"
    assert call["config"].temperature == 0.4
    assert call["config"].max_output_tokens == 12
    assert call["config"].stop_sequences == ["END"]


def test_gemini_pil_image_conversion_creates_png_part(monkeypatch):
    from_bytes = MagicMock(return_value="image-part")
    monkeypatch.setattr(gemini.types.Part, "from_bytes", from_bytes)

    assert gemini.GeminiLM._image_to_part(Image.new("RGB", (1, 1))) == "image-part"
    payload = from_bytes.call_args.kwargs
    assert payload["mime_type"] == "image/png"
    assert payload["data"].startswith(b"\x89PNG\r\n\x1a\n")


def test_groq_sends_native_image_url_and_preserves_json_roles():
    model = object.__new__(groq.GroqLM)
    model.model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
    model._make_request_with_retry = MagicMock(return_value="answer")
    prompt = json.dumps([
        {"role": "system", "content": "Be precise"},
        {"role": "user", "content": "Describe this"},
    ])

    assert model.generate_until([
        request(
            prompt,
            {"until": ["END"], "temperature": 0.2, "max_gen_toks": 9, "top_p": 0.8},
            visual=[Image.new("RGB", (2, 2), "blue")],
        )
    ]) == ["answer"]

    call = model._make_request_with_retry.call_args.kwargs
    assert call["messages"][0] == {"role": "system", "content": "Be precise"}
    content = call["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": "Describe this"}
    url = content[1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]).startswith(b"\x89PNG")
    assert call["stop"] == ["END"]
    assert call["max_tokens"] == 9
    assert call["generation_settings"] == {"temperature": 0.2, "top_p": 0.8}


def test_groq_chat_audio_directs_users_to_transcription_adapter():
    model = object.__new__(groq.GroqLM)
    model.model_name = "whisper-large-v3"
    model._make_request_with_retry = MagicMock()

    with pytest.raises(NotImplementedError, match="openai-asr.*Groq.*transcription"):
        model.generate_until([request("Transcribe", audio=[audio()])])
    model._make_request_with_retry.assert_not_called()
