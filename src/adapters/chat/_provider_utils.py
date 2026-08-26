"""Small shared helpers for provider-backed chat adapters."""

import base64
import copy
import io
import json
from typing import Any

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
from PIL import Image


def request_auxiliary(instance: Any) -> dict[str, Any]:
    """Read lm-eval multimodal data from request's third argument."""
    if hasattr(instance, "args"):
        args = instance.args
    elif isinstance(instance, tuple):
        args = instance
    else:
        return {}
    return dict(args[2]) if len(args) > 2 and isinstance(args[2], dict) else {}


def image_url_parts(images: list[Any]) -> list[dict[str, Any]]:
    """Encode PIL images as OpenAI-compatible PNG data URL blocks."""
    parts = []
    for image in images:
        if not isinstance(image, Image.Image):
            raise TypeError("visual auxiliary inputs must be PIL images")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encoded}"},
        })
    return parts


def input_audio_parts(audios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Encode lm-eval audio dictionaries as raw-base64 WAV blocks."""
    parts = []
    for audio in audios:
        array = np.asarray(audio["array"], dtype=np.float32)
        buffer = io.BytesIO()
        sf.write(
            buffer,
            array,
            audio["sampling_rate"],
            format="WAV",
            subtype="PCM_16",
        )
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        parts.append({
            "type": "input_audio",
            "input_audio": {"data": encoded, "format": "wav"},
        })
    return parts


def inject_content_parts(
    messages: list[dict[str, Any]], parts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Prepend content blocks to last user turn without mutating prompt."""
    result = copy.deepcopy(messages)
    user_message = next(
        (message for message in reversed(result) if message.get("role") == "user"),
        None,
    )
    if user_message is None:
        user_message = {"role": "user", "content": []}
        result.append(user_message)

    content = user_message.get("content", "")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    elif not isinstance(content, list):
        content = [{"type": "text", "text": str(content)}]
    user_message["content"] = parts + content
    user_message.pop("type", None)
    return result


def parse_messages(value: Any) -> list[dict[str, Any]]:
    """Convert lm-eval prompts and dataset chat JSON into provider messages."""
    if hasattr(value, "prompt"):
        value = value.prompt

    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            decoded = None
        if isinstance(decoded, list):
            value = decoded
        else:
            return [{"role": "user", "content": value}]

    if isinstance(value, dict):
        if isinstance(value.get("messages"), list):
            value = value["messages"]
        elif "role" in value:
            value = [value]
        else:
            return [{"role": "user", "content": str(value)}]

    if not isinstance(value, list):
        return [{"role": "user", "content": str(value)}]

    messages = []
    for item in value:
        if isinstance(item, dict):
            message = dict(item)
            message.setdefault("role", "user")
            message.setdefault("content", "")
            messages.append(message)
        else:
            messages.append({"role": "user", "content": str(item)})
    return messages


def parse_request(instance: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read an lm-eval Instance plus useful direct-call fallback forms."""
    if hasattr(instance, "args"):
        args = instance.args
        prompt = args[0] if args else ""
        kwargs = args[1] if len(args) > 1 and isinstance(args[1], dict) else {}
    elif isinstance(instance, tuple):
        prompt = instance[0] if instance else ""
        kwargs = instance[1] if len(instance) > 1 and isinstance(instance[1], dict) else {}
    elif isinstance(instance, dict) and "role" not in instance:
        prompt = instance.get("messages", instance.get("prompt", instance.get("text", "")))
        kwargs = instance.get("gen_kwargs", {})
        if "until" in instance and "until" not in kwargs:
            kwargs = {**kwargs, "until": instance["until"]}
    else:
        prompt, kwargs = instance, {}
    return parse_messages(prompt), dict(kwargs)


def generation_options(
    kwargs: dict[str, Any], *, temperature: float, max_tokens: int
) -> dict[str, Any]:
    """Map lm-eval generation names to common chat-completion names."""
    options: dict[str, Any] = {
        "temperature": kwargs.get("temperature", temperature),
        "max_tokens": kwargs.get("max_tokens", kwargs.get("max_gen_toks", max_tokens)),
    }
    until = kwargs.get("until")
    if isinstance(until, str):
        until = [until]
    if until:
        options["stop"] = until
    for name in ("top_p", "frequency_penalty", "presence_penalty", "seed"):
        if name in kwargs:
            options[name] = kwargs[name]
    if kwargs.get("do_sample") is False and "temperature" not in kwargs:
        options["temperature"] = 0.0
    return options


def response_text(response: Any) -> str:  # pylint: disable=too-many-return-statements
    """Extract text from OpenAI-compatible object or mapping responses."""
    if isinstance(response, str):
        return response
    if isinstance(response, list):
        for item in response:
            text = response_text(item)
            if text:
                return text
        return ""
    if isinstance(response, dict):
        choices = response.get("choices")
        if choices:
            return response_text(choices[0])
        message = response.get("message")
        if message is not None:
            return response_text(message)
        for key in ("content", "text", "output", "result"):
            if response.get(key) is not None:
                return response_text(response[key])
        return ""
    choices = getattr(response, "choices", None)
    if choices:
        return response_text(choices[0])
    message = getattr(response, "message", None)
    if message is not None:
        return response_text(message)
    for name in ("content", "text", "output", "result"):
        value = getattr(response, name, None)
        if value is not None:
            return response_text(value)
    return ""


def chat_template(chat_history: list[dict[str, Any]] | str) -> str:
    """Preserve role structure for later parsing by generate_until."""
    if isinstance(chat_history, str):
        return chat_history
    return json.dumps(chat_history, ensure_ascii=False)
