"""Anthropic API adapter for LM Evaluation Harness with image support.

Extends lm-eval's built-in AnthropicChat to add native image support.
For text-only requests, delegates entirely to the parent implementation.
Anthropic's hosted Messages API does not support audio input.

Dependencies: lm-eval[api], pillow
"""

import base64
import copy
import io
import json
import logging
import os
from typing import Any, Dict, List, Optional, Protocol, Tuple, cast

import requests as http_requests
from tqdm import tqdm

from lm_eval.api.registry import register_model  # type: ignore[import-untyped]
from lm_eval.models.anthropic_llms import AnthropicChat  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class _AnthropicRuntime(Protocol):
    """Runtime members omitted by lm-eval's incomplete type declarations."""

    model: str
    base_url: str
    header: dict
    verify_certificate: bool
    timeout: float
    _max_gen_toks: int


# ---------------------------------------------------------------------------
# Multimodal helpers (self-contained, no external src.* imports)
# ---------------------------------------------------------------------------


def _build_anthropic_image_parts(images: List[Any]) -> List[dict]:
    """Encode PIL-compatible images as Anthropic image content blocks."""
    parts = []
    for image in images:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        parts.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(buf.getvalue()).decode("ascii"),
            },
        })
    return parts


def _parse_chat_prompt(prompt_obj: Any) -> List[Dict[str, Any]]:
    """Parse a prompt object (JsonChatStr or string) into a chat message list."""
    if hasattr(prompt_obj, "prompt"):
        try:
            parsed: List[Dict[str, Any]] = json.loads(prompt_obj.prompt)
            return parsed
        except (json.JSONDecodeError, TypeError):
            return [{"role": "user", "content": str(prompt_obj.prompt)}]
    if isinstance(prompt_obj, str):
        try:
            parsed = json.loads(prompt_obj)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        return [{"role": "user", "content": prompt_obj}]
    return [{"role": "user", "content": str(prompt_obj)}]


def _has_audio(requests: list) -> bool:
    """Check if any request contains audio data in auxiliary_args."""
    return any(
        len(req.args) > 2
        and isinstance(req.args[2], dict)
        and bool(req.args[2].get("audio"))
        for req in requests
        if hasattr(req, "args")
    )


def _has_visual(requests: list) -> bool:
    """Check if any request contains image data in auxiliary_args."""
    return any(
        len(req.args) > 2
        and isinstance(req.args[2], dict)
        and bool(req.args[2].get("visual"))
        for req in requests
        if hasattr(req, "args")
    )


def _inject_images_into_anthropic_messages(
    messages: List[dict], image_parts: List[dict]
) -> List[dict]:
    """Inject native image blocks into the last user message."""
    messages = copy.deepcopy(messages)
    if not messages:
        messages.append({"role": "user", "content": []})

    last_user = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user = msg
            break
    if last_user is None:
        last_user = messages[-1]

    content = last_user.get("content", "")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    elif not isinstance(content, list):
        content = [{"type": "text", "text": str(content)}]

    last_user["content"] = image_parts + content
    last_user.pop("type", None)
    return messages


def _messages_to_anthropic_payload(
    messages: List[dict],
    model: str,
    max_tokens: int,
    temperature: float,
    stop_sequences: List[str],
) -> dict:
    """Build an Anthropic Messages API payload from chat messages.

    Extracts system message (if first message has role=system) and converts
    remaining messages to Anthropic content-block format.
    """
    system = None
    if messages and messages[0].get("role") == "system":
        system = messages[0].get("content", "")
        messages = messages[1:]

    cleaned = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            cleaned.append({"role": role, "content": content})
        else:
            cleaned.append({
                "role": role,
                "content": [{"type": "text", "text": str(content)}],
            })

    stop_sequences = [s for s in stop_sequences if s and s.strip()]

    payload: Dict[str, Any] = {
        "model": model,
        "messages": cleaned,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if stop_sequences:
        payload["stop_sequences"] = stop_sequences
    if system:
        payload["system"] = system
    return payload


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@register_model("anthropic")
# @register_model("anthropic-chat-completions")
class AnthropicAudioLM(AnthropicChat):
    """Anthropic Messages API adapter with native image support.

    Inherits all text-only functionality from lm-eval's AnthropicChat
    (payload creation, header/auth, response parsing, tenacity retry).
    Overrides generate_until() only when audio is present.
    """

    MULTIMODAL = True

    def __init__(
        self,
        base_url: Optional[str] = None,
        tokenizer_backend: Optional[str] = None,
        **kwargs: Any,
    ):
        base_url = base_url or os.environ.get(
            "BASE_URL", "https://api.anthropic.com/v1/messages"
        )
        super().__init__(
            base_url=base_url,  # pyright: ignore[reportCallIssue]
            tokenizer_backend=tokenizer_backend,  # pyright: ignore[reportCallIssue]
            **kwargs,
        )
        runtime = cast(_AnthropicRuntime, self)
        logger.info(
            "Initialized AnthropicAudioLM with model '%s' at %s",
            runtime.model,
            runtime.base_url,
        )

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def generate_until(
        self, requests: list, disable_tqdm: bool = False
    ) -> List[str]:
        if not requests:
            return []

        if _has_audio(requests):
            raise NotImplementedError(
                "Anthropic Messages API does not support audio input"
            )

        if not _has_visual(requests):
            result: List[str] = super().generate_until(
                requests,
                disable_tqdm=disable_tqdm,  # pyright: ignore[reportCallIssue]
            )
            return result

        runtime = cast(_AnthropicRuntime, self)
        results: List[str] = []
        for req in tqdm(
            requests,
            desc=f"Generating {runtime.model}",
            disable=disable_tqdm,
        ):
            prompt_obj = req.args[0]
            gen_kwargs = req.args[1] if len(req.args) > 1 else {}
            aux = req.args[2] if len(req.args) > 2 else {}
            images = aux.get("visual") if isinstance(aux, dict) else None

            messages = _parse_chat_prompt(prompt_obj)

            if images:
                image_parts = _build_anthropic_image_parts(images)
                messages = _inject_images_into_anthropic_messages(
                    messages, image_parts
                )

            gen_kwargs = copy.deepcopy(gen_kwargs)
            gen_kwargs.pop("do_sample", None)
            max_tokens = gen_kwargs.pop(
                "max_tokens",
                gen_kwargs.pop("max_gen_toks", runtime._max_gen_toks),
            )
            temperature = gen_kwargs.pop("temperature", 0)
            until = gen_kwargs.pop("until", ["\n\nHuman:"])
            if isinstance(until, str):
                until = [until]

            payload = _messages_to_anthropic_payload(
                messages=messages,
                model=runtime.model,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_sequences=until,
            )

            try:
                resp = http_requests.post(
                    runtime.base_url,
                    json=payload,
                    headers=runtime.header,
                    verify=runtime.verify_certificate,
                    timeout=runtime.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                text = ""
                for block in data.get("content", []):
                    if block.get("type") == "text":
                        text = block.get("text", "")
                        break
                results.append(text)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("Anthropic generation error: %s", e)
                results.append("")

        assert len(results) == len(requests), (
            f"Result count mismatch: {len(results)} vs {len(requests)}"
        )
        return results

    # ------------------------------------------------------------------ #
    # Loglikelihood stubs
    # ------------------------------------------------------------------ #

    def loglikelihood(
        self, requests: list, **kwargs: Any
    ) -> List[Tuple[float, bool]]:
        logger.warning(
            "Anthropic Messages API does not support loglikelihood. "
            "Returning dummy values for %d requests.",
            len(requests),
        )
        return [(0.0, True) for _ in requests]

    def loglikelihood_rolling(
        self, requests: list, disable_tqdm: bool = False
    ) -> List[float]:
        logger.warning(
            "Anthropic Messages API does not support loglikelihood_rolling. "
            "Returning dummy values for %d requests.",
            len(requests),
        )
        return [0.0 for _ in requests]
