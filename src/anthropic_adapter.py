"""Anthropic API adapter for LM Evaluation Harness with audio support.

Extends lm-eval's built-in AnthropicChat to add multimodal audio support.
For text-only requests, delegates entirely to the parent implementation.
For audio requests, injects Anthropic-format audio content blocks and uses
the inherited model_call() / parse_generations() HTTP machinery.

Dependencies: lm-eval[api], numpy, soundfile
"""

import base64
import copy
import io
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests as http_requests
import soundfile as sf  # type: ignore[import-untyped]
from tqdm import tqdm

from lm_eval.api.registry import register_model  # type: ignore[import-untyped]
from lm_eval.models.anthropic_llms import AnthropicChat  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio helpers (self-contained, no external src.* imports)
# ---------------------------------------------------------------------------


def _audio_dicts_to_base64_wav(audio_dicts: List[dict]) -> List[str]:
    """Convert lm-eval audio dicts to base64-encoded WAV strings."""
    results = []
    for audio in audio_dicts:
        array = np.array(audio["array"])
        if array.dtype != np.float32:
            array = array.astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, array, audio["sampling_rate"], format="WAV", subtype="PCM_16")
        results.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return results


def _build_anthropic_audio_parts(audio_dicts: List[dict]) -> List[dict]:
    """Build Anthropic-format audio content blocks."""
    return [
        {
            "type": "audio",
            "source": {
                "type": "base64",
                "media_type": "audio/wav",
                "data": b64,
            },
        }
        for b64 in _audio_dicts_to_base64_wav(audio_dicts)
    ]


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
        and "audio" in req.args[2]
        for req in requests
        if hasattr(req, "args")
    )


def _inject_audio_into_anthropic_messages(
    messages: List[dict], audio_parts: List[dict]
) -> List[dict]:
    """Inject audio content blocks into the last user message (Anthropic format).

    Anthropic's Messages API expects content as a list of typed blocks:
      [{"type": "audio", "source": {...}}, {"type": "text", "text": "..."}]
    """
    messages = copy.deepcopy(messages)
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

    last_user["content"] = audio_parts + content
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
@register_model("anthropic-chat-completions")
class AnthropicAudioLM(AnthropicChat):
    """Anthropic Messages API adapter with audio support.

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
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            **kwargs,
        )
        logger.info(
            "Initialized AnthropicAudioLM with model '%s' at %s",
            self.model,
            self.base_url,
        )

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def generate_until(
        self, requests: list, disable_tqdm: bool = False
    ) -> List[str]:
        if not requests:
            return []

        if not _has_audio(requests):
            result: List[str] = super().generate_until(requests, disable_tqdm=disable_tqdm)
            return result

        results: List[str] = []
        for req in tqdm(
            requests,
            desc=f"Generating {self.model}",
            disable=disable_tqdm,
        ):
            prompt_obj = req.args[0]
            gen_kwargs = req.args[1] if len(req.args) > 1 else {}
            aux = req.args[2] if len(req.args) > 2 else {}
            audio_dicts = aux.get("audio") if isinstance(aux, dict) else None

            messages = _parse_chat_prompt(prompt_obj)

            if audio_dicts:
                audio_parts = _build_anthropic_audio_parts(audio_dicts)
                messages = _inject_audio_into_anthropic_messages(
                    messages, audio_parts
                )

            gen_kwargs = copy.deepcopy(gen_kwargs)
            gen_kwargs.pop("do_sample", None)
            max_tokens = gen_kwargs.pop(
                "max_tokens",
                gen_kwargs.pop("max_gen_toks", self._max_gen_toks),
            )
            temperature = gen_kwargs.pop("temperature", 0)
            until = gen_kwargs.pop("until", ["\n\nHuman:"])
            if isinstance(until, str):
                until = [until]

            payload = _messages_to_anthropic_payload(
                messages=messages,
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_sequences=until,
            )

            try:
                resp = http_requests.post(
                    self.base_url,
                    json=payload,
                    headers=self.header,
                    verify=self.verify_certificate,
                    timeout=self.timeout,
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
    ) -> List[List[Tuple[float, bool]]]:
        logger.warning(
            "Anthropic Messages API does not support loglikelihood_rolling. "
            "Returning dummy values for %d requests.",
            len(requests),
        )
        return [[(0.0, True)] for _ in requests]
