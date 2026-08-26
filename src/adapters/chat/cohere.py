"""Cohere v2 Chat API adapter with image support for LM Evaluation Harness.

Extends lm-eval's built-in LocalChatCompletion (OpenAI-compatible chat format)
to work with Cohere's v2 Chat API.

Dependencies: lm-eval[api], numpy, soundfile
"""

import base64
import copy
import io
import json
import logging
import os
from functools import cached_property
from typing import Any, Dict, List, Optional, Protocol, Tuple, cast

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
from lm_eval.api.registry import register_model  # type: ignore[import-untyped]
from lm_eval.models.openai_completions import (  # type: ignore[import-untyped]
    LocalChatCompletion,
)

logger = logging.getLogger(__name__)


class _LocalChatRuntime(Protocol):
    """Runtime members omitted by lm-eval's incomplete type declarations."""

    model: str
    base_url: str

    def model_call(self, **kwargs: Any) -> Any:
        """Call chat-completions endpoint."""
        raise NotImplementedError

    def parse_generations(self, response: Any) -> List[str]:
        """Extract generated text from endpoint response."""
        raise NotImplementedError


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


def _build_openai_audio_parts(audio_dicts: List[dict]) -> List[dict]:
    """Build OpenAI-compatible input_audio content parts (used by Cohere v2)."""
    return [
        {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}
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


def _inject_audio_into_messages(
    messages: List[dict], audio_parts: List[dict]
) -> List[dict]:
    """Inject audio content parts into the last user message."""
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


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@register_model("cohere")
class CohereAudioLM(LocalChatCompletion):
    """Cohere v2 chat adapter with native image input support.

    Inherits all text-only functionality from lm-eval's LocalChatCompletion
    (tenacity retry and async batching), while translating payload and response
    fields to Cohere's native wire format. Audio transcription uses the
    separate ``cohere-asr`` adapter.
    """

    MULTIMODAL = True

    def __init__(
        self,
        base_url: Optional[str] = None,
        tokenizer_backend: Optional[str] = None,
        tokenized_requests: bool = False,
        **kwargs: Any,
    ):
        base_url = base_url or os.environ.get(
            "BASE_URL", "https://api.cohere.com/v2/chat"
        )
        super().__init__(
            base_url=base_url,  # pyright: ignore[reportCallIssue]
            tokenizer_backend=tokenizer_backend,  # pyright: ignore[reportCallIssue]
            tokenized_requests=tokenized_requests,  # pyright: ignore[reportCallIssue]
            **kwargs,
        )
        runtime = cast(_LocalChatRuntime, self)
        logger.info(
            "Initialized CohereAudioLM with model '%s' at %s",
            runtime.model,
            runtime.base_url,
        )

    # ------------------------------------------------------------------ #
    # Auth — Cohere uses Bearer token with CO_API_KEY
    # ------------------------------------------------------------------ #

    @cached_property
    def api_key(self) -> str:
        key = os.environ.get("CO_API_KEY") or os.environ.get("API_KEY")
        if not key:
            raise ValueError(
                "No API key found. Set CO_API_KEY or API_KEY environment variable."
            )
        return key

    @cached_property
    def header(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _create_payload(
        self,
        messages: List[Dict[str, Any]],
        generate: bool = False,
        gen_kwargs: Optional[dict] = None,
        seed: int = 1234,
        eos: Optional[str] = None,
        **_kwargs: Any,
    ) -> dict:
        """Translate lm-eval generation options to Cohere v2 fields."""
        if not generate:
            raise NotImplementedError("Cohere Chat API does not provide loglikelihood")
        options = copy.deepcopy(gen_kwargs) if gen_kwargs else {}
        options.pop("do_sample", None)
        default_max_tokens = getattr(self, "_max_gen_toks", 4096)
        max_tokens = options.pop(
            "max_tokens", options.pop("max_gen_toks", default_max_tokens)
        )
        stop = options.pop("until", options.pop("stop", None))
        if isinstance(stop, str):
            stop = [stop]
        if eos:
            stop = [*(stop or []), eos]
        if "top_p" in options:
            options["p"] = options.pop("top_p")
        return {
            "model": cast(_LocalChatRuntime, self).model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": options.pop("temperature", 0),
            **({"stop_sequences": [item for item in stop if item]} if stop else {}),
            **options,
        }

    @staticmethod
    def parse_generations(outputs: Any, **_kwargs: Any) -> List[str]:
        """Extract text blocks from Cohere v2 chat responses."""
        responses = outputs if isinstance(outputs, list) else [outputs]
        results = []
        for response in responses:
            content = response.get("message", {}).get("content", [])
            results.append("".join(
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ))
        return results

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def generate_until(
        self, requests: list, disable_tqdm: bool = False
    ) -> List[str]:
        if not requests:
            return []

        if not _has_audio(requests):
            result: List[str] = super().generate_until(
                requests,
                disable_tqdm=disable_tqdm,  # pyright: ignore[reportCallIssue]
            )
            return result
        raise NotImplementedError(
            "Cohere Chat does not accept audio; use the cohere-asr adapter"
        )

    # ------------------------------------------------------------------ #
    # Loglikelihood stubs
    # ------------------------------------------------------------------ #

    def loglikelihood(
        self, requests: list, **kwargs: Any
    ) -> List[Tuple[float, bool]]:
        logger.warning(
            "Cohere Chat API does not support loglikelihood. "
            "Returning dummy values for %d requests.",
            len(requests),
        )
        return [(0.0, True) for _ in requests]

    def loglikelihood_rolling(
        self, requests: list, disable_tqdm: bool = False
    ) -> List[float]:
        logger.warning(
            "Cohere Chat API does not support loglikelihood_rolling. "
            "Returning dummy values for %d requests.",
            len(requests),
        )
        return [0.0 for _ in requests]
