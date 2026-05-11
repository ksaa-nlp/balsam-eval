"""OpenAI API adapter for LM Evaluation Harness with audio support.

Extends lm-eval's built-in OpenAIChatCompletion to add multimodal audio support.
For text-only requests, delegates entirely to the parent implementation (retry,
batching, collation, caching). For audio requests, processes one at a time using
the inherited model_call() and parse_generations() machinery.

Also works as an OpenAI-compatible adapter — pass a custom base_url to target
any OpenAI-compatible endpoint (Together, Fireworks, etc.).

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
import soundfile as sf  # type: ignore[import-untyped]
from tqdm import tqdm

from lm_eval.api.registry import register_model  # type: ignore[import-untyped]
from lm_eval.models.api_models import JsonChatStr  # type: ignore[import-untyped]
from lm_eval.models.openai_completions import (  # type: ignore[import-untyped]
    OpenAIChatCompletion,
)

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


def _build_openai_audio_parts(audio_dicts: List[dict]) -> List[dict]:
    """Build OpenAI-format input_audio content parts."""
    return [
        {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}
        for b64 in _audio_dicts_to_base64_wav(audio_dicts)
    ]


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


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@register_model("openai")
@register_model("openai-chat-completions")
class OpenAIAudioLM(OpenAIChatCompletion):
    """OpenAI chat-completions adapter with audio support.

    Inherits all text-only functionality from lm-eval's OpenAIChatCompletion
    (payload creation, response parsing, tenacity retry, async batching,
    caching). Overrides generate_until() only when audio is present.
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
            "BASE_URL", "https://api.openai.com/v1/chat/completions"
        )
        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )
        logger.info(
            "Initialized OpenAIAudioLM with model '%s' at %s",
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
                audio_parts = _build_openai_audio_parts(audio_dicts)
                messages = _inject_audio_into_messages(messages, audio_parts)

            chat_str = JsonChatStr(json.dumps(messages))
            try:
                response = self.model_call(
                    messages=[chat_str],
                    generate=True,
                    gen_kwargs=copy.deepcopy(gen_kwargs),
                )
                parsed = self.parse_generations(response)
                results.append(parsed[0] if parsed else "")
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("Generation error: %s", e)
                results.append("")

        assert len(results) == len(requests), (
            f"Result count mismatch: {len(results)} vs {len(requests)}"
        )
        return results

    # ------------------------------------------------------------------ #
    # Loglikelihood stubs (API does not expose token logprobs)
    # ------------------------------------------------------------------ #

    def loglikelihood(
        self, requests: list, **kwargs: Any
    ) -> List[Tuple[float, bool]]:
        logger.warning(
            "OpenAI Chat API does not support loglikelihood. "
            "Returning dummy values for %d requests.",
            len(requests),
        )
        return [(0.0, True) for _ in requests]

    def loglikelihood_rolling(
        self, requests: list, disable_tqdm: bool = False
    ) -> List[List[Tuple[float, bool]]]:
        logger.warning(
            "OpenAI Chat API does not support loglikelihood_rolling. "
            "Returning dummy values for %d requests.",
            len(requests),
        )
        return [[(0.0, True)] for _ in requests]
