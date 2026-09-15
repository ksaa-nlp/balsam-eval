"""Mistral chat-completions adapter for LM Evaluation Harness."""

from typing import Any, cast

from lm_eval.api.registry import register_model  # type: ignore[import-untyped]
from lm_eval.models.openai_completions import (  # type: ignore[import-untyped]
    LocalChatCompletion,
)


@register_model("mistral-chat-completions")
class MistralChatCompletion(LocalChatCompletion):
    """Build Mistral-compatible payloads without lm-eval's default seed."""

    def _create_payload(self, *args: Any, **kwargs: Any) -> dict:
        payload = cast(dict[str, Any], super()._create_payload(*args, **kwargs))
        payload.pop("seed", None)
        return payload
