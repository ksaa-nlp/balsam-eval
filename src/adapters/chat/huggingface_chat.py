"""Hugging Face InferenceClient chat-completion adapter for lm-eval."""

import os
import time
from typing import Any, Optional

from lm_eval.api.model import LM  # type: ignore[import-untyped]
from lm_eval.api.registry import register_model  # type: ignore[import-untyped]
from tqdm import tqdm

from src.adapters.chat._provider_utils import (
    chat_template,
    generation_options,
    image_url_parts,
    inject_content_parts,
    parse_request,
    request_auxiliary,
    response_text,
)

try:
    from huggingface_hub import InferenceClient
except ImportError:  # pragma: no cover - exercised only without optional extra
    InferenceClient = None  # type: ignore[assignment,misc]


@register_model("huggingface-chat")
class HuggingFaceChatLM(LM):
    """Chat adapter for Hugging Face Inference Providers and Endpoints."""

    MULTIMODAL = True

    def __init__(
        self,
        model: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        token: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_timeout: float = 1.0,
        **_kwargs: Any,
    ) -> None:
        super().__init__()
        if InferenceClient is None:
            raise ImportError(
                "huggingface-chat requires optional dependency 'huggingface_hub'"
            )
        self.model_name = model or model_name or os.getenv("MODEL")
        if not self.model_name:
            raise ValueError("Hugging Face model is required")
        api_key = (
            api_key
            or token
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            or os.getenv("API_KEY")
        )
        base_url = base_url or os.getenv("HF_BASE_URL") or os.getenv("BASE_URL")
        provider = provider or os.getenv("HF_INFERENCE_PROVIDER", "auto")
        client_kwargs: dict[str, Any] = {"token": api_key, "timeout": timeout}
        if base_url:
            client_kwargs["base_url"] = base_url
        else:
            client_kwargs["provider"] = provider
        self.client = InferenceClient(**client_kwargs)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max(1, max_retries)
        self.retry_timeout = retry_timeout
        self._tokenizer_name = self.model_name

    @property
    def tokenizer_name(self) -> str:
        return self._tokenizer_name

    def _complete(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> str:
        last_error: Exception = RuntimeError("Hugging Face returned empty content")
        for attempt in range(self.max_retries):
            try:
                result = self.client.chat_completion(
                    messages=messages, model=self.model_name, **options
                )
                text = response_text(result)
                if text:
                    return text
                last_error = RuntimeError("Hugging Face returned empty content")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                last_error = exc
            if attempt + 1 < self.max_retries:
                time.sleep(self.retry_timeout * (attempt + 1))
        raise RuntimeError(
            f"Hugging Face request failed after {self.max_retries} attempts"
        ) from last_error

    def generate_until(self, requests: list[Any], disable_tqdm: bool = False) -> list[str]:
        results = []
        for instance in tqdm(requests, desc=f"Generating {self.model_name}", disable=disable_tqdm):
            messages, kwargs = parse_request(instance)
            auxiliary = request_auxiliary(instance)
            if auxiliary.get("audio"):
                raise NotImplementedError(
                    "Hugging Face chat completion does not support input_audio blocks"
                )
            parts = image_url_parts(auxiliary.get("visual", []))
            if parts:
                messages = inject_content_parts(messages, parts)
            results.append(
                self._complete(
                    messages,
                    generation_options(
                        kwargs, temperature=self.temperature, max_tokens=self.max_tokens
                    ),
                )
            )
        return results

    def loglikelihood(self, requests: list[Any]) -> list[tuple[float, bool]]:
        raise NotImplementedError(
            "Hugging Face chat completion does not provide prompt loglikelihood"
        )

    def loglikelihood_rolling(self, requests: list[Any]) -> list[float]:
        raise NotImplementedError(
            "Hugging Face chat completion does not provide rolling loglikelihood"
        )

    def apply_chat_template(
        self, chat_history: list[dict[str, Any]] | str, add_generation_prompt: bool = True
    ) -> str:
        return chat_template(chat_history)
