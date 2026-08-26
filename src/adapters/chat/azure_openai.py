"""Azure OpenAI chat-completion adapter for lm-eval."""

import logging
import os
import time
from typing import Any, Optional, cast

from lm_eval.api.model import LM  # type: ignore[import-untyped]
from lm_eval.api.registry import register_model  # type: ignore[import-untyped]
from openai import AzureOpenAI
from tqdm import tqdm

from src.adapters.chat._provider_utils import (
    chat_template,
    generation_options,
    image_url_parts,
    inject_content_parts,
    input_audio_parts,
    parse_request,
    request_auxiliary,
    response_text,
)

logger = logging.getLogger(__name__)


@register_model("azure-openai")
class AzureOpenAIChatLM(LM):
    """Azure deployment-backed chat completions."""

    MULTIMODAL = True

    def __init__(
        self,
        model: Optional[str] = None,
        deployment: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        endpoint: Optional[str] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_timeout: float = 1.0,
        **_kwargs: Any,
    ) -> None:
        super().__init__()
        resolved_model = (
            deployment
            or azure_deployment
            or model
            or os.getenv("AZURE_OPENAI_DEPLOYMENT")
            or os.getenv("MODEL")
        )
        if not resolved_model:
            raise ValueError("Azure OpenAI deployment is required")
        self.model_name: str = resolved_model
        endpoint = (
            endpoint
            or base_url
            or os.getenv("AZURE_OPENAI_ENDPOINT")
            or os.getenv("BASE_URL")
        )
        if not endpoint:
            raise ValueError("Azure OpenAI endpoint is required")
        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("API_KEY")
        if not api_key:
            raise ValueError("Azure OpenAI API key is required")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max(1, max_retries)
        self.retry_timeout = retry_timeout
        self._tokenizer_name = self.model_name
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint.rstrip("/"),
            api_version=api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            timeout=timeout,
            max_retries=0,
        )

    @property
    def tokenizer_name(self) -> str:
        return self._tokenizer_name

    def _complete(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> str:
        last_error: Exception = RuntimeError("Azure OpenAI returned empty content")
        for attempt in range(self.max_retries):
            try:
                result = self.client.chat.completions.create(
                    model=self.model_name, messages=cast(Any, messages), **options
                )
                text = response_text(result)
                if text:
                    return text
                last_error = RuntimeError("Azure OpenAI returned empty content")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                last_error = exc
            if attempt + 1 < self.max_retries:
                time.sleep(self.retry_timeout * (attempt + 1))
        raise RuntimeError(
            f"Azure OpenAI request failed after {self.max_retries} attempts"
        ) from last_error

    def generate_until(self, requests: list[Any], disable_tqdm: bool = False) -> list[str]:
        results = []
        for instance in tqdm(requests, desc=f"Generating {self.model_name}", disable=disable_tqdm):
            messages, kwargs = parse_request(instance)
            auxiliary = request_auxiliary(instance)
            parts = image_url_parts(auxiliary.get("visual", []))
            parts.extend(input_audio_parts(auxiliary.get("audio", [])))
            if parts:
                messages = inject_content_parts(messages, parts)
            options = generation_options(
                kwargs, temperature=self.temperature, max_tokens=self.max_tokens
            )
            if "max_completion_tokens" in kwargs:
                options.pop("max_tokens", None)
                options["max_completion_tokens"] = kwargs["max_completion_tokens"]
            results.append(self._complete(messages, options))
        return results

    def loglikelihood(self, requests: list[Any]) -> list[tuple[float, bool]]:
        raise NotImplementedError("Azure OpenAI Chat API does not provide prompt loglikelihood")

    def loglikelihood_rolling(self, requests: list[Any]) -> list[float]:
        raise NotImplementedError("Azure OpenAI Chat API does not provide rolling loglikelihood")

    def apply_chat_template(
        self, chat_history: list[dict[str, Any]] | str, add_generation_prompt: bool = True
    ) -> str:
        return chat_template(chat_history)
