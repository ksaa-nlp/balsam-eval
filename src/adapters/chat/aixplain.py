"""aiXplain Models REST API chat adapter for lm-eval."""

import base64
import copy
import io
import os
import time
from typing import Any, Optional, cast
from urllib.parse import quote, urlparse

import requests as http_requests
from lm_eval.api.model import LM  # type: ignore[import-untyped]
from lm_eval.api.registry import register_model  # type: ignore[import-untyped]
from PIL import Image
from tqdm import tqdm

from src.adapters.chat._provider_utils import (
    chat_template,
    generation_options,
    parse_request,
    response_text,
)


def _image_data_url(image: Image.Image) -> str:
    output = io.BytesIO()
    converted = image
    image_format = (image.format or "png").lower()
    image_format = {"jpg": "jpeg"}.get(image_format, image_format)
    if image_format not in {"png", "jpeg", "gif", "webp"}:
        image_format = "png"
    if image_format == "jpeg" and image.mode not in {"RGB", "L"}:
        converted = image.convert("RGB")
    elif image_format == "png" and image.mode not in {"1", "L", "LA", "P", "RGB", "RGBA"}:
        converted = image.convert("RGBA" if "A" in image.getbands() else "RGB")
    converted.save(output, format=image_format.upper())
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/{image_format};base64,{encoded}"


def _inject_images(
    messages: list[dict[str, Any]], images: list[Image.Image]
) -> list[dict[str, Any]]:
    result = copy.deepcopy(messages)
    target = next((message for message in reversed(result) if message.get("role") == "user"), None)
    if target is None:
        target = {"role": "user", "content": []}
        result.append(target)
    content = target.get("content", "")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    elif not isinstance(content, list):
        content = [{"type": "text", "text": str(content)}]
    image_parts = [
        {"type": "image_url", "image_url": {"url": _image_data_url(image)}}
        for image in images
    ]
    target["content"] = image_parts + content
    return result


def _auxiliary(instance: Any) -> dict[str, Any]:
    if hasattr(instance, "args") and len(instance.args) > 2:
        value = instance.args[2]
    elif isinstance(instance, tuple) and len(instance) > 2:
        value = instance[2]
    elif isinstance(instance, dict):
        value = instance.get("auxiliary_args", {})
    else:
        value = {}
    return value if isinstance(value, dict) else {}


@register_model("aixplain")
class AiXplainChatLM(LM):
    """Chat adapter using aiXplain's model execution REST endpoint."""

    MULTIMODAL = True

    def __init__(
        self,
        model: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_timeout: float = 1.0,
        max_poll_attempts: int = 120,
        **_kwargs: Any,
    ) -> None:
        super().__init__()
        self.model_name = (
            model
            or model_name
            or os.getenv("AIXPLAIN_MODEL_ID")
            or os.getenv("MODEL")
        )
        if not self.model_name:
            raise ValueError("aiXplain model ID is required")
        api_key = (
            api_key
            or os.getenv("AIXPLAIN_API_KEY")
            or os.getenv("TEAM_API_KEY")
            or os.getenv("API_KEY")
        )
        if not api_key:
            raise ValueError("aiXplain API key is required")
        root = (
            base_url
            or os.getenv("AIXPLAIN_BASE_URL")
            or os.getenv("BASE_URL")
            or "https://models.aixplain.com"
        )
        root = root.rstrip("/")
        if root.endswith("/api/v2/execute"):
            self.url = f"{root}/{quote(self.model_name, safe='')}"
        elif "/api/v2/execute/" in root:
            self.url = root
        else:
            self.url = f"{root}/api/v2/execute/{quote(self.model_name, safe='')}"
        self._poll_host = urlparse(self.url).hostname
        self.headers = {"x-api-key": api_key, "Content-Type": "application/json"}
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.retry_timeout = retry_timeout
        self.max_poll_attempts = max(1, max_poll_attempts)
        self._tokenizer_name = self.model_name

    @property
    def tokenizer_name(self) -> str:
        return self._tokenizer_name

    @staticmethod
    def _payload_text(payload: Any) -> str:
        if not isinstance(payload, dict):
            return response_text(payload)
        data = payload.get("data")
        if isinstance(data, dict):
            text = response_text(data)
            if text:
                return text
        if isinstance(data, str) and not data.startswith(("http://", "https://")):
            return data
        text = response_text(payload.get("details"))
        if text:
            return text
        return response_text(payload)

    def _poll(self, url: str) -> dict[str, Any]:
        parsed = urlparse(url)
        trusted_host = (
            parsed.hostname == self._poll_host
            or bool(parsed.hostname and parsed.hostname.endswith(".aixplain.com"))
        )
        if parsed.scheme != "https" or not trusted_host or parsed.username or parsed.password:
            raise ValueError("aiXplain returned an untrusted polling URL")
        for attempt in range(self.max_poll_attempts):
            response = http_requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            payload = cast(dict[str, Any], response.json())
            status = str(payload.get("status", "")).upper()
            if status == "FAILED":
                raise RuntimeError(
                    payload.get("error")
                    or payload.get("message")
                    or "aiXplain job failed"
                )
            if payload.get("completed") is True or status == "SUCCESS":
                return payload
            if attempt + 1 < self.max_poll_attempts:
                time.sleep(self.retry_timeout)
        raise TimeoutError("aiXplain result polling timed out")

    def _complete(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
        *,
        multimodal: bool = False,
    ) -> str:
        payload = {"data" if multimodal else "text": messages, **options}
        last_error: Exception = RuntimeError("aiXplain returned empty content")
        for attempt in range(self.max_retries):
            try:
                response = http_requests.post(
                    self.url, headers=self.headers, json=payload, timeout=self.timeout
                )
                response.raise_for_status()
                body = response.json()
                status = str(body.get("status", "")).upper()
                if status == "FAILED":
                    raise RuntimeError(
                        body.get("error")
                        or body.get("message")
                        or "aiXplain request failed"
                    )
                data = body.get("data")
                if (
                    body.get("completed") is not True
                    and isinstance(data, str)
                    and data.startswith(("http://", "https://"))
                ):
                    body = self._poll(data)
                text = self._payload_text(body)
                if text:
                    return text
                last_error = RuntimeError("aiXplain returned empty content")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                last_error = exc
            if attempt + 1 < self.max_retries:
                time.sleep(self.retry_timeout * (attempt + 1))
        raise RuntimeError(
            f"aiXplain request failed after {self.max_retries} attempts"
        ) from last_error

    def generate_until(self, requests: list[Any], disable_tqdm: bool = False) -> list[str]:
        results = []
        for instance in tqdm(requests, desc=f"Generating {self.model_name}", disable=disable_tqdm):
            messages, kwargs = parse_request(instance)
            auxiliary = _auxiliary(instance)
            if auxiliary.get("audio"):
                raise NotImplementedError(
                    "aiXplain chat audio input has no reliable model-independent execution schema"
                )
            images = auxiliary.get("visual") or []
            if images:
                messages = _inject_images(messages, images)
            results.append(
                self._complete(
                    messages,
                    generation_options(
                        kwargs, temperature=self.temperature, max_tokens=self.max_tokens
                    ),
                    multimodal=bool(images),
                )
            )
        return results

    def loglikelihood(self, requests: list[Any]) -> list[tuple[float, bool]]:
        raise NotImplementedError("aiXplain Models API does not provide prompt loglikelihood")

    def loglikelihood_rolling(self, requests: list[Any]) -> list[float]:
        raise NotImplementedError("aiXplain Models API does not provide rolling loglikelihood")

    def apply_chat_template(
        self, chat_history: list[dict[str, Any]] | str, add_generation_prompt: bool = True
    ) -> str:
        return chat_template(chat_history)
