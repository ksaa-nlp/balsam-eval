"""Anthropic API adapter for LM Evaluation Harness with image support.

Extends lm-eval's built-in AnthropicChat to add native image support.
For text-only requests, delegates entirely to the parent implementation.
Anthropic's hosted Messages API does not support audio input.

Dependencies: lm-eval[api], pillow
"""

import asyncio
import base64
import copy
import io
import json
import logging
import os
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, Tuple, cast

import aiohttp
import requests as http_requests
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential
from lm_eval.api.registry import register_model  # type: ignore[import-untyped]
from lm_eval.models.anthropic_llms import AnthropicChat  # type: ignore[import-untyped]
from lm_eval.models.utils import chunks

from src.adapters.chat._provider_utils import validate_generation_results
from src.adapters.chat._retry import is_retryable_error

logger = logging.getLogger(__name__)


class AnthropicNonRetriableError(RuntimeError):
    """Raised when Anthropic returns a non-retriable client error (4xx)."""


class _AnthropicRuntime(Protocol):
    """Runtime members omitted by lm-eval's incomplete type declarations."""

    model: str
    base_url: str
    header: dict
    verify_certificate: bool
    timeout: float
    _max_gen_toks: int
    _seed: int
    _concurrent: int
    _batch_size: int
    max_retries: int
    cache_hook: Any


# ---------------------------------------------------------------------------
# Reasoning model detection
# ---------------------------------------------------------------------------

_REASONING_MODEL_KEYWORDS = (
    "fable",
    "haiku-5",
    "opus-4",
    "opus-5",
    "sonnet-4",
    "sonnet-5",
    "claude-4",
    "claude-5",
    "thinking",
    "reasoning",
)


def _is_reasoning_model(model_name: Optional[str]) -> bool:
    """Check if model string or IS_REASONING env indicates an Anthropic reasoning model."""
    if os.environ.get("IS_REASONING", "").lower() in ("1", "true", "yes"):
        return True
    if not model_name:
        return False
    lower = model_name.lower()
    return any(keyword in lower for keyword in _REASONING_MODEL_KEYWORDS)


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
    temperature: Optional[float],
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
    }
    if temperature is not None:
        payload["temperature"] = temperature
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
        self._force_reasoning: bool = False
        runtime = cast(_AnthropicRuntime, self)
        logger.info(
            "Initialized AnthropicAudioLM with model '%s' at %s",
            runtime.model,
            runtime.base_url,
        )

    @property
    def is_reasoning(self) -> bool:
        """Indicates if reasoning model behavior is active or auto-detected."""
        runtime = cast(_AnthropicRuntime, self)
        return getattr(self, "_force_reasoning", False) or _is_reasoning_model(
            getattr(runtime, "model", None)
        )

    def _create_payload(
        self,
        messages: List[Dict],
        *,
        generate: bool = True,
        gen_kwargs: Optional[dict] = None,
        seed: int = 1234,
        eos: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Build Messages API payload, stripping sampling parameters for reasoning models."""
        payload = super()._create_payload(
            messages,
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=seed,
            eos=eos,
            **kwargs,
        )
        if self.is_reasoning:
            # Modern Claude reasoning models strictly reject temperature, top_p, and top_k
            payload.pop("temperature", None)
            payload.pop("top_p", None)
            payload.pop("top_k", None)
        return payload

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
            return validate_generation_results(result, len(requests), "Anthropic")

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
            temperature = None if self.is_reasoning else gen_kwargs.pop("temperature", 0)
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

            last_error: Exception = RuntimeError("Anthropic returned empty content")
            max_retries = max(1, getattr(runtime, "max_retries", 3))
            for attempt in range(max_retries):
                try:
                    resp = http_requests.post(
                        runtime.base_url,
                        json=payload,
                        headers=runtime.header,
                        verify=runtime.verify_certificate,
                        timeout=runtime.timeout,
                    )
                    # Check for 400 deprecation error to auto-correct
                    if (
                        resp.status_code == 400
                        and "deprecated" in resp.text
                        and any(p in resp.text for p in ("temperature", "top_p", "top_k"))
                    ):
                        logger.warning(
                            "Anthropic visual request returned 400 for model '%s': %s. "
                            "Auto-correcting payload by stripping deprecated parameters and retrying...",
                            runtime.model,
                            resp.text,
                        )
                        self._force_reasoning = True
                        payload.pop("temperature", None)
                        payload.pop("top_p", None)
                        payload.pop("top_k", None)
                        resp = http_requests.post(
                            runtime.base_url,
                            json=payload,
                            headers=runtime.header,
                            verify=runtime.verify_certificate,
                            timeout=runtime.timeout,
                        )
                    resp.raise_for_status()
                    data = resp.json()
                    text = next(
                        (block.get("text", "") for block in data.get("content", [])
                         if block.get("type") == "text"),
                        "",
                    )
                    if text:
                        results.append(text)
                        break
                    last_error = RuntimeError("Anthropic returned empty content")
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    last_error = exc
                    if not is_retryable_error(exc, provider="anthropic"):
                        raise
                if attempt + 1 < max_retries:
                    time.sleep(getattr(runtime, "retry_timeout", 1) * (attempt + 1))
            else:
                raise RuntimeError(
                    f"Anthropic request failed after {max_retries} attempts"
                ) from last_error

        assert len(results) == len(requests), (
            f"Result count mismatch: {len(results)} vs {len(requests)}"
        )
        return validate_generation_results(results, len(requests), "Anthropic")

    # ------------------------------------------------------------------ #
    # Async batch execution and fail-fast retry handling
    # ------------------------------------------------------------------ #

    async def amodel_call(
        self,
        session: aiohttp.ClientSession,
        sem: asyncio.Semaphore,
        messages: Any,
        *,
        generate: bool = True,
        cache_keys: Optional[list] = None,
        ctxlens: Optional[List[int]] = None,
        gen_kwargs: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Any:
        """Call Anthropic API with 400 auto-correction and fail-fast client error handling."""
        runtime = cast(_AnthropicRuntime, self)
        gen_kwargs = copy.deepcopy(gen_kwargs)
        payload = self._create_payload(
            self.create_message(messages),
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=getattr(runtime, "_seed", 1234),
            **kwargs,
        )
        cache_method = "generate_until" if generate else "loglikelihood"
        acquired = await sem.acquire()
        try:
            async with session.post(
                runtime.base_url,
                json=payload,
                headers=runtime.header,
            ) as response:
                if not response.ok:
                    error_text = await response.text()
                    # Check if Anthropic rejected deprecated sampling parameters
                    if (
                        response.status == 400
                        and "deprecated" in error_text
                        and any(p in error_text for p in ("temperature", "top_p", "top_k"))
                    ):
                        logger.warning(
                            "Anthropic API returned 400 for model '%s': %s. "
                            "Auto-correcting payload by stripping deprecated parameters and retrying...",
                            runtime.model,
                            error_text,
                        )
                        self._force_reasoning = True
                        payload.pop("temperature", None)
                        payload.pop("top_p", None)
                        payload.pop("top_k", None)
                        async with session.post(
                            runtime.base_url,
                            json=payload,
                            headers=runtime.header,
                        ) as retry_response:
                            if not retry_response.ok:
                                retry_error_text = await retry_response.text()
                                if 400 <= retry_response.status < 500 and retry_response.status != 429:
                                    raise AnthropicNonRetriableError(
                                        f"Anthropic client error ({retry_response.status}): {retry_error_text}"
                                    )
                                retry_response.raise_for_status()
                            outputs = await retry_response.json()
                    else:
                        # Fail fast on non-retriable 4xx client errors to stop infinite tenacity retries
                        if 400 <= response.status < 500 and response.status != 429:
                            raise AnthropicNonRetriableError(
                                f"Anthropic client error ({response.status}): {error_text}"
                            )
                        response.raise_for_status()
                        outputs = await response.json()
                else:
                    outputs = await response.json()

            answers = (
                self.parse_generations(outputs=outputs)
                if generate
                else self.parse_logprobs(
                    outputs=outputs,
                    tokens=messages,
                    ctxlens=ctxlens,
                )
            )
            if cache_keys:
                for res, cache in zip(answers, cache_keys):
                    self.cache_hook.add_partial(cache_method, cache, res)
            return answers
        except AnthropicNonRetriableError:
            raise
        except BaseException as e:
            logger.error("Anthropic exception: %r, retrying.", e)
            raise e
        finally:
            if acquired:
                sem.release()

    async def get_batched_requests(
        self,
        requests: list,
        cache_keys: list,
        *,
        generate: bool = True,
        ctxlens: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute batched requests, stopping tenacity from retrying client errors."""
        runtime = cast(_AnthropicRuntime, self)
        ctxlens = ctxlens if ctxlens else [None] * len(requests)
        conn = aiohttp.TCPConnector(
            limit=runtime._concurrent, ssl=runtime.verify_certificate
        )
        sem = asyncio.Semaphore(runtime._concurrent)
        async with aiohttp.ClientSession(
            connector=conn, timeout=aiohttp.ClientTimeout(total=runtime.timeout)
        ) as session:
            retry_: Callable[..., Awaitable[Any]] = retry(
                stop=stop_after_attempt(getattr(runtime, "max_retries", 3)),
                wait=wait_exponential(multiplier=0.5, min=1, max=10),
                retry=retry_if_not_exception_type(AnthropicNonRetriableError),
                reraise=True,
                before_sleep=lambda retry_state: logger.info(
                    "Anthropic retry attempt %d", retry_state.attempt_number
                ),
            )(self.amodel_call)
            tasks = [
                asyncio.create_task(
                    retry_(
                        session=session,
                        sem=sem,
                        messages=message,
                        cache_keys=cache_key,
                        generate=generate,
                        ctxlens=ctxlen,
                        **kwargs,
                    )
                )
                for message, cache_key, ctxlen in zip(
                    chunks(requests, n=runtime._batch_size),
                    chunks(cache_keys, n=runtime._batch_size),
                    chunks(ctxlens, n=runtime._batch_size),
                )
            ]
            return await tqdm_asyncio.gather(*tasks, desc="Requesting Anthropic API")

    # ------------------------------------------------------------------ #
    # Loglikelihood stubs
    # ------------------------------------------------------------------ #

    def loglikelihood(
        self, requests: list, **kwargs: Any
    ) -> List[Tuple[float, bool]]:
        raise NotImplementedError(
            "Anthropic Messages API does not support loglikelihood"
        )

    def loglikelihood_rolling(
        self, requests: list, disable_tqdm: bool = False
    ) -> List[float]:
        raise NotImplementedError(
            "Anthropic Messages API does not support loglikelihood_rolling"
        )
