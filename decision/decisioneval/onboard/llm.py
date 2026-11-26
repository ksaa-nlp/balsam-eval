"""The LLM that writes adapter transforms in the onboarding synthesize stage.

Deliberately stdlib: each provider is one HTTPS POST, so there is no SDK to pin.
One generic set of variables configures it, so the same key slot can hold an
Anthropic key or a key for any OpenAI-compatible host.

The LLM is never on the evaluation path, and whatever it proposes must pass the
canary gate before it is trusted.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

LlmProvider = Callable[[str], str]

KEY_ENV = "DECISION_SPEC_LLM_API_KEY"
PROVIDER_ENV = "DECISION_SPEC_LLM_PROVIDER"
MODEL_ENV = "DECISION_SPEC_LLM_MODEL"
BASE_URL_ENV = "DECISION_SPEC_LLM_BASE_URL"
ENV_NAMES = (KEY_ENV, PROVIDER_ENV, MODEL_ENV, BASE_URL_ENV)

ANTHROPIC = "anthropic"
OPENAI = "openai"
DEFAULT_MODELS = {ANTHROPIC: "claude-opus-5-5", OPENAI: "gpt-5.1"}
# Lets a key alone pick its provider, so the backend's existing secret still resolves to Anthropic.
ANTHROPIC_KEY_PREFIX = "sk-ant-"

ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
# On a policy decline the API re-runs the request on a model it picks, in the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
OPENAI_BASE_URL = "https://api.openai.com/v1"

# Thinking counts toward this cap on reasoning models.
MAX_OUTPUT_TOKENS = 16000
TIMEOUT_S = 300.0


class LlmError(Exception):
    """The provider could not be reached or answered unusably."""


class LlmConfigError(LlmError):
    """The LLM variables name something this module cannot build."""


@dataclass
class AnthropicProvider:
    """The Messages API. The key never leaves this object."""

    api_key: str
    model: str = DEFAULT_MODELS[ANTHROPIC]
    max_tokens: int = MAX_OUTPUT_TOKENS
    effort: str = "high"
    timeout_s: float = TIMEOUT_S
    base_url: str = ANTHROPIC_BASE_URL

    def __call__(self, prompt: str) -> str:
        payload = _post_json(
            f"{self.base_url.rstrip('/')}/v1/messages",
            {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "output_config": {"effort": self.effort},
                "fallbacks": "default",
                "messages": [{"role": "user", "content": prompt}],
            },
            {
                "x-api-key": self.api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "anthropic-beta": FALLBACK_BETA,
            },
            model=self.model,
            timeout_s=self.timeout_s,
        )
        return _text_of(payload)


@dataclass
class OpenAICompatibleProvider:
    """Chat Completions on OpenAI or any host that speaks it. The key never leaves this object."""

    api_key: str
    model: str = DEFAULT_MODELS[OPENAI]
    max_tokens: int = MAX_OUTPUT_TOKENS
    timeout_s: float = TIMEOUT_S
    base_url: str = OPENAI_BASE_URL

    def __call__(self, prompt: str) -> str:
        payload = _post_json(
            f"{self.base_url.rstrip('/')}/chat/completions",
            {
                "model": self.model,
                "max_completion_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
            {"authorization": f"Bearer {self.api_key}"},
            model=self.model,
            timeout_s=self.timeout_s,
        )
        return _chat_text_of(payload)


def _post_json(url: str, body: dict[str, Any], headers: dict[str, str], *, model: str, timeout_s: float) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"content-type": "application/json", **headers},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise LlmError(f"{model}: HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        raise LlmError(f"{model}: {exc}") from exc


def _text_of(payload: object) -> str:
    if not isinstance(payload, dict):
        raise LlmError("provider reply is not an object")
    if payload.get("stop_reason") == "refusal":
        raise LlmError(f"the model declined: {(payload.get('stop_details') or {}).get('category')}")
    if payload.get("stop_reason") == "max_tokens":
        raise LlmError("the reply hit max_tokens before it finished")
    blocks = payload.get("content")
    if not isinstance(blocks, list):
        raise LlmError("provider reply has no content blocks")
    parts = [
        block.get("text", "")
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return _non_empty("".join(parts))


def _chat_text_of(payload: object) -> str:
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise LlmError("provider reply has no choices")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    if message.get("refusal"):
        raise LlmError(f"the model declined: {message['refusal']}")
    if choice.get("finish_reason") == "length":
        raise LlmError("the reply hit max_tokens before it finished")
    content = message.get("content")
    return _non_empty(content if isinstance(content, str) else "")


def _non_empty(text: str) -> str:
    text = text.strip()
    if not text:
        raise LlmError("provider returned no text")
    return text


def provider_from_env(environ: dict[str, str] | None = None) -> LlmProvider | None:
    """The configured provider, or None when no key is set.

    An explicit provider wins; otherwise the key's prefix decides.
    """
    environ = dict(os.environ if environ is None else environ)

    def value(name: str) -> str:
        return (environ.get(name) or "").strip()

    key = value(KEY_ENV)
    if not key:
        return None
    name = value(PROVIDER_ENV).lower() or (ANTHROPIC if key.startswith(ANTHROPIC_KEY_PREFIX) else OPENAI)
    if name not in DEFAULT_MODELS:
        raise LlmConfigError(f"{PROVIDER_ENV} must be one of {', '.join(DEFAULT_MODELS)}, got {name!r}")
    model = value(MODEL_ENV) or DEFAULT_MODELS[name]
    base_url = value(BASE_URL_ENV)
    if name == ANTHROPIC:
        return AnthropicProvider(api_key=key, model=model, base_url=base_url or ANTHROPIC_BASE_URL)
    return OpenAICompatibleProvider(api_key=key, model=model, base_url=base_url or OPENAI_BASE_URL)


def extract_json(text: str) -> dict[str, Any] | None:
    """The outermost JSON object in an LLM reply, unwrapped from a `response` key if present."""
    stripped = text.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        block = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(block, dict):
        return None
    return block.get("response") if "response" in block else block
