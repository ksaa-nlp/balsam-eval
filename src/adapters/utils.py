"""Utility functions for adapter and URL processing."""

import logging
import os
from urllib.parse import urlsplit, urlunsplit

from src.adapter_config import ASR_ADAPTERS

logger = logging.getLogger(__name__)


def get_max_tokens_config(adapter: str, model_name: str) -> dict:
    """Get adapter-specific max_tokens config based on adapter type and model.

    For thinking/reasoning models, different adapters use different parameter names:
    - OpenAI (o1, o3, GPT-5 series): max_completion_tokens
    - DeepSeek (R1): max_completion_tokens
    - Gemini (2.0 Flash Thinking): max_tokens (standard)
    - Anthropic (extended thinking): max_tokens (standard)

    Args:
        adapter: The adapter type (e.g., "gemini", "groq", "openai-chat-completions")
        model_name: The model name (to detect thinking/reasoning models)

    Returns:
        Dict with the appropriate parameter name and value
        Example: {"max_tokens": 1024} or {"max_completion_tokens": 8192}
    """
    # Check if IS_REASONING environment variable is set to 1
    is_reasoning_env = os.getenv("IS_REASONING", "0").strip() == "1"

    # If IS_REASONING=1, use MAX_TOKENS if exists, otherwise default to 8192
    if is_reasoning_env:
        raw_max_tokens = os.getenv("MAX_TOKENS", "8192")
        try:
            max_tokens = int(raw_max_tokens)
        except ValueError as exc:
            raise ValueError("MAX_TOKENS must be an integer") from exc
        if max_tokens <= 0:
            raise ValueError("MAX_TOKENS must be greater than zero")
        # For OpenAI reasoning models, use max_completion_tokens
        if adapter in ("openai-chat-completions", "openai", "azure-openai"):
            return {"max_completion_tokens": max_tokens}
        return {"max_tokens": max_tokens}

    # Otherwise, use the current logic (IS_REASONING=0 or not set)
    model_lower = model_name.lower()

    # Detect thinking/reasoning models by adapter and model name
    thinking_model_patterns = {
        "openai-chat-completions": [
            "o1-",
            "o3-",
            "o4-",
            "gpt-5",
            "gpt5",
        ],
        "openai": [
            "o1-",
            "o3-",
            "o4-",
            "gpt-5",
            "gpt5",
        ],
        "azure-openai": [
            "o1-",
            "o3-",
            "o4-",
            "gpt-5",
            "gpt5",
        ],
        "local-chat-completions": [
            "deepseek-r1",
            "deepseek-reasoner",
            "r1",
            "qwq",
            "skywork-o1",
            "marco-o1",
        ],
        "gemini": ["thinking", "2.0-flash-thinking"],
        "anthropic-chat-completions": ["extended-thinking"],
        "anthropic": ["extended-thinking"],
    }

    # Check if this is a thinking model for the current adapter
    is_thinking_model = (
        adapter in thinking_model_patterns
        and any(pattern in model_lower for pattern in thinking_model_patterns[adapter])
    )

    # Handle thinking models with model-specific token limits
    if is_thinking_model:
        result = _get_thinking_model_config(adapter, model_lower)
        if result:
            return result

    # ASR adapters don't use max_tokens, return minimal config
    if adapter in ASR_ADAPTERS:
        return {"max_tokens": 1024}

    # Adapter-specific defaults for non-thinking models
    adapter_defaults = {
        "gemini": 1024,
        "groq": 1024,
        "openai-chat-completions": 1024,
        "openai": 1024,
        "anthropic-chat-completions": 1024,
        "anthropic": 1024,
        "cohere": 1024,
        "local-chat-completions": 1024,
    }

    default_value = adapter_defaults.get(adapter, 1024)
    return {"max_tokens": default_value}


def _get_thinking_model_config(adapter: str, model_lower: str) -> dict | None:
    """Get max tokens config for thinking models.

    Args:
        adapter: The adapter type
        model_lower: Lowercase model name

    Returns:
        Dict with appropriate token config, or None if not a thinking model
    """
    if adapter in ("openai-chat-completions", "openai", "azure-openai"):
        # GPT-5.2 supports up to 128,000 output tokens
        if "gpt-5.2" in model_lower or "gpt5.2" in model_lower:
            return {"max_completion_tokens": 128000}
        # GPT-5 and o-series models
        return {"max_completion_tokens": 8192}

    if adapter == "local-chat-completions":
        # DeepSeek R1, QwQ, Skywork-o1, etc.
        if any(pattern in model_lower for pattern in ["deepseek", "r1"]):
            return {"max_completion_tokens": 8192, "max_tokens": 8192}
        return {"max_tokens": 8192}

    if adapter in ("gemini", "anthropic-chat-completions", "anthropic"):
        return {"max_tokens": 8192}

    return None


def convert_anthropic_url(url: str | None) -> str:
    """Resolve an Anthropic Messages endpoint without changing custom hosts.

    Args:
        url: The base URL (can be None, empty, or an Anthropic URL)

    Returns:
        Anthropic Messages API URL

    Examples:
        >>> convert_anthropic_url(None)
        'https://api.anthropic.com/v1/messages'

        >>> convert_anthropic_url("https://api.anthropic.com/v1/messages")
        'https://api.anthropic.com/v1/messages'
    """
    default_url = "https://api.anthropic.com/v1/messages"

    # If URL is None or empty, use default
    if not url or url.strip() == "":
        return default_url

    url = url.strip()

    parsed = urlsplit(url)
    if parsed.hostname == "api.anthropic.com":
        path = parsed.path.rstrip("/")
        if path in {"", "/v1", "/v1/messages", "/v1/chat/completions"}:
            return urlunsplit(parsed._replace(path="/v1/messages"))

    # Custom gateways must remain custom. Rewriting them could leak credentials.
    return url


def process_adapter_and_url(
    adapter: str, base_url: str | None, verbose: bool = True
) -> tuple[str, str | None]:
    """Normalize backend adapter aliases and provider endpoints.

    Args:
        adapter: The adapter type from environment
        base_url: The base URL from environment
        verbose: Whether to log/print the conversion (default: True)

    Returns:
        Tuple of (processed_adapter, processed_base_url)

    Examples:
        >>> process_adapter_and_url("anthropic-chat-completions", None, verbose=False)
        ('anthropic', 'https://api.anthropic.com/v1/messages')

        >>> process_adapter_and_url("openai-chat-completions", "https://api.openai.com",
        ...                         verbose=False)
        ('openai', 'https://api.openai.com')
    """
    if adapter == "anthropic-chat-completions":
        custom_url = base_url.strip() if base_url else ""
        parsed = urlsplit(custom_url) if custom_url else None
        if parsed is not None and parsed.hostname != "api.anthropic.com":
            return "local-chat-completions", custom_url

    if adapter in {"anthropic", "anthropic-chat-completions"}:
        if verbose:
            if adapter == "anthropic-chat-completions":
                log_message = "Routing anthropic-chat-completions to native Anthropic Messages"
                logger.info(log_message)
                print(log_message)

        processed_adapter = "anthropic"
        processed_base_url = convert_anthropic_url(base_url)

        if verbose:
            url_message = f"Using base_url: {processed_base_url}"
            logger.info(url_message)
            print(url_message)

        return processed_adapter, processed_base_url

    if adapter == "openai-chat-completions":
        return "openai", base_url

    return adapter, base_url
