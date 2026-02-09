"""Utility functions for adapter and URL processing."""

import logging

logger = logging.getLogger(__name__)


def get_max_tokens_config(adapter: str, model_name: str) -> dict:
    """
    Get adapter-specific max_tokens config based on adapter type and model.

    For thinking/reasoning models, different adapters use different parameter names:
    - OpenAI (o1, o3, GPT-5 series): max_completion_tokens
    - DeepSeek (R1): max_completion_tokens  
    - Gemini (2.0 Flash Thinking): max_tokens (standard)
    - Anthropic (extended thinking): max_tokens (standard)

    Args:
        adapter: The adapter type (e.g., "gemini", "groq", "openai-chat-completions", etc.)
        model_name: The model name (to detect thinking/reasoning models)

    Returns:
        Dict with the appropriate parameter name and value
        Example: {"max_tokens": 4096} or {"max_completion_tokens": 8192}
    """
    model_lower = model_name.lower()
    
    # Detect thinking/reasoning models by adapter and model name
    thinking_model_patterns = {
        "openai-chat-completions": [
            "o1-", "o3-", "o4-",  # o-series models (with dash to avoid false matches)
            "gpt-5",              # Matches gpt-5, gpt-5.1, gpt-5.2, gpt-5-turbo, etc.
            "gpt5",               # Matches gpt5 variants without dash
        ],
        "local-chat-completions": [
            "deepseek-r1", "deepseek-reasoner", "r1", 
            "qwq", 
            "skywork-o1",
            "marco-o1",
        ],
        "gemini": ["thinking", "2.0-flash-thinking"],
        "anthropic-chat-completions": ["extended-thinking"],
    }
    
    # Check if this is a thinking model for the current adapter
    is_thinking_model = False
    if adapter in thinking_model_patterns:
        is_thinking_model = any(
            pattern in model_lower 
            for pattern in thinking_model_patterns[adapter]
        )
    
    # Handle thinking models with model-specific token limits
    if is_thinking_model:
        if adapter in ["openai-chat-completions"]:
            # Determine max tokens based on specific model
            max_tokens = 8192  # Default for most reasoning models
            
            # GPT-5.2 supports up to 128,000 output tokens
            if "gpt-5.2" in model_lower or "gpt5.2" in model_lower:
                max_tokens = 128000
            # GPT-5.1 and GPT-5 base
            elif "gpt-5" in model_lower or "gpt5" in model_lower:
                max_tokens = 8192
            # o-series models
            elif any(x in model_lower for x in ["o1", "o3", "o4"]):
                max_tokens = 8192
                
            return {"max_completion_tokens": max_tokens, "max_tokens": max_tokens}
            
        elif adapter in ["local-chat-completions"]:
            # DeepSeek R1, QwQ, Skywork-o1, etc.
            if any(pattern in model_lower for pattern in ["deepseek", "r1"]):
                return {"max_completion_tokens": 8192, "max_tokens": 8192}
            else:
                # Fallback for other reasoning models
                return {"max_tokens": 8192}
                
        elif adapter in ["gemini"]:
            # Gemini thinking models use standard max_tokens
            return {"max_tokens": 8192}
            
        elif adapter in ["anthropic-chat-completions"]:
            # Anthropic extended thinking uses standard max_tokens
            return {"max_tokens": 8192}
    
    # Adapter-specific defaults for non-thinking models
    adapter_defaults = {
        "gemini": 4096,
        "groq": 4096,
        "openai-chat-completions": 4096,
        "anthropic-chat-completions": 4096,
        "local-chat-completions": 4096,
    }

    default_value = adapter_defaults.get(adapter, 4096)
    return {"max_tokens": default_value}


def convert_anthropic_url(url):
    """
    Convert Anthropic API URL to OpenAI-compatible format.

    Args:
        url: The base URL (can be None, empty, or an Anthropic URL)

    Returns:
        OpenAI-compatible Anthropic URL

    Examples:
        >>> convert_anthropic_url(None)
        'https://api.anthropic.com/v1/chat/completions'

        >>> convert_anthropic_url("https://api.anthropic.com/v1/messages")
        'https://api.anthropic.com/v1/chat/completions'

        >>> convert_anthropic_url("https://api.anthropic.com")
        'https://api.anthropic.com/v1/chat/completions'
    """
    # Default OpenAI-compatible Anthropic endpoint
    default_url = "https://api.anthropic.com/v1/chat/completions"

    # If URL is None or empty, use default
    if not url or url.strip() == "":
        return default_url

    url = url.strip()

    # If it's already the chat/completions endpoint, return as-is
    if "/chat/completions" in url:
        return url

    # Convert /v1/messages to /v1/chat/completions
    if "/v1/messages" in url:
        return url.replace("/v1/messages", "/v1/chat/completions")

    # If it's just the base domain, add the path
    if "api.anthropic.com" in url and "/v1" not in url:
        # Remove trailing slash if present
        url = url.rstrip("/")
        return f"{url}/v1/chat/completions"

    # If it has /v1 but no endpoint, add chat/completions
    if "/v1" in url and not url.endswith("/v1"):
        return url  # Assume it's already properly formatted
    elif url.endswith("/v1"):
        return f"{url}/chat/completions"

    # Fallback to default if we can't parse it
    return default_url


def process_adapter_and_url(adapter, base_url, verbose=True):
    """
    Process adapter and base_url, converting Anthropic to local-chat-completions.

    Args:
        adapter: The adapter type from environment
        base_url: The base URL from environment
        verbose: Whether to log/print the conversion (default: True)

    Returns:
        Tuple of (processed_adapter, processed_base_url)

    Examples:
        >>> process_adapter_and_url("anthropic-chat-completions", None, verbose=False)
        ('local-chat-completions', 'https://api.anthropic.com/v1/chat/completions')

        >>> process_adapter_and_url("openai-chat-completions", "https://api.openai.com", verbose=False)
        ('openai-chat-completions', 'https://api.openai.com')
    """
    if adapter == "anthropic-chat-completions":
        if verbose:
            log_message = (
                "Converting anthropic-chat-completions to local-chat-completions"
            )
            logger.info(log_message)
            print(log_message)

        processed_adapter = "local-chat-completions"
        processed_base_url = convert_anthropic_url(base_url)

        if verbose:
            url_message = f"Using base_url: {processed_base_url}"
            logger.info(url_message)
            print(url_message)

        return processed_adapter, processed_base_url

    return adapter, base_url
