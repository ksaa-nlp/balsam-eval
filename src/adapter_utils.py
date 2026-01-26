"""Utility functions for adapter and URL processing."""

import logging

logger = logging.getLogger(__name__)


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
            log_message = "Converting anthropic-chat-completions to local-chat-completions"
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