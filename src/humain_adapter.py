"""Humain API adapter for LM Evaluation Harness.

This adapter provides a custom implementation for the Humain (iq.humain.com) API,
which hosts the Allam Arabic language models. It uses the OpenAI-compatible format
with Humain-specific authentication.
"""

import os
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from tqdm import tqdm

try:
    import requests
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
except ImportError:
    raise ImportError(
        "Please install lm-eval: pip install lm-eval"
    )

logger = logging.getLogger(__name__)


@register_model("humain")
class HumainLM(LM):
    """
    Humain-specific adapter for the Allam Arabic language models.

    This adapter uses the OpenAI-compatible chat completions format
    with Humain's API endpoint at iq.humain.com.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        retry_timeout: float = 30.0,
        max_retries: int = 3,
        **kwargs,
    ):
        super().__init__()

        # Debug: Log initialization parameters
        logger.info(f"=== HumainLM __init__ ===")
        logger.info(f"model: {model}")
        logger.info(f"model_name: {model_name}")
        logger.info(f"api_key provided: {'YES' if api_key else 'NO'}")
        logger.info(f"base_url: {base_url}")
        logger.info(f"temperature: {temperature}")
        logger.info(f"max_tokens: {max_tokens}")
        logger.info(f"kwargs: {kwargs}")

        # Support both 'model' and 'model_name' parameters
        self.model_name = model or model_name or os.environ.get("MODEL", "allam-2-34b-prod")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry_timeout = retry_timeout
        self.max_retries = max_retries
        self._tokenizer_name = self.model_name

        # Get API key from parameters or environment
        api_key = api_key or os.environ.get("API_KEY") or os.environ.get("HUMAIN_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key provided. Set HUMAIN_API_KEY or API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.api_key = api_key

        # Default Humain base URL
        base_url = base_url or os.environ.get("BASE_URL", "https://iq.humain.com/v1/chat/completions")

        # Ensure base_url has the full endpoint path
        if not base_url.endswith("/chat/completions"):
            if base_url.endswith("/v1"):
                base_url += "/chat/completions"
            elif base_url.endswith("/v1/"):
                base_url += "chat/completions"
            elif not base_url.endswith("/"):
                base_url += "/v1/chat/completions"
            else:
                base_url += "v1/chat/completions"

        self.base_url = base_url
        self._last_request_time = 0
        self._min_request_interval = 1.0  # Minimum seconds between requests

        logger.info(f"✅ Initialized HumainLM with model '{self.model_name}' at {self.base_url}")

    # ---------------------------------------------------------------------
    # Required LM Eval properties
    # ---------------------------------------------------------------------

    @property
    def tokenizer_name(self) -> str:
        return self._tokenizer_name

    @property
    def max_sequence_length(self) -> int:
        """Humain Allam models support up to 32K tokens."""
        return 32768

    @property
    def batch_size(self) -> int:
        return 1

    # ---------------------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------------------

    def _extract_instance_data(self, instance: Any) -> Tuple[str, List[str]]:
        """
        Extract prompt and stop sequences from various instance formats.

        Args:
            instance: Request instance (can be Instance object, tuple, dict, or string)

        Returns:
            Tuple of (prompt, stop_sequences)
        """
        # Handle Instance objects from lm_eval
        if hasattr(instance, "__class__") and instance.__class__.__name__ == "Instance":
            if hasattr(instance, "args"):
                args = instance.args
                if hasattr(args, "prompt"):
                    prompt = args.prompt
                    until = getattr(args, "until", [])
                    if until and not isinstance(until, list):
                        until = [until] if until else []
                    return prompt, until
                if hasattr(args, "context"):
                    return args.context, []
            return str(instance), []

        # Handle tuple format: (prompt, stop_sequences)
        if isinstance(instance, tuple):
            if len(instance) >= 2:
                stop = instance[1]
                if not isinstance(stop, list):
                    stop = [stop] if stop else []
                return instance[0], stop
            return instance[0], []

        # Handle dict format
        if isinstance(instance, dict):
            prompt = instance.get("prompt", instance.get("context", ""))
            stop = instance.get("until", [])
            if not isinstance(stop, list):
                stop = [stop] if stop else []
            return prompt, stop

        # Fallback: convert to string
        return str(instance), []

    def _make_request_with_retry(
        self,
        messages: List[Dict[str, str]],
        stop: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Make a request to Humain API with retry logic.

        Args:
            messages: List of message dicts
            stop: Optional stop sequences
            max_tokens: Optional max tokens override

        Returns:
            Generated text (or empty string on failure)
        """
        final_response = ""

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"API call attempt {attempt + 1}/{self.max_retries}")

                # Rate limiting: ensure minimum time between requests
                current_time = time.time()
                time_since_last_request = current_time - self._last_request_time
                if time_since_last_request < self._min_request_interval:
                    sleep_time = self._min_request_interval - time_since_last_request
                    logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                    time.sleep(sleep_time)
                self._last_request_time = time.time()

                # Prepare request payload
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": max_tokens or self.max_tokens,
                }

                if stop:
                    payload["stop"] = stop[:4]  # Limit to 4 stop sequences

                # Prepare headers - use 'apikey' header (lowercase) as required by Humain
                # User-Agent mimics curl to avoid WAF blocking python-requests
                headers = {
                    "Content-Type": "application/json",
                    "apikey": self.api_key,  # lowercase 'apikey' header, no Bearer prefix
                    "User-Agent": "curl/7.88.1",
                    "Accept": "*/*",
                }

                logger.debug(f"Request payload: {payload}")

                response = requests.post(
                    self.base_url,
                    json=payload,
                    headers=headers,
                    timeout=120,
                )

                # Debug: log the actual response
                logger.info(f"Response status: {response.status_code}")
                logger.info(f"Response text (first 500 chars): {response.text[:500]}")

                # Check if response is HTML (WAF rejection)
                content_type = response.headers.get("content-type", "")
                if "text/html" in content_type:
                    logger.error(f"❌ Request blocked by WAF/Security layer")
                    logger.error(f"Response is HTML instead of JSON: {response.text[:200]}")
                    # Check for common WAF rejection patterns
                    if "Request Rejected" in response.text or "support ID" in response.text.lower():
                        logger.error(f"This appears to be a WAF rejection.")
                        logger.error(f"Possible causes:")
                        logger.error(f"  - Rate limiting (too many requests)")
                        logger.error(f"  - IP blocked or not whitelisted")
                        logger.error(f"  - Invalid or expired API key")
                        logger.error(f"  - Missing required security headers")
                        logger.error(f"  - Geographic restrictions")
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_timeout * (attempt + 2)  # Longer wait for WAF issues
                        logger.warning(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error("All retries failed - WAF is blocking requests")
                        return ""

                # Check for errors
                if response.status_code != 200:
                    logger.warning(
                        f"API returned status {response.status_code}: {response.text[:500]}"
                    )

                    # Specific guidance for 401 errors
                    if response.status_code == 401 and "No API key found" in response.text:
                        logger.error(f"\n{'='*60}")
                        logger.error(f"AUTHENTICATION ERROR: The Humain API is not recognizing your API key.")
                        logger.error(f"\nThis typically means:")
                        logger.error(f"  1. Your API key has not been activated in the Humain dashboard")
                        logger.error(f"  2. Your IP address needs to be whitelisted")
                        logger.error(f"  3. You're using an incorrect API key for this endpoint")
                        logger.error(f"  4. API access requires approval from Humain support")
                        logger.error(f"\nNext steps:")
                        logger.error(f"  - Log in to your Humain account at iq.humain.com")
                        logger.error(f"  - Check API settings and enable API access")
                        logger.error(f"  - Look for IP whitelisting options")
                        logger.error(f"  - Contact Humain support with the request_id from the error")
                        logger.error(f"{'='*60}\n")

                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_timeout * (attempt + 1))
                        continue
                    else:
                        logger.error(f"All retries failed with status {response.status_code}")
                        return ""

                # Parse response - check if it's valid JSON first
                try:
                    response_data = response.json()
                except ValueError as json_err:
                    logger.error(f"Failed to parse JSON response: {json_err}")
                    logger.error(f"Response content-type: {content_type}")
                    logger.error(f"Response body: {response.text[:1000]}")

                    if attempt < self.max_retries - 1:
                        logger.warning("Retrying due to invalid JSON response...")
                        time.sleep(self.retry_timeout * (attempt + 1))
                        continue
                    else:
                        logger.error("All retries failed with invalid JSON response")
                        return ""

                # Extract content from response
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    choice = response_data["choices"][0]
                    if "message" in choice:
                        response_text = choice["message"].get("content", "")
                    else:
                        response_text = choice.get("text", "")
                else:
                    response_text = ""

                if response_text.strip():
                    final_response = response_text
                    logger.debug(f"✅ Got valid response: {len(response_text)} chars")
                    break
                else:
                    if attempt < self.max_retries - 1:
                        logger.warning(f"Empty response, retrying...")
                        time.sleep(self.retry_timeout * (attempt + 1))
                    else:
                        logger.error(f"All retries returned empty response")
                        final_response = ""

            except requests.exceptions.RequestException as e:
                logger.error(f"Request error (attempt {attempt + 1}): {type(e).__name__}: {e}")

                if attempt < self.max_retries - 1:
                    wait_time = self.retry_timeout * (attempt + 1)
                    logger.info(f"⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All attempts failed. Using empty string.")
                    final_response = ""
            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt + 1}): {type(e).__name__}: {e}")

                if attempt < self.max_retries - 1:
                    wait_time = self.retry_timeout * (attempt + 1)
                    logger.info(f"⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All attempts failed. Using empty string.")
                    final_response = ""

        return final_response

    # ---------------------------------------------------------------------
    # Generation methods
    # ---------------------------------------------------------------------

    def generate_until(self, instances: List[Any]) -> List[str]:
        """
        Generate text until stop sequences are encountered.

        This is the main method used by lm_eval for text generation tasks.

        Args:
            instances: List of request instances

        Returns:
            List of generated strings (same length as instances)
        """
        logger.info(f"{'=' * 80}")
        logger.info(f"GENERATE_UNTIL called with {len(instances)} instances")
        logger.info(f"{'=' * 80}")

        results = []

        for instance in tqdm(instances, desc=f"Generating {self.model_name}", unit="req"):
            # Extract prompt and stop sequences
            prompt, stop_seqs = self._extract_instance_data(instance)

            logger.debug(f"Prompt length: {len(prompt)} chars")
            logger.debug(f"Stop sequences: {stop_seqs}")

            # Handle empty prompts
            if not prompt or not prompt.strip():
                logger.warning(f"Empty prompt encountered")
                results.append("")
                continue

            # Create message
            messages = [{"role": "user", "content": prompt}]

            # Make request with retry
            response = self._make_request_with_retry(
                messages=messages,
                stop=stop_seqs if stop_seqs else None
            )

            results.append(response)

        logger.info(f"\n{'=' * 80}")
        logger.info(f"GENERATE_UNTIL COMPLETE")
        logger.info(f"Input requests: {len(instances)}")
        logger.info(f"Output results: {len(results)}")
        logger.info(f"Match: {'✅ YES' if len(results) == len(instances) else '❌ NO'}")
        logger.info(f"{'=' * 80}\n")

        # Ensure 1:1 mapping
        assert len(results) == len(instances), (
            f"Result count mismatch: {len(results)} results for {len(instances)} requests"
        )

        return results

    def greedy_until(self, instances: List[Any]) -> List[str]:
        """
        Greedy generation (same as generate_until with temperature=0).

        Args:
            instances: List of request instances

        Returns:
            List of generated strings
        """
        # For Humain, greedy_until is the same as generate_until when temperature=0
        return self.generate_until(instances)

    # ---------------------------------------------------------------------
    # Loglikelihood (unsupported by Humain)
    # ---------------------------------------------------------------------

    def loglikelihood(self, instances: List[Any]) -> List[Tuple[float, bool]]:
        """
        Humain API does not support loglikelihood computation.
        Returns dummy values to allow evaluation to continue.

        Note: Metrics that require loglikelihood will not work correctly.
        """
        logger.warning(
            f"⚠️  Humain API doesn't support loglikelihood. "
            f"Returning dummy values for {len(instances)} instances. "
            f"Accuracy and perplexity metrics will not be accurate."
        )
        return [(0.0, True) for _ in instances]

    def loglikelihood_rolling(
        self, instances: List[Any]
    ) -> List[List[Tuple[float, bool]]]:
        """
        Humain API does not support rolling loglikelihood computation.
        Returns dummy values.
        """
        logger.warning(
            f"⚠️  Humain API doesn't support loglikelihood_rolling. "
            f"Returning dummy values for {len(instances)} instances."
        )
        return [[(0.0, True)] for _ in instances]

    # ---------------------------------------------------------------------
    # Chat template support
    # ---------------------------------------------------------------------

    def apply_chat_template(
        self,
        chat_history: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Format chat messages into a prompt string for the model.

        Humain uses standard OpenAI-compatible chat format.

        Args:
            chat_history: List of message dicts with 'role' and 'content' keys
            add_generation_prompt: Whether to add a prompt for generation (ignored)

        Returns:
            Formatted chat prompt as a string
        """
        if not chat_history:
            return ""

        # For OpenAI-compatible APIs, we return the messages as-is
        # The actual chat template formatting is handled by the API
        formatted_parts = []
        for message in chat_history:
            role = message.get("role", "user")
            content = message.get("content", "")
            formatted_parts.append(f"{role}: {content}")

        return "\n".join(formatted_parts)

    # ---------------------------------------------------------------------
    # Tokenization (basic implementation)
    # ---------------------------------------------------------------------

    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization fallback."""
        import re
        return [t for t in re.split(r"\s+|[,.!?;:\"()\[\]{}]", text) if t]

    def detokenize(self, tokens: List[str]) -> str:
        """Convert tokens back to text."""
        return " ".join(tokens)

    def token_count(self, instances: List[str]) -> List[int]:
        """Return approximate token counts using simple word counting."""
        return [len(self.tokenize(str(x))) for x in instances]