"""Groq API adapter for LM Evaluation Harness.

This adapter uses the official Groq SDK to interact with Groq Cloud API by:
1. Using the native Groq Python client
2. Cleaning messages to ensure API compatibility
3. Handling Groq-specific limitations (no loglikelihood support)
"""

import os
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from tqdm import tqdm

from groq import Groq
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

logger = logging.getLogger(__name__)


@register_model("groq")
class GroqLM(LM):
    """
    Groq-specific adapter using the official Groq SDK.

    This adapter uses the native Groq client and ensures message compatibility
    with Groq Cloud API.
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
        
        # Support both 'model' and 'model_name' parameters
        self.model_name = model or model_name or os.environ.get("MODEL", "llama-3.3-70b-versatile")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry_timeout = retry_timeout
        self.max_retries = max_retries
        self._tokenizer_name = self.model_name
        
        # Get API key from parameters or environment
        api_key = api_key or os.environ.get("API_KEY") or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key provided. Set GROQ_API_KEY or API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Default Groq base URL
        base_url = base_url or os.environ.get("BASE_URL") or "https://api.groq.com"

        # Clean base_url - remove endpoint paths if present
        # Groq library expects just the base URL (e.g., https://api.groq.com)
        # and will automatically append the correct endpoint path
        base_url = self._clean_base_url(base_url)

        # Initialize Groq client
        self.client = Groq(
            api_key=api_key,
            base_url=base_url
        )
        
        logger.info(f"✅ Initialized GroqLM with model '{self.model_name}' at {base_url}")

    # ---------------------------------------------------------------------
    # Required LM Eval properties
    # ---------------------------------------------------------------------

    @property
    def tokenizer_name(self) -> str:
        return self._tokenizer_name

    @property
    def max_sequence_length(self) -> int:
        """Groq supports up to 32K tokens for most models."""
        return 32768

    @property
    def batch_size(self) -> int:
        return 8

    # ---------------------------------------------------------------------
    # URL cleaning utilities
    # ---------------------------------------------------------------------

    @staticmethod
    def _clean_base_url(base_url: str) -> str:
        """
        Clean the base URL by removing endpoint paths.

        The Groq library expects just the base URL (e.g., https://api.groq.com)
        and will automatically append /openai/v1/chat/completions or other
        endpoint paths as needed. If the user provides a full URL with the
        endpoint path, we strip it out.

        Args:
            base_url: The base URL to clean

        Returns:
            Cleaned base URL without endpoint paths
        """
        # Remove trailing slashes
        base_url = base_url.rstrip('/')

        # List of common endpoint paths that should be stripped
        endpoint_patterns = [
            '/openai/v1/chat/completions',
            '/v1/chat/completions',
            '/chat/completions',
            '/openai/v1',
            '/v1',
            '/openai',
        ]

        # Check if base_url ends with any of these patterns and remove them
        for pattern in endpoint_patterns:
            if base_url.endswith(pattern):
                base_url = base_url[:-len(pattern)]
                base_url = base_url.rstrip('/')  # Remove any trailing slash after removal
                logger.info(f"Removed endpoint path '{pattern}' from base_url")
                break

        return base_url

    # ---------------------------------------------------------------------
    # Message cleaning utilities
    # ---------------------------------------------------------------------

    @staticmethod
    def _clean_message(message: Union[Dict[str, Any], str]) -> Dict[str, str]:
        """
        Clean message to ensure Groq API compatibility.
        
        Groq's API only accepts standard OpenAI format:
        {
            "role": "system" | "user" | "assistant",
            "content": "string"
        }
        
        Any other properties (like 'type', 'name', etc.) will cause errors.
        
        Args:
            message: Message dict or string
            
        Returns:
            Cleaned message dict with only 'role' and 'content'
        """
        if isinstance(message, str):
            return {
                "role": "user",
                "content": message
            }
        
        if not isinstance(message, dict):
            return {
                "role": "user",
                "content": str(message)
            }
        
        # Extract only the supported fields
        role = message.get("role", "user")
        content = message.get("content", "")
        
        # Ensure content is a string (not a list or dict)
        if isinstance(content, (list, dict)):
            # If content is structured (like multimodal), convert to string
            if isinstance(content, list):
                # Extract text from content blocks
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif "text" in item:
                            text_parts.append(item["text"])
                    elif isinstance(item, str):
                        text_parts.append(item)
                content = " ".join(text_parts)
            else:
                # Dict content
                content = str(content)
        
        return {
            "role": role,
            "content": str(content)
        }

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

    # ---------------------------------------------------------------------
    # API request with retry logic
    # ---------------------------------------------------------------------

    def _make_request_with_retry(
        self,
        messages: List[Dict[str, str]],
        stop: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Make a request to Groq API with retry logic.
        
        Args:
            messages: List of message dicts (already cleaned)
            stop: Optional stop sequences
            max_tokens: Optional max tokens override
            
        Returns:
            Generated text (or empty string on failure)
        """
        final_response = ""
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"API call attempt {attempt + 1}/{self.max_retries}")
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                    stop=stop if stop else None,
                )
                
                response_text = response.choices[0].message.content or ""
                
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
                
            except Exception as e:
                logger.error(f"API error (attempt {attempt + 1}): {type(e).__name__}: {e}")
                
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

            # Create cleaned message
            messages = [self._clean_message({"role": "user", "content": prompt})]

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
        # For Groq, greedy_until is the same as generate_until when temperature=0
        return self.generate_until(instances)

    # ---------------------------------------------------------------------
    # Loglikelihood (unsupported by Groq)
    # ---------------------------------------------------------------------

    def loglikelihood(self, instances: List[Any]) -> List[Tuple[float, bool]]:
        """
        Groq API does not support loglikelihood computation.
        Returns dummy values to allow evaluation to continue.
        
        Note: Metrics that require loglikelihood will not work correctly.
        """
        logger.warning(
            f"⚠️  Groq doesn't support loglikelihood. "
            f"Returning dummy values for {len(instances)} instances. "
            f"Accuracy and perplexity metrics will not be accurate."
        )
        return [(0.0, True) for _ in instances]

    def loglikelihood_rolling(
        self, instances: List[Any]
    ) -> List[List[Tuple[float, bool]]]:
        """
        Groq API does not support rolling loglikelihood computation.
        Returns dummy values.
        """
        logger.warning(
            f"⚠️  Groq doesn't support loglikelihood_rolling. "
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

        Groq uses standard chat format, so we join messages with their roles.
        The actual chat template formatting is handled by Groq's API.

        Args:
            chat_history: List of message dicts with 'role' and 'content' keys
            add_generation_prompt: Whether to add a prompt for generation (ignored)

        Returns:
            Formatted chat prompt as a string
        """
        if not chat_history:
            return ""

        # For Groq models, we can return a simple text representation
        # The actual formatting to model-specific templates (like Llama 3's
        # <｜begin▁of▁sentence｜> tags) is handled by Groq's API
        formatted_parts = []
        for message in chat_history:
            role = message.get("role", "user")
            content = message.get("content", "")
            formatted_parts.append(f"{role}: {content}")

        return "\n".join(formatted_parts)

    # ---------------------------------------------------------------------
    # Utility methods
    # ---------------------------------------------------------------------

    def get_model_name(self) -> str:
        """Return the model name being used."""
        return self.model_name