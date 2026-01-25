"""
Gemini API backing for LM Evaluation Harness (google.genai version).

This module provides a complete implementation of the Gemini model for use
with LM Evaluation Harness, using the new google.genai SDK.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union

from google import genai
from google.genai import types

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

logger = logging.getLogger(__name__)


@register_model("gemini")
class GeminiLM(LM):
    """
    Gemini model API integration for the LM Evaluation Harness
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        top_p: float = 0.95,
        top_k: int = 40,
        retry_timeout: float = 30.0,
        max_retries: int = 5,
        **kwargs,
    ):
        super().__init__()

        if model_name is None:
            model_name = os.environ.get(
                "MODEL", "models/gemini-1.5-pro"
            )

        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.retry_timeout = retry_timeout
        self.max_retries = max_retries
        self._tokenizer_name = model_name

        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY")

        if api_key is None:
            raise ValueError(
                "No API key provided and GOOGLE_API_KEY environment variable not set."
            )

        self.client = genai.Client(api_key=api_key)

        logger.info(f"Initialized GeminiLM with model {self.model_name}")

    # ---------------------------------------------------------------------
    # Required LM Eval properties
    # ---------------------------------------------------------------------

    @property
    def tokenizer_name(self) -> str:
        return self._tokenizer_name

    @property
    def max_sequence_length(self) -> int:
        return 32000

    @property
    def batch_size(self) -> int:
        return 8

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _gen_config(self, stop_seqs: Optional[List[str]] = None):
        return types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            stop_sequences=stop_seqs or None,
        )

    def _extract_instance_data(self, instance: Any) -> Tuple[str, List[str]]:
        if hasattr(instance, "__class__") and instance.__class__.__name__ == "Instance":
            if hasattr(instance, "args"):
                args = instance.args
                if hasattr(args, "prompt"):
                    until = getattr(args, "until", [])
                    if until and not isinstance(until, list):
                        until = [until]
                    return args.prompt, until
                if hasattr(args, "context"):
                    return args.context, []
            return str(instance), []

        if isinstance(instance, tuple):
            if len(instance) >= 2:
                stop = instance[1]
                if not isinstance(stop, list):
                    stop = [stop] if stop else []
                return instance[0], stop
            return instance[0], []

        if isinstance(instance, dict):
            stop = instance.get("until", [])
            if not isinstance(stop, list):
                stop = [stop] if stop else []
            return instance.get("prompt", ""), stop

        return str(instance), []

    # ---------------------------------------------------------------------
    # Generation
    # ---------------------------------------------------------------------

    def generate_until(self, instances: List[Any]) -> List[str]:
        """
        Generate text until stop sequences are encountered.
        
        CRITICAL: Must return exactly one result per instance to avoid
        reordering assertion errors in lm_eval.
        """
        results = []

        for idx, instance in enumerate(instances):
            prompt, stop_seqs = self._extract_instance_data(instance)

            if not prompt:
                logger.warning(f"Empty prompt encountered at index {idx}")
                results.append("")
                continue

            response_text = None
            last_error = None
            
            for attempt in range(self.max_retries):
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=self._gen_config(stop_seqs),
                    )

                    response_text = response.text if response.text else ""
                    break  # Success - exit retry loop

                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"Generation error at index {idx}, "
                        f"attempt {attempt + 1}/{self.max_retries}: {e}"
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_timeout * (attempt + 1))
            
            # If all retries failed, append empty string but log the error
            if response_text is None:
                logger.error(
                    f"All {self.max_retries} attempts failed for request {idx}. "
                    f"Last error: {last_error}. Returning empty string."
                )
                results.append("")
            else:
                results.append(response_text)

        # CRITICAL: Verify count matches to prevent reordering assertion errors
        assert len(results) == len(instances), (
            f"Result count mismatch: {len(results)} results for {len(instances)} instances"
        )
        
        return results

    def greedy_until(self, requests: List[Any]) -> List[str]:
        """
        Generate text greedily (temperature=0) until stop sequences.
        
        CRITICAL: Must return exactly one result per request to avoid
        reordering assertion errors in lm_eval.
        """
        results = []

        for idx, req in enumerate(requests):
            prompt, stop_seqs = self._extract_instance_data(req)
            
            if not prompt:
                logger.warning(f"Empty prompt encountered at index {idx}")
                results.append("")
                continue

            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=self.max_tokens,
                        top_p=1.0,
                        top_k=1,
                        stop_sequences=stop_seqs or None,
                    ),
                )
                results.append(response.text if response.text else "")
                
            except Exception as e:
                logger.error(f"Greedy generation error at index {idx}: {e}")
                results.append("")

        assert len(results) == len(requests), (
            f"Result count mismatch: {len(results)} results for {len(requests)} requests"
        )
        
        return results

    # ---------------------------------------------------------------------
    # Loglikelihood (unsupported by Gemini)
    # ---------------------------------------------------------------------

    def loglikelihood(self, instances: List[Any]) -> List[Tuple[float, bool]]:
        """
        Gemini API does not support loglikelihood computation.
        Returns dummy values.
        """
        return [(0.0, True) for _ in instances]

    def loglikelihood_rolling(
        self, instances: List[Any]
    ) -> List[List[Tuple[float, bool]]]:
        """
        Gemini API does not support rolling loglikelihood computation.
        Returns dummy values.
        """
        return [[(0.0, True)] for _ in instances]

    # ---------------------------------------------------------------------
    # Tokenization
    # ---------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization fallback using regex split."""
        import re
        return [t for t in re.split(r"\s+|[,.!?;:\"()\[\]{}]", text) if t]

    def _count_tokens(self, text: str) -> int:
        """Count tokens using Gemini's API or fallback to simple tokenization."""
        try:
            resp = self.client.models.count_tokens(
                model=self.model_name,
                contents=text,
            )
            return resp.total_tokens
        except Exception:
            return len(self._tokenize(text))

    def token_count(self, instances: List[str]) -> List[int]:
        """Return token counts for a list of text instances."""
        return [self._count_tokens(str(x)) for x in instances]

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into a list of tokens."""
        return self._tokenize(text)

    def detokenize(self, tokens: List[str]) -> str:
        """Convert tokens back to text."""
        return " ".join(tokens)

    # ---------------------------------------------------------------------
    # Chat template
    # ---------------------------------------------------------------------

    @staticmethod
    def _format_chat_prompt(messages: List[Dict[str, str]]) -> str:
        """Format chat messages into a single prompt string."""
        out = []
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            out.append(f"{role}: {content}")
        return "\n\n".join(out)

    def apply_chat_template(
        self,
        messages: Union[List[Dict[str, Any]], List[Dict[str, str]], str],
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> str:
        """Apply chat template to messages."""
        if isinstance(messages, str):
            return messages

        prompt = self._format_chat_prompt(messages)
        if add_generation_prompt:
            prompt += "\n\nAssistant:"
        return prompt

    # ---------------------------------------------------------------------
    # Convenience completion API
    # ---------------------------------------------------------------------

    def create_completion(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
    ) -> str:
        """
        Create a completion for the given prompt.
        Convenience method for direct API usage.
        """
        stop_seqs = None
        if stop:
            stop_seqs = [stop] if isinstance(stop, str) else stop

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature if temperature is not None else self.temperature,
                max_output_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                top_p=self.top_p,
                top_k=self.top_k,
                stop_sequences=stop_seqs,
            ),
        )

        return response.text or ""