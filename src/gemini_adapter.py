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
    # Generation with GUARANTEED 1:1 Mapping
    # ---------------------------------------------------------------------

    def generate_until(self, instances: List[Any]) -> List[str]:
        """
        Generate text until stop sequences are encountered.
        """
        logger.info(f"=" * 80)
        logger.info(f"GENERATE_UNTIL called with {len(instances)} instances")
        logger.info(f"=" * 80)
        
        results = []

        for idx, instance in enumerate(instances):
            logger.info(f"\n--- Processing instance {idx + 1}/{len(instances)} ---")
            logger.debug(f"Instance type: {type(instance)}")
            
            prompt, stop_seqs = self._extract_instance_data(instance)
            
            logger.info(f"Prompt length: {len(prompt)} chars")
            logger.info(f"Stop sequences: {stop_seqs}")
            logger.debug(f"Prompt preview: {prompt[:200]}..." if len(prompt) > 200 else f"Prompt: {prompt}")

            # Handle empty prompts immediately
            if not prompt:
                logger.warning(f"❌ Empty prompt encountered at index {idx}")
                results.append("")  # ALWAYS append
                continue

            # DEFAULT: Start with empty string
            # This guarantees we have something to append even if all retries fail
            final_response = ""
            
            # Try multiple times
            for attempt in range(self.max_retries):
                try:
                    logger.debug(f"API call attempt {attempt + 1}/{self.max_retries} for instance {idx}")
                    
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=self._gen_config(stop_seqs),
                    )

                    # Extract text or use empty string
                    response_text = response.text if response.text else ""
                    
                    logger.debug(f"Response received: {len(response_text)} chars")
                    logger.debug(f"Response preview: {response_text[:200]}..." if len(response_text) > 200 else f"Response: {response_text}")
                    
                    # If we got a non-empty response, use it and stop trying
                    if response_text.strip() != "":
                        final_response = response_text
                        logger.info(f"✅ Got valid response at index {idx} on attempt {attempt + 1}")
                        break  # Got a good response, exit retry loop
                    
                    # Empty response - should we retry?
                    if attempt < self.max_retries - 1:
                        logger.warning(
                            f"⚠️  Empty response at index {idx}, "
                            f"attempt {attempt + 1}/{self.max_retries}. Retrying..."
                        )
                        time.sleep(self.retry_timeout * (attempt + 1))
                        # Don't update final_response, keep it as ""
                        continue
                    else:
                        # Last attempt and still empty
                        logger.error(
                            f"❌ All {self.max_retries} attempts returned empty response "
                            f"for request {idx}. Using empty string."
                        )
                        final_response = ""  # Explicitly set to empty
                        break

                except Exception as e:
                    logger.error(
                        f"❌ Generation error at index {idx}, "
                        f"attempt {attempt + 1}/{self.max_retries}: {type(e).__name__}: {e}"
                    )
                    
                    # If this is NOT the last attempt, retry
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_timeout * (attempt + 1)
                        logger.info(f"⏳ Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # Last attempt failed with exception
                        logger.error(
                            f"❌ All {self.max_retries} attempts failed for request {idx}. "
                            f"Last error: {e}. Returning empty string."
                        )
                        final_response = ""  # Explicitly set to empty
                        break
            
            # CRITICAL: ALWAYS append exactly one result per request
            # No conditions, no exceptions, no matter what happened above
            results.append(final_response)
            logger.info(f"✅ Result {idx} added to results list (total: {len(results)})")

        # CRITICAL: Verify count matches to prevent reordering assertion errors
        logger.info(f"\n" + "=" * 80)
        logger.info(f"GENERATE_UNTIL COMPLETE")
        logger.info(f"Input instances: {len(instances)}")
        logger.info(f"Output results: {len(results)}")
        
        if len(results) != len(instances):
            logger.error(f"❌ CRITICAL MISMATCH - THIS WILL CAUSE ASSERTION ERROR!")
            logger.error(f"Expected {len(instances)} results, got {len(results)}")
            # Emergency padding to prevent crash
            while len(results) < len(instances):
                results.append("")
                logger.error(f"Emergency padding: added empty string (total now: {len(results)})")
            logger.info(f"Match after padding: ✅ YES")
        else:
            logger.info(f"Match: ✅ YES - Perfect 1:1 mapping!")
        
        logger.info(f"=" * 80 + "\n")
        
        assert len(results) == len(instances), (
            f"Result count mismatch: {len(results)} results for {len(instances)} instances. "
            f"This mismatch will cause reordering assertion errors in lm_eval."
        )
        
        return results

    def greedy_until(self, requests: List[Any]) -> List[str]:
        """
        Generate text greedily (temperature=0) until stop sequences.
        
        CRITICAL FIX: Always returns exactly one result per request.
        No exceptions - guaranteed 1:1 mapping to prevent assertion errors.
        """
        logger.info(f"=" * 80)
        logger.info(f"GREEDY_UNTIL called with {len(requests)} requests")
        logger.info(f"=" * 80)
        
        results = []

        for idx, req in enumerate(requests):
            logger.info(f"\n--- Processing greedy request {idx + 1}/{len(requests)} ---")
            logger.debug(f"Request type: {type(req)}")
            
            prompt, stop_seqs = self._extract_instance_data(req)
            
            logger.info(f"Prompt length: {len(prompt)} chars")
            logger.info(f"Stop sequences: {stop_seqs}")
            logger.debug(f"Prompt preview: {prompt[:200]}..." if len(prompt) > 200 else f"Prompt: {prompt}")
            
            # Handle empty prompts immediately
            if not prompt:
                logger.warning(f"❌ Empty prompt encountered at index {idx}")
                results.append("")  # ALWAYS append
                continue

            # DEFAULT: Start with empty string
            final_response = ""
            
            # Try multiple times
            for attempt in range(self.max_retries):
                try:
                    logger.debug(f"Greedy API call attempt {attempt + 1}/{self.max_retries} for request {idx}")
                    
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
                    
                    response_text = response.text if response.text else ""
                    
                    logger.debug(f"Greedy response received: {len(response_text)} chars")
                    logger.debug(f"Response preview: {response_text[:200]}..." if len(response_text) > 200 else f"Response: {response_text}")
                    
                    # If we got a non-empty response, use it and stop trying
                    if response_text.strip() != "":
                        final_response = response_text
                        logger.info(f"✅ Got valid greedy response at index {idx} on attempt {attempt + 1}")
                        break
                    
                    # Empty response - should we retry?
                    if attempt < self.max_retries - 1:
                        logger.warning(
                            f"⚠️  Empty greedy response at index {idx}, "
                            f"attempt {attempt + 1}/{self.max_retries}. Retrying..."
                        )
                        time.sleep(self.retry_timeout * (attempt + 1))
                        continue
                    else:
                        logger.error(
                            f"❌ All {self.max_retries} attempts returned empty response "
                            f"for greedy request {idx}. Using empty string."
                        )
                        final_response = ""
                        break
                    
                except Exception as e:
                    logger.error(
                        f"❌ Greedy generation error at index {idx}, "
                        f"attempt {attempt + 1}/{self.max_retries}: {type(e).__name__}: {e}"
                    )
                    
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_timeout * (attempt + 1)
                        logger.info(f"⏳ Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"❌ All attempts failed for greedy request {idx}. Using empty string.")
                        final_response = ""
                        break

            # CRITICAL: ALWAYS append exactly one result per request
            results.append(final_response)
            logger.info(f"✅ Greedy result {idx} added to results list (total: {len(results)})")

        logger.info(f"\n" + "=" * 80)
        logger.info(f"GREEDY_UNTIL COMPLETE")
        logger.info(f"Input requests: {len(requests)}")
        logger.info(f"Output results: {len(results)}")
        
        if len(results) != len(requests):
            logger.error(f"❌ CRITICAL MISMATCH!")
            # Emergency padding
            while len(results) < len(requests):
                results.append("")
                logger.error(f"Emergency padding: {len(results)}")
            logger.info(f"Match after padding: ✅ YES")
        else:
            logger.info(f"Match: ✅ YES - Perfect 1:1 mapping!")
        
        logger.info(f"=" * 80 + "\n")

        assert len(results) == len(requests), (
            f"Result count mismatch: {len(results)} results for {len(requests)} requests. "
            f"This mismatch will cause reordering assertion errors in lm_eval."
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
        logger.info(f"LOGLIKELIHOOD called with {len(instances)} instances (returning dummy values)")
        return [(0.0, True) for _ in instances]

    def loglikelihood_rolling(
        self, instances: List[Any]
    ) -> List[List[Tuple[float, bool]]]:
        """
        Gemini API does not support rolling loglikelihood computation.
        Returns dummy values.
        """
        logger.info(f"LOGLIKELIHOOD_ROLLING called with {len(instances)} instances (returning dummy values)")
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
        
        Includes retry logic for empty responses.
        """
        stop_seqs = None
        if stop:
            stop_seqs = [stop] if isinstance(stop, str) else stop

        final_response = ""
        
        for attempt in range(self.max_retries):
            try:
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

                response_text = response.text or ""
                
                # Retry on empty response
                if response_text.strip() == "":
                    if attempt < self.max_retries - 1:
                        logger.warning(
                            f"Empty completion response, "
                            f"attempt {attempt + 1}/{self.max_retries}. Retrying..."
                        )
                        time.sleep(self.retry_timeout * (attempt + 1))
                        continue
                    else:
                        logger.error(
                            f"All {self.max_retries} attempts returned empty response. "
                            "Returning empty string."
                        )
                        final_response = ""
                        break
                else:
                    final_response = response_text
                    break
                    
            except Exception as e:
                logger.warning(f"Completion error, attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_timeout * (attempt + 1))
                else:
                    logger.error(f"All attempts failed. Last error: {e}. Returning empty string.")
                    final_response = ""
                    break
        
        return final_response