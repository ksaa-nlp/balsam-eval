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
import io
import base64
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


@register_model("gemini")
class GeminiLM(LM):
    """
    Gemini model API integration for the LM Evaluation Harness
    """

    MULTIMODAL = True

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        top_k: int = 40,
        retry_timeout: float = 30.0,
        max_retries: int = 3,
        **kwargs,
    ):
        super().__init__()

        if model_name is None:
            model_name = os.environ.get("MODEL", "models/gemini-1.5-pro")

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

        self.client = genai.Client(
            api_key=api_key,
            http_options={"timeout": 120_000},
        )

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

    def _extract_instance_data(
        self, instance: Any
    ) -> Tuple[str, List[str], Optional[List[dict]]]:
        """
        Returns (prompt, stop_seqs, audio_dicts_or_None).
        audio_dicts items are {"array": np.ndarray, "sampling_rate": int}.
        """
        audio = None

        if hasattr(instance, "args"):
            args = instance.args
            # Multimodal: args is (prompt_obj, gen_kwargs, auxiliary_args)
            if len(args) >= 3:
                aux = args[2]
                audio = aux.get("audio") if isinstance(aux, dict) else None

            prompt_obj = args[0] if args else instance
            gen_kwargs = args[1] if len(args) > 1 else {}
            until = gen_kwargs.get("until", []) if isinstance(gen_kwargs, dict) else []
            if isinstance(until, str):
                until = [until]

            prompt_str = (
                prompt_obj.prompt if hasattr(prompt_obj, "prompt") else str(prompt_obj)
            )
            return prompt_str, until, audio

        # Fallback for plain tuple / dict
        if isinstance(instance, tuple):
            stop = instance[1] if len(instance) >= 2 else []
            if not isinstance(stop, list):
                stop = [stop] if stop else []
            return instance[0], stop, None

        if isinstance(instance, dict):
            stop = instance.get("until", [])
            if not isinstance(stop, list):
                stop = [stop] if stop else []
            return instance.get("prompt", ""), stop, None

        return str(instance), [], None

    def _audio_dicts_to_parts(self, audio_dicts: List[dict]) -> list:
        """Convert lm_eval audio dicts → google.genai Part objects."""
        parts = []
        for audio in audio_dicts:
            array = np.array(audio["array"])
            if array.dtype != np.float32:
                array = array.astype(np.float32)
            buf = io.BytesIO()
            sf.write(buf, array, audio["sampling_rate"], format="WAV", subtype="PCM_16")
            parts.append(
                types.Part.from_bytes(
                    data=buf.getvalue(),
                    mime_type="audio/wav",
                )
            )
        return parts

    # ---------------------------------------------------------------------
    # Generation with GUARANTEED 1:1 Mapping
    # ---------------------------------------------------------------------

    def generate_until(self, instances: List[Any]) -> List[str]:
        logger.info("=" * 80)
        logger.info("GENERATE_UNTIL called with %d instances", len(instances))
        logger.info("=" * 80)

        results = []

        for idx, instance in enumerate(instances):
            prompt, stop_seqs, audio_dicts = self._extract_instance_data(instance)

            if not prompt and not audio_dicts:
                logger.warning("Empty prompt at index %d", idx)
                results.append("")
                continue

            # Build contents: audio parts first, then the text prompt
            if audio_dicts:
                contents = self._audio_dicts_to_parts(audio_dicts) + [prompt]
            else:
                contents = prompt

            final_response = ""
            for attempt in range(self.max_retries):
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=contents,
                        config=self._gen_config(stop_seqs),
                    )
                    response_text = response.text or ""
                    if response_text.strip():
                        final_response = response_text
                        break
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_timeout * (attempt + 1))
                except Exception as e:
                    logger.error("Generation error idx=%d attempt=%d: %s", idx, attempt + 1, e)
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_timeout * (attempt + 1))

            results.append(final_response)

        assert len(results) == len(instances)
        return results
    # ---------------------------------------------------------------------
    # Loglikelihood (unsupported by Gemini)
    # ---------------------------------------------------------------------

    def loglikelihood(self, instances: List[Any]) -> List[Tuple[float, bool]]:
        """
        Gemini API does not support loglikelihood computation.
        Returns dummy values.
        """
        logger.info(
            f"LOGLIKELIHOOD called with {len(instances)} instances (returning dummy values)"
        )
        return [(0.0, True) for _ in instances]

    def loglikelihood_rolling(
        self, instances: List[Any]
    ) -> List[List[Tuple[float, bool]]]:
        """
        Gemini API does not support rolling loglikelihood computation.
        Returns dummy values.
        """
        logger.info(
            f"LOGLIKELIHOOD_ROLLING called with {len(instances)} instances (returning dummy values)"
        )
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
                        temperature=(
                            temperature if temperature is not None else self.temperature
                        ),
                        max_output_tokens=(
                            max_tokens if max_tokens is not None else self.max_tokens
                        ),
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
                logger.warning(
                    f"Completion error, attempt {attempt + 1}/{self.max_retries}: {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_timeout * (attempt + 1))
                else:
                    logger.error(
                        f"All attempts failed. Last error: {e}. Returning empty string."
                    )
                    final_response = ""
                    break

        return final_response
