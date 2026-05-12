"""Qwen3-ASR API adapter for LM Evaluation Harness.

Supports Qwen ASR models via an OpenAI-compatible API endpoint:
- Qwen3-ASR-Flash (Qwen's proprietary API via DashScope)
- Self-hosted qwen-asr-serve instances (vLLM backend)
- Any Qwen ASR model served with an OpenAI-compatible chat completions API

Sends audio as base64-encoded WAV in chat completions messages with
input_audio content blocks.

Dependencies: openai, numpy, soundfile
"""

import base64
import io
import logging
import os
import time
from typing import Any, List, Optional, Tuple

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
from openai import OpenAI
from tqdm import tqdm

from lm_eval.api.model import LM  # type: ignore[import-untyped]
from lm_eval.api.registry import register_model  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


@register_model("qwen-asr")
class QwenASRLM(LM):
    """Qwen3-ASR API adapter.

    Sends audio to a Qwen ASR API endpoint (qwen-asr-serve, DashScope, etc.)
    via OpenAI-compatible chat completions with audio content blocks.
    """

    MULTIMODAL = True

    def __init__(
        self,
        model: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        language: Optional[str] = None,
        retry_timeout: float = 30.0,
        max_retries: int = 3,
        **_kwargs,
    ):
        super().__init__()

        self.model_name = (
            model or model_name
            or os.environ.get("MODEL", "qwen3-asr-flash")
        )
        self.language = language or os.environ.get("ASR_LANGUAGE")
        self.retry_timeout = retry_timeout
        self.max_retries = max_retries
        self._tokenizer_name = self.model_name

        resolved_key = (
            api_key
            or os.environ.get("API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not resolved_key:
            raise ValueError(
                "No API key provided. Set API_KEY or OPENAI_API_KEY "
                "environment variable or pass api_key parameter."
            )

        resolved_base = (
            base_url
            or os.environ.get("BASE_URL")
        )
        if not resolved_base:
            raise ValueError(
                "No base URL provided. Set BASE_URL environment variable "
                "or pass base_url parameter. Examples:\n"
                "  - DashScope: https://dashscope.aliyuncs.com/compatible-mode/v1\n"
                "  - Self-hosted: http://localhost:8000/v1"
            )
        resolved_base = resolved_base.rstrip("/")
        if not resolved_base.endswith("/v1"):
            if "/v1" not in resolved_base:
                resolved_base = f"{resolved_base}/v1"

        self.client = OpenAI(api_key=resolved_key, base_url=resolved_base)

        logger.info(
            "Initialized QwenASRLM (API) with model '%s' at %s (language=%s)",
            self.model_name,
            resolved_base,
            self.language,
        )

    # --------------------------------------------------------------------- #
    # Required LM Eval properties
    # --------------------------------------------------------------------- #

    @property
    def tokenizer_name(self) -> str:
        return self._tokenizer_name

    @property
    def max_sequence_length(self) -> int:
        return 0

    @property
    def batch_size(self) -> int:
        return 1

    # --------------------------------------------------------------------- #
    # Audio helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _extract_audio(instance: Any) -> Optional[List[dict]]:
        """Extract audio dicts from a request instance."""
        if hasattr(instance, "args") and len(instance.args) >= 3:
            aux = instance.args[2]
            if isinstance(aux, dict) and "audio" in aux:
                audio: List[dict] = aux["audio"]
                return audio
        return None

    @staticmethod
    def _audio_dict_to_base64_wav(audio_dict: dict) -> str:
        """Convert an audio dict to a base64-encoded WAV string."""
        array = np.array(audio_dict["array"])
        if array.dtype != np.float32:
            array = array.astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, array, audio_dict["sampling_rate"], format="WAV", subtype="PCM_16")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # --------------------------------------------------------------------- #
    # Transcription
    # --------------------------------------------------------------------- #

    def _transcribe_audio(self, audio_dict: dict) -> str:
        """Transcribe by sending audio to the Qwen ASR API endpoint."""
        audio_b64 = self._audio_dict_to_base64_wav(audio_dict)
        data_uri = f"data:audio/wav;base64,{audio_b64}"

        prompt = "Transcribe this audio."
        if self.language:
            prompt = f"Transcribe this audio in {self.language}."

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": data_uri, "format": "wav"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                )
                text = response.choices[0].message.content or ""
                if text.strip():
                    return text.strip()

                if attempt < self.max_retries - 1:
                    logger.warning("Empty Qwen ASR API response, retrying...")
                    time.sleep(self.retry_timeout * (attempt + 1))
            except Exception as e:  # noqa: BLE001
                logger.error(
                    "Qwen ASR API error (attempt %d/%d): %s: %s",
                    attempt + 1, self.max_retries, type(e).__name__, e,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_timeout * (attempt + 1))

        return ""

    # --------------------------------------------------------------------- #
    # Generation (lm_eval interface)
    # --------------------------------------------------------------------- #

    def generate_until(self, requests: List[Any]) -> List[str]:
        logger.info("=" * 80)
        logger.info("GENERATE_UNTIL (Qwen ASR API) called with %d requests", len(requests))
        logger.info("=" * 80)

        results: List[str] = []

        for instance in tqdm(requests, desc=f"Transcribing {self.model_name}", unit="req"):
            audio_dicts = self._extract_audio(instance)

            if not audio_dicts:
                logger.warning("No audio found in request, returning empty string")
                results.append("")
                continue

            transcriptions = []
            for audio_dict in audio_dicts:
                text = self._transcribe_audio(audio_dict)
                transcriptions.append(text)

            results.append(" ".join(t for t in transcriptions if t))

        assert len(results) == len(requests), (
            f"Result count mismatch: {len(results)} results for {len(requests)} requests"
        )
        return results

    # --------------------------------------------------------------------- #
    # Loglikelihood (unsupported)
    # --------------------------------------------------------------------- #

    def loglikelihood(self, requests: List[Any]) -> List[Tuple[float, bool]]:
        logger.warning(
            "ASR models do not support loglikelihood. "
            "Returning dummy values for %d requests.",
            len(requests),
        )
        return [(0.0, True) for _ in requests]

    def loglikelihood_rolling(
        self, requests: List[Any]
    ) -> List[List[Tuple[float, bool]]]:
        logger.warning(
            "ASR models do not support loglikelihood_rolling. "
            "Returning dummy values for %d requests.",
            len(requests),
        )
        return [[(0.0, True)] for _ in requests]
