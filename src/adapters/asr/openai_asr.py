"""OpenAI-compatible ASR adapter for LM Evaluation Harness.

Supports any OpenAI-compatible transcription endpoint:
- OpenAI Whisper API
- Groq Whisper (whisper-large-v3-turbo, distil-whisper-large-v3-en, etc.)
- Local Whisper servers (faster-whisper-server, whisper.cpp, etc.)

Sends audio to /v1/audio/transcriptions and returns the transcription text.
For text-only requests (no audio), returns an empty string.

Dependencies: openai, numpy, soundfile
"""

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


@register_model("openai-asr")
class OpenAIWhisperLM(LM):
    """OpenAI-compatible ASR adapter using the /v1/audio/transcriptions endpoint.

    Works with OpenAI, Groq, and any local server exposing the same API.
    """

    MULTIMODAL = True

    def __init__(
        self,
        model: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        language: Optional[str] = None,
        temperature: float = 0.0,
        retry_timeout: float = 30.0,
        max_retries: int = 3,
        **_kwargs,
    ):
        super().__init__()

        self.model_name = (
            model or model_name
            or os.environ.get("MODEL", "whisper-1")
        )
        self.language = language or os.environ.get("ASR_LANGUAGE")
        self.temperature = temperature
        self.retry_timeout = retry_timeout
        self.max_retries = max_retries
        self._tokenizer_name = self.model_name

        api_key = (
            api_key
            or os.environ.get("API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "No API key provided. Set OPENAI_API_KEY or API_KEY environment "
                "variable or pass api_key parameter."
            )

        base_url = base_url or os.environ.get("BASE_URL")
        if base_url:
            base_url = self._normalize_base_url(base_url)

        self.client = OpenAI(api_key=api_key, base_url=base_url)

        logger.info(
            "Initialized OpenAIWhisperLM with model '%s' at %s (language=%s)",
            self.model_name,
            base_url or "https://api.openai.com/v1",
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
        """ASR models do not have a sequence length limit."""
        return 0

    @property
    def batch_size(self) -> int:
        """ASR requests are processed one at a time."""
        return 1

    # --------------------------------------------------------------------- #
    # URL helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _normalize_base_url(url: str) -> str:
        """Ensure base_url ends with /v1 for the OpenAI SDK."""
        url = url.rstrip("/")
        endpoint_suffixes = [
            "/audio/transcriptions",
            "/chat/completions",
        ]
        for suffix in endpoint_suffixes:
            if url.endswith(suffix):
                url = url[: -len(suffix)]
                break
        if not url.endswith("/v1"):
            if "/v1" not in url:
                url = f"{url}/v1"
        return url

    # --------------------------------------------------------------------- #
    # Audio helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _audio_dict_to_wav_bytes(audio_dict: dict) -> bytes:
        """Convert a single lm-eval audio dict to WAV bytes."""
        array = np.array(audio_dict["array"])
        if array.dtype != np.float32:
            array = array.astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, array, audio_dict["sampling_rate"], format="WAV", subtype="PCM_16")
        return buf.getvalue()

    @staticmethod
    def _extract_audio(instance: Any) -> Optional[List[dict]]:
        """Extract audio dicts from a request instance."""
        if hasattr(instance, "args") and len(instance.args) >= 3:
            aux = instance.args[2]
            if isinstance(aux, dict) and "audio" in aux:
                audio: List[dict] = aux["audio"]
                return audio
        return None

    # --------------------------------------------------------------------- #
    # Transcription
    # --------------------------------------------------------------------- #

    def _transcribe_audio(self, wav_bytes: bytes) -> str:
        """Send audio to the transcription endpoint with retry."""
        for attempt in range(self.max_retries):
            try:
                buf = io.BytesIO(wav_bytes)
                buf.name = "audio.wav"

                kwargs: dict[str, Any] = {
                    "model": self.model_name,
                    "file": buf,
                    "response_format": "text",
                    "temperature": self.temperature,
                }
                if self.language:
                    kwargs["language"] = self.language

                transcription = self.client.audio.transcriptions.create(**kwargs)

                text = (
                    transcription
                    if isinstance(transcription, str)
                    else getattr(transcription, "text", str(transcription))
                )

                if text.strip():
                    return text.strip()

                if attempt < self.max_retries - 1:
                    logger.warning("Empty transcription, retrying...")
                    time.sleep(self.retry_timeout * (attempt + 1))
            except Exception as e:  # noqa: BLE001  # pylint: disable=broad-exception-caught
                logger.error(
                    "Transcription error (attempt %d/%d): %s: %s",
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
        logger.info("GENERATE_UNTIL (ASR) called with %d requests", len(requests))
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
                wav_bytes = self._audio_dict_to_wav_bytes(audio_dict)
                text = self._transcribe_audio(wav_bytes)
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
