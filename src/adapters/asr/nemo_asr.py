"""NVIDIA NIM ASR adapter for LM Evaluation Harness.

Supports NVIDIA ASR models via the NVIDIA NIM API (OpenAI-compatible):
- Parakeet CTC/TDT/RNNT (nvidia/parakeet-ctc-1.1b, nvidia/parakeet-tdt-1.1b, etc.)
- Canary (nvidia/canary-1b, nvidia/canary-1b-flash, etc.)
- Any ASR model served via NVIDIA NIM

Uses the OpenAI SDK to call the /v1/audio/transcriptions endpoint on
NVIDIA's API catalog (https://integrate.api.nvidia.com/v1) or a
self-hosted NIM instance.

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

NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"


@register_model("nemo-asr")
class NeMoASRLM(LM):
    """NVIDIA NIM ASR adapter.

    Sends audio to a NVIDIA NIM /v1/audio/transcriptions endpoint.
    Works with the NVIDIA API catalog and self-hosted NIM instances.
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
            or os.environ.get("MODEL", "nvidia/parakeet-ctc-1.1b")
        )
        self.language = language or os.environ.get("ASR_LANGUAGE")
        self.temperature = temperature
        self.retry_timeout = retry_timeout
        self.max_retries = max_retries
        self._tokenizer_name = self.model_name

        resolved_key = (
            api_key
            or os.environ.get("API_KEY")
            or os.environ.get("NVIDIA_API_KEY")
        )
        if not resolved_key:
            raise ValueError(
                "No API key provided. Set NVIDIA_API_KEY or API_KEY "
                "environment variable or pass api_key parameter."
            )

        resolved_base = base_url or os.environ.get("BASE_URL", NVIDIA_NIM_BASE_URL)
        resolved_base = resolved_base.rstrip("/")
        if not resolved_base.endswith("/v1"):
            if "/v1" not in resolved_base:
                resolved_base = f"{resolved_base}/v1"

        self.client = OpenAI(api_key=resolved_key, base_url=resolved_base)

        logger.info(
            "Initialized NeMoASRLM (API) with model '%s' at %s (language=%s)",
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
    def _audio_dict_to_wav_bytes(audio_dict: dict) -> bytes:
        """Convert lm-eval audio dict to WAV bytes."""
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
        """Send audio to the NVIDIA NIM transcription endpoint with retry."""
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
                    logger.warning("Empty NIM transcription, retrying...")
                    time.sleep(self.retry_timeout * (attempt + 1))
            except Exception as e:  # noqa: BLE001
                logger.error(
                    "NVIDIA NIM ASR error (attempt %d/%d): %s: %s",
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
        logger.info("GENERATE_UNTIL (NeMo NIM API) called with %d requests", len(requests))
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
