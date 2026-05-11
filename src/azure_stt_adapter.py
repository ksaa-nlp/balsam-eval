"""Azure Speech Services adapter for LM Evaluation Harness.

Uses the Azure Cognitive Services Speech-to-Text REST API for ASR evaluation.
Requires an Azure Speech resource key and region.

Dependencies: requests, numpy, soundfile
"""

import io
import logging
import os
import time
from typing import Any, List, Optional, Tuple

import numpy as np
import requests as http_requests
import soundfile as sf  # type: ignore[import-untyped]
from tqdm import tqdm

from lm_eval.api.model import LM  # type: ignore[import-untyped]
from lm_eval.api.registry import register_model  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


@register_model("azure-stt")
class AzureSTTLM(LM):
    """Azure Cognitive Services Speech-to-Text adapter.

    Uses the REST API (short audio, up to 60 seconds per request).
    For longer audio, consider using the Azure Speech SDK directly.
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
            or os.environ.get("MODEL", "azure-stt")
        )
        self.language = (
            language
            or os.environ.get("ASR_LANGUAGE", "ar-SA")
        )
        self.retry_timeout = retry_timeout
        self.max_retries = max_retries
        self._tokenizer_name = self.model_name

        self.api_key = (
            api_key
            or os.environ.get("API_KEY")
            or os.environ.get("AZURE_SPEECH_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "No API key provided. Set AZURE_SPEECH_KEY or API_KEY "
                "environment variable or pass api_key parameter."
            )

        self.region = os.environ.get("AZURE_SPEECH_REGION", "eastus")

        if base_url:
            self.endpoint_url = base_url.rstrip("/")
        else:
            self.endpoint_url = (
                f"https://{self.region}.stt.speech.microsoft.com"
                f"/speech/recognition/conversation/cognitiveservices/v1"
            )

        logger.info(
            "Initialized AzureSTTLM at %s (language=%s, region=%s)",
            self.endpoint_url,
            self.language,
            self.region,
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
    # Audio helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _audio_dict_to_wav_bytes(audio_dict: dict) -> Tuple[bytes, int]:
        """Convert lm-eval audio dict to WAV bytes and sample rate."""
        array = np.array(audio_dict["array"])
        if array.dtype != np.float32:
            array = array.astype(np.float32)
        sample_rate = audio_dict["sampling_rate"]
        buf = io.BytesIO()
        sf.write(buf, array, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue(), sample_rate

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

    def _transcribe_audio(self, wav_bytes: bytes, sample_rate: int) -> str:
        """Transcribe audio via Azure STT REST API with retry."""
        url = f"{self.endpoint_url}?language={self.language}&format=detailed"

        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key or "",
            "Content-Type": f"audio/wav; codecs=audio/pcm; samplerate={sample_rate}",
            "Accept": "application/json",
        }

        for attempt in range(self.max_retries):
            try:
                resp = http_requests.post(
                    url,
                    headers=headers,
                    data=wav_bytes,
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()

                status = data.get("RecognitionStatus", "")
                if status == "Success":
                    text = str(data.get("DisplayText", ""))
                    if text.strip():
                        return text.strip()

                if status == "NoMatch":
                    logger.warning("Azure STT: no speech detected in audio")
                    return ""

                if attempt < self.max_retries - 1:
                    logger.warning(
                        "Azure STT status '%s', retrying...", status
                    )
                    time.sleep(self.retry_timeout * (attempt + 1))
            except Exception as e:  # noqa: BLE001  # pylint: disable=broad-exception-caught
                logger.error(
                    "Azure STT error (attempt %d/%d): %s: %s",
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
        logger.info("GENERATE_UNTIL (Azure STT) called with %d requests", len(requests))
        logger.info("=" * 80)

        results: List[str] = []

        for instance in tqdm(requests, desc="Transcribing (Azure STT)", unit="req"):
            audio_dicts = self._extract_audio(instance)

            if not audio_dicts:
                logger.warning("No audio found in request, returning empty string")
                results.append("")
                continue

            transcriptions = []
            for audio_dict in audio_dicts:
                wav_bytes, sample_rate = self._audio_dict_to_wav_bytes(audio_dict)
                text = self._transcribe_audio(wav_bytes, sample_rate)
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
