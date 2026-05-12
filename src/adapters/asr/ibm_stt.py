"""IBM Watson Speech-to-Text adapter for LM Evaluation Harness.

Uses the IBM Watson Speech-to-Text service REST API for ASR evaluation.
Requires an IBM Cloud API key and service URL.

Supported models include:
- ar-MS_Telephony (Arabic)
- en-US_Multimedia, en-US_Telephony (English)
- And many other language-specific models

Dependencies: ibm-watson, ibm-cloud-sdk-core, numpy, soundfile
"""

import io
import logging
import os
import time
from typing import Any, List, Optional, Tuple

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
from tqdm import tqdm

from lm_eval.api.model import LM  # type: ignore[import-untyped]
from lm_eval.api.registry import register_model  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

try:
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator  # type: ignore[import-untyped]
    from ibm_watson import SpeechToTextV1  # type: ignore[import-untyped]
except ImportError:
    IAMAuthenticator = None  # type: ignore[assignment,misc]
    SpeechToTextV1 = None  # type: ignore[assignment,misc]


@register_model("ibm-stt")
class IBMSTTLM(LM):
    """IBM Watson Speech-to-Text adapter.

    Authenticates via IAM API key and connects to the Watson STT service.
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

        if SpeechToTextV1 is None or IAMAuthenticator is None:
            raise ImportError(
                "ibm-watson and ibm-cloud-sdk-core are required for the ibm-stt adapter. "
                "Install them with: pip install ibm-watson ibm-cloud-sdk-core"
            )

        self.model_name = (
            model or model_name
            or os.environ.get("MODEL", "ar-MS_Telephony")
        )
        self.language = language or os.environ.get("ASR_LANGUAGE")
        self.retry_timeout = retry_timeout
        self.max_retries = max_retries
        self._tokenizer_name = self.model_name

        resolved_api_key = (
            api_key
            or os.environ.get("API_KEY")
            or os.environ.get("IBM_API_KEY")
        )
        if not resolved_api_key:
            raise ValueError(
                "No API key provided. Set IBM_API_KEY or API_KEY "
                "environment variable or pass api_key parameter."
            )

        service_url = (
            base_url
            or os.environ.get("BASE_URL")
            or os.environ.get("IBM_STT_URL")
        )
        if not service_url:
            raise ValueError(
                "No service URL provided. Set IBM_STT_URL or BASE_URL "
                "environment variable or pass base_url parameter."
            )

        authenticator = IAMAuthenticator(resolved_api_key)
        self.client = SpeechToTextV1(authenticator=authenticator)
        self.client.set_service_url(service_url.rstrip("/"))

        logger.info(
            "Initialized IBMSTTLM with model '%s' at %s",
            self.model_name,
            service_url,
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
        """Transcribe audio via IBM Watson STT with retry."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.recognize(
                    audio=io.BytesIO(wav_bytes),
                    content_type="audio/wav",
                    model=self.model_name,
                ).get_result()

                transcripts = []
                for result in response.get("results", []):
                    alternatives = result.get("alternatives", [])
                    if alternatives:
                        transcripts.append(alternatives[0].get("transcript", ""))

                text = " ".join(transcripts).strip()
                if text:
                    return text

                if attempt < self.max_retries - 1:
                    logger.warning("Empty transcription from IBM STT, retrying...")
                    time.sleep(self.retry_timeout * (attempt + 1))
            except Exception as e:  # noqa: BLE001
                logger.error(
                    "IBM STT error (attempt %d/%d): %s: %s",
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
        logger.info("GENERATE_UNTIL (IBM STT) called with %d requests", len(requests))
        logger.info("=" * 80)

        results: List[str] = []

        for instance in tqdm(requests, desc="Transcribing (IBM STT)", unit="req"):
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
