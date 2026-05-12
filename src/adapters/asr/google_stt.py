"""Google Cloud Speech-to-Text adapter for LM Evaluation Harness.

Uses the Google Cloud Speech-to-Text API (v1) for ASR evaluation.
Authenticates via GOOGLE_APPLICATION_CREDENTIALS (service account key)
or Application Default Credentials (gcloud auth).

Dependencies: google-cloud-speech, numpy, soundfile
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
    from google.cloud import speech  # type: ignore[import-untyped]
except ImportError:
    speech = None  # type: ignore[assignment]


@register_model("google-stt")
class GoogleSTTLM(LM):
    """Google Cloud Speech-to-Text adapter.

    Authenticates via:
    - GOOGLE_APPLICATION_CREDENTIALS env var (service account JSON path)
    - Application Default Credentials (gcloud auth application-default login)
    """

    MULTIMODAL = True

    SAMPLE_RATE = 16000

    def __init__(
        self,
        model: Optional[str] = None,
        model_name: Optional[str] = None,
        language: Optional[str] = None,
        api_key: Optional[str] = None,  # pylint: disable=unused-argument
        base_url: Optional[str] = None,  # pylint: disable=unused-argument
        retry_timeout: float = 30.0,
        max_retries: int = 3,
        **_kwargs,
    ):
        super().__init__()

        if speech is None:
            raise ImportError(
                "google-cloud-speech is required for the google-stt adapter. "
                "Install it with: pip install google-cloud-speech"
            )

        self.model_name = (
            model or model_name
            or os.environ.get("MODEL", "default")
        )
        self.language = (
            language
            or os.environ.get("ASR_LANGUAGE", "ar-SA")
        )
        self.retry_timeout = retry_timeout
        self.max_retries = max_retries
        self._tokenizer_name = self.model_name

        self.client = speech.SpeechClient()

        logger.info(
            "Initialized GoogleSTTLM with model '%s' (language=%s)",
            self.model_name,
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
        """Transcribe audio using Google Cloud Speech-to-Text with retry."""
        audio = speech.RecognitionAudio(content=wav_bytes)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code=self.language,
            enable_automatic_punctuation=True,
        )

        for attempt in range(self.max_retries):
            try:
                response = self.client.recognize(config=config, audio=audio)

                transcripts = []
                for result in response.results:
                    if result.alternatives:
                        transcripts.append(result.alternatives[0].transcript)

                text = " ".join(transcripts).strip()
                if text:
                    return text

                if attempt < self.max_retries - 1:
                    logger.warning("Empty transcription, retrying...")
                    time.sleep(self.retry_timeout * (attempt + 1))
            except Exception as e:  # noqa: BLE001  # pylint: disable=broad-exception-caught
                logger.error(
                    "Google STT error (attempt %d/%d): %s: %s",
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
        logger.info("GENERATE_UNTIL (Google STT) called with %d requests", len(requests))
        logger.info("=" * 80)

        results: List[str] = []

        for instance in tqdm(requests, desc="Transcribing (Google STT)", unit="req"):
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
