"""HuggingFace Inference API ASR adapter for LM Evaluation Harness.

Supports any ASR model hosted on the HuggingFace Inference API:
- Whisper variants (openai/whisper-large-v3, etc.)
- Wav2Vec2 / HuBERT (facebook/wav2vec2-large-960h, etc.)
- SenseVoice (FunAudioLLM/SenseVoiceSmall, etc.)
- Any model available via HF Inference API or Inference Endpoints

Uses huggingface_hub InferenceClient — no local model loading.

Dependencies: huggingface-hub, numpy, soundfile
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
    from huggingface_hub import InferenceClient  # type: ignore[import-untyped]
except ImportError:
    InferenceClient = None  # type: ignore[assignment,misc]


@register_model("hf-asr")
class HuggingFaceASRLM(LM):
    """HuggingFace Inference API ASR adapter.

    Calls the HF Inference API (serverless or dedicated Inference Endpoints)
    for automatic speech recognition.
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

        if InferenceClient is None:
            raise ImportError(
                "huggingface-hub is required for the hf-asr adapter. "
                "Install it with: pip install huggingface-hub"
            )

        self.model_name = (
            model or model_name
            or os.environ.get("MODEL", "openai/whisper-large-v3")
        )
        self.language = language or os.environ.get("ASR_LANGUAGE")
        self.retry_timeout = retry_timeout
        self.max_retries = max_retries
        self._tokenizer_name = self.model_name

        token = (
            api_key
            or os.environ.get("API_KEY")
            or os.environ.get("HF_TOKEN")
        )
        if not token:
            raise ValueError(
                "No API token provided. Set HF_TOKEN or API_KEY "
                "environment variable or pass api_key parameter."
            )

        client_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "token": token,
        }
        if base_url or os.environ.get("BASE_URL"):
            client_kwargs["api_url"] = (
                base_url or os.environ.get("BASE_URL", "")
            ).rstrip("/")

        self.client = InferenceClient(**client_kwargs)

        logger.info(
            "Initialized HuggingFaceASRLM (API) with model '%s' (language=%s)",
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
        """Return max sequence length (unused for ASR)."""
        return 0

    @property
    def batch_size(self) -> int:
        """Return batch size."""
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
        """Transcribe audio via the HuggingFace Inference API with retry."""
        for attempt in range(self.max_retries):
            try:
                result = self.client.automatic_speech_recognition(wav_bytes)

                text = ""
                if isinstance(result, str):
                    text = result
                elif isinstance(result, dict):
                    text = str(result.get("text", ""))
                elif hasattr(result, "text"):
                    text = str(result.text)

                if text.strip():
                    return text.strip()

                if attempt < self.max_retries - 1:
                    logger.warning("Empty HF transcription, retrying...")
                    time.sleep(self.retry_timeout * (attempt + 1))
            except (OSError, ValueError, RuntimeError) as e:
                logger.error(
                    "HF ASR API error (attempt %d/%d): %s: %s",
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
        logger.info("GENERATE_UNTIL (HF ASR API) called with %d requests", len(requests))
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
