"""Cohere Transcribe REST adapter."""

import os
from typing import Optional

from lm_eval.api.registry import register_model  # type: ignore[import-untyped]

from src.adapters.asr._http import HTTPASRLM


@register_model("cohere-asr")
class CohereASRLM(HTTPASRLM):
    """Transcribe audio with Cohere Transcribe."""

    def __init__(self, model: Optional[str] = None, model_name: Optional[str] = None,
                 api_key: Optional[str] = None, base_url: Optional[str] = None,
                 language: Optional[str] = None, **kwargs):
        key = api_key or os.environ.get("COHERE_API_KEY") or os.environ.get("API_KEY")
        if not key:
            raise ValueError("No API key provided. Set COHERE_API_KEY or API_KEY.")
        super().__init__(
            model_name=model or model_name or os.environ.get("MODEL", "cohere-transcribe-03-2026"),
            api_key=key,
            language=language or os.environ.get("ASR_LANGUAGE", "en"),
            base_url=base_url or os.environ.get(
                "COHERE_ASR_URL", "https://api.cohere.com/v2/audio/transcriptions"
            ),
            **kwargs,
        )

    def _transcribe_audio(self, wav_bytes: bytes) -> str:
        def transcribe() -> str:
            response = self._request(
                "POST", self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                data={"model": self.model_name, "language": self.language},
            )
            return str(response.json().get("text") or "")
        return self._retry_empty(transcribe, "Cohere")
