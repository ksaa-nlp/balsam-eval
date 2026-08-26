"""Deepgram pre-recorded speech-to-text REST adapter."""

import os
from typing import Optional

from lm_eval.api.registry import register_model  # type: ignore[import-untyped]

from src.adapters.asr._http import HTTPASRLM


@register_model("deepgram-stt")
class DeepgramSTTLM(HTTPASRLM):
    """Transcribe pre-recorded audio with Deepgram."""

    def __init__(self, model: Optional[str] = None, model_name: Optional[str] = None,
                 api_key: Optional[str] = None, base_url: Optional[str] = None,
                 language: Optional[str] = None, **kwargs):
        key = api_key or os.environ.get("DEEPGRAM_API_KEY") or os.environ.get("API_KEY")
        if not key:
            raise ValueError("No API key provided. Set DEEPGRAM_API_KEY or API_KEY.")
        super().__init__(
            model_name=model or model_name or os.environ.get("MODEL", "nova-3"),
            api_key=key,
            language=language or os.environ.get("ASR_LANGUAGE", "en"),
            base_url=base_url or os.environ.get(
                "DEEPGRAM_STT_URL", "https://api.deepgram.com/v1/listen"
            ),
            **kwargs,
        )

    def _transcribe_audio(self, wav_bytes: bytes) -> str:
        def transcribe() -> str:
            response = self._request(
                "POST", self.base_url,
                headers={"Authorization": f"Token {self.api_key}", "Content-Type": "audio/wav"},
                params={
                    "model": self.model_name,
                    "language": self.language,
                    "smart_format": "true",
                },
                data=wav_bytes,
            )
            channels = response.json().get("results", {}).get("channels", [])
            return " ".join(
                str(channel.get("alternatives", [{}])[0].get("transcript") or "").strip()
                for channel in channels if channel.get("alternatives")
            )
        return self._retry_empty(transcribe, "Deepgram")
