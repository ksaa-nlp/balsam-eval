"""ElevenLabs Scribe REST adapter."""

import os
from typing import Optional

from lm_eval.api.registry import register_model  # type: ignore[import-untyped]

from src.adapters.asr._http import HTTPASRLM


@register_model("elevenlabs-stt")
class ElevenLabsSTTLM(HTTPASRLM):
    """Transcribe audio with ElevenLabs Scribe."""

    def __init__(self, model: Optional[str] = None, model_name: Optional[str] = None,
                 api_key: Optional[str] = None, base_url: Optional[str] = None,
                 language: Optional[str] = None, **kwargs):
        key = api_key or os.environ.get("ELEVENLABS_API_KEY") or os.environ.get("API_KEY")
        if not key:
            raise ValueError("No API key provided. Set ELEVENLABS_API_KEY or API_KEY.")
        super().__init__(
            model_name=model or model_name or os.environ.get("MODEL", "scribe_v2"),
            api_key=key,
            language=language or os.environ.get("ASR_LANGUAGE", "ar"),
            base_url=base_url or os.environ.get(
                "ELEVENLABS_STT_URL",
                "https://api.elevenlabs.io/v1/speech-to-text",
            ),
            **kwargs,
        )

    def _transcribe_audio(self, wav_bytes: bytes) -> str:
        def transcribe() -> str:
            data = {"model_id": self.model_name}
            if self.language:
                data["language_code"] = self.language
            response = self._request(
                "POST", self.base_url,
                headers={"xi-api-key": self.api_key},
                files={"file": ("audio.wav", wav_bytes, "audio/wav")}, data=data,
            )
            payload = response.json()
            if "transcripts" in payload:
                return " ".join(str(item.get("text") or "") for item in payload["transcripts"])
            return str(payload.get("text") or "")
        return self._retry_empty(transcribe, "ElevenLabs")
