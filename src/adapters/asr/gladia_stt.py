"""Gladia v2 asynchronous transcription adapter."""

import os
from typing import Optional

from lm_eval.api.registry import register_model  # type: ignore[import-untyped]

from src.adapters.asr._http import HTTPASRLM


@register_model("gladia-stt")
class GladiaSTTLM(HTTPASRLM):
    """Upload, submit, and poll Gladia v2 transcription jobs."""

    def __init__(self, model: Optional[str] = None, model_name: Optional[str] = None,
                 api_key: Optional[str] = None, base_url: Optional[str] = None,
                 language: Optional[str] = None, **kwargs):
        key = api_key or os.environ.get("GLADIA_API_KEY") or os.environ.get("API_KEY")
        if not key:
            raise ValueError("No API key provided. Set GLADIA_API_KEY or API_KEY.")
        super().__init__(
            model_name=model or model_name or os.environ.get("MODEL", "solaria-1"),
            api_key=key, language=language or os.environ.get("ASR_LANGUAGE"),
            base_url=base_url or os.environ.get("GLADIA_STT_URL", "https://api.gladia.io/v2"),
            **kwargs,
        )

    def _transcribe_audio(self, wav_bytes: bytes) -> str:
        headers = {"x-gladia-key": self.api_key}
        upload = self._request(
            "POST", f"{self.base_url}/upload", headers=headers,
            files={"audio": ("audio.wav", wav_bytes, "audio/wav")},
        ).json()
        payload = {"audio_url": upload["audio_url"], "model": self.model_name}
        if self.language:
            payload["language_config"] = {"languages": [self.language]}
        submitted = self._request(
            "POST", f"{self.base_url}/pre-recorded", headers=headers, json=payload
        ).json()
        job_id = submitted["id"]
        result = self._poll(
            lambda: self._request(
                "GET", f"{self.base_url}/pre-recorded/{job_id}", headers=headers
            ).json(),
            lambda body: str(body.get("status") or ""), pending={"queued", "processing"},
            succeeded={"done"}, failed={"error"}, provider="Gladia",
        )
        transcription = result.get("result", {}).get("transcription", {})
        return str(transcription.get("full_transcript") or "").strip()
