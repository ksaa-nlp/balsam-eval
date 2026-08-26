"""AssemblyAI asynchronous transcription adapter."""

import os
from typing import Optional

from lm_eval.api.registry import register_model  # type: ignore[import-untyped]

from src.adapters.asr._http import HTTPASRLM


@register_model("assemblyai-stt")
class AssemblyAISTTLM(HTTPASRLM):
    """Upload, submit, and poll AssemblyAI transcription jobs."""

    def __init__(self, model: Optional[str] = None, model_name: Optional[str] = None,
                 api_key: Optional[str] = None, base_url: Optional[str] = None,
                 language: Optional[str] = None, **kwargs):
        key = api_key or os.environ.get("ASSEMBLYAI_API_KEY") or os.environ.get("API_KEY")
        if not key:
            raise ValueError("No API key provided. Set ASSEMBLYAI_API_KEY or API_KEY.")
        super().__init__(
            model_name=model or model_name or os.environ.get("MODEL", "universal-3-5-pro"),
            api_key=key, language=language or os.environ.get("ASR_LANGUAGE"),
            base_url=base_url or os.environ.get(
                "ASSEMBLYAI_STT_URL", "https://api.assemblyai.com/v2"
            ),
            **kwargs,
        )

    def _transcribe_audio(self, wav_bytes: bytes) -> str:
        headers = {"authorization": self.api_key}
        upload = self._request(
            "POST", f"{self.base_url}/upload", headers=headers, data=wav_bytes
        ).json()
        payload = {"audio_url": upload["upload_url"], "speech_models": [self.model_name]}
        if self.language:
            payload["language_code"] = self.language
        else:
            payload["language_detection"] = True
        submitted = self._request(
            "POST", f"{self.base_url}/transcript", headers=headers, json=payload
        ).json()
        job_id = submitted["id"]
        result = self._poll(
            lambda: self._request(
                "GET", f"{self.base_url}/transcript/{job_id}", headers=headers
            ).json(),
            lambda body: str(body.get("status") or ""), pending={"queued", "processing"},
            succeeded={"completed"}, failed={"error"}, provider="AssemblyAI",
        )
        return str(result.get("text") or "").strip()
