"""Rev AI asynchronous speech-to-text adapter."""

import json
import os
from typing import Optional

from lm_eval.api.registry import register_model  # type: ignore[import-untyped]

from src.adapters.asr._http import HTTPASRLM


@register_model("revai-stt")
class RevAISTTLM(HTTPASRLM):
    """Upload, submit, and poll Rev AI transcription jobs."""

    def __init__(self, model: Optional[str] = None, model_name: Optional[str] = None,
                 api_key: Optional[str] = None, base_url: Optional[str] = None,
                 language: Optional[str] = None, **kwargs):
        key = api_key or os.environ.get("REVAI_API_KEY") or os.environ.get("API_KEY")
        if not key:
            raise ValueError("No API key provided. Set REVAI_API_KEY or API_KEY.")
        super().__init__(
            model_name=model or model_name or os.environ.get("MODEL", "machine"),
            api_key=key, language=language or os.environ.get("ASR_LANGUAGE", "en"),
            base_url=base_url or os.environ.get(
                "REVAI_STT_URL", "https://api.rev.ai/speechtotext/v1"
            ),
            **kwargs,
        )

    def _transcribe_audio(self, wav_bytes: bytes) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        options = {"transcriber": self.model_name, "language": self.language}
        submitted = self._request(
            "POST", f"{self.base_url}/jobs", headers=headers,
            files={"media": ("audio.wav", wav_bytes, "audio/wav")},
            data={"options": json.dumps(options)},
        ).json()
        job_id = submitted["id"]
        self._poll(
            lambda: self._request("GET", f"{self.base_url}/jobs/{job_id}", headers=headers).json(),
            lambda body: str(body.get("status") or ""), pending={"in_progress"},
            succeeded={"transcribed"}, failed={"failed"}, provider="Rev AI",
        )
        response = self._request(
            "GET", f"{self.base_url}/jobs/{job_id}/transcript",
            headers={**headers, "Accept": "text/plain"},
        )
        return response.text.strip()
