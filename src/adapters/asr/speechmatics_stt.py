"""Speechmatics batch transcription adapter."""

import json
import os
from typing import Optional

from lm_eval.api.registry import register_model  # type: ignore[import-untyped]

from src.adapters.asr._http import HTTPASRLM


@register_model("speechmatics-stt")
class SpeechmaticsSTTLM(HTTPASRLM):
    """Upload, submit, and poll Speechmatics batch jobs."""

    def __init__(self, model: Optional[str] = None, model_name: Optional[str] = None,
                 api_key: Optional[str] = None, base_url: Optional[str] = None,
                 language: Optional[str] = None, **kwargs):
        key = api_key or os.environ.get("SPEECHMATICS_API_KEY") or os.environ.get("API_KEY")
        if not key:
            raise ValueError("No API key provided. Set SPEECHMATICS_API_KEY or API_KEY.")
        super().__init__(
            model_name=model or model_name or os.environ.get("MODEL", "standard"),
            api_key=key, language=language or os.environ.get("ASR_LANGUAGE", "ar"),
            base_url=base_url or os.environ.get(
                "SPEECHMATICS_STT_URL", "https://asr.api.speechmatics.com/v2"
            ),
            **kwargs,
        )

    def _transcribe_audio(self, wav_bytes: bytes) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        config = {
            "type": "transcription",
            "transcription_config": {
                "language": self.language,
                "operating_point": self.model_name,
            },
        }
        submitted = self._request(
            "POST", f"{self.base_url}/jobs", headers=headers,
            files={"data_file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"config": json.dumps(config)},
        ).json()
        job_id = submitted["id"]
        self._poll(
            lambda: self._request("GET", f"{self.base_url}/jobs/{job_id}", headers=headers).json(),
            lambda body: str(body.get("job", {}).get("status") or ""),
            pending={"running", "queued"}, succeeded={"done"},
            failed={"rejected", "deleted"}, provider="Speechmatics",
        )
        transcript = self._request(
            "GET", f"{self.base_url}/jobs/{job_id}/transcript", headers=headers,
        ).json()
        parts = []
        for item in transcript.get("results", []):
            alternatives = item.get("alternatives") or []
            if alternatives:
                parts.append(str(alternatives[0].get("content") or ""))
        return "".join(parts).strip()
