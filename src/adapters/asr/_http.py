"""Shared REST utilities for speech-to-text adapters."""

import io
import logging
import time
from typing import Any, Callable, List, Optional, Tuple, cast

import numpy as np
import requests as http_requests
import soundfile as sf  # type: ignore[import-untyped]
from tqdm import tqdm

from lm_eval.api.model import LM  # type: ignore[import-untyped]


logger = logging.getLogger(__name__)


class HTTPASRLM(LM):
    """Common LM contract, audio conversion, HTTP retry, and polling behavior."""

    MULTIMODAL = True

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str,
        language: Optional[str],
        base_url: str,
        max_retries: int = 3,
        retry_timeout: float = 2.0,
        request_timeout: float = 120.0,
        poll_interval: float = 2.0,
        poll_timeout: float = 600.0,
    ) -> None:
        super().__init__()
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 1:
            raise ValueError("max_retries must be a positive integer")
        for name, value, allow_zero in (
            ("retry_timeout", retry_timeout, True),
            ("request_timeout", request_timeout, False),
            ("poll_interval", poll_interval, True),
            ("poll_timeout", poll_timeout, False),
        ):
            if not np.isfinite(value) or value < 0 or (not allow_zero and value == 0):
                qualifier = "positive" if not allow_zero else "non-negative"
                raise ValueError(f"{name} must be finite and {qualifier}")

        self.model_name = model_name
        self.api_key = api_key
        self.language = language
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.retry_timeout = retry_timeout
        self.request_timeout = request_timeout
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self._tokenizer_name = model_name
        self.session = http_requests.Session()

    @property
    def tokenizer_name(self) -> str:
        """Return model name as tokenizer identifier."""
        return self._tokenizer_name

    @property
    def max_sequence_length(self) -> int:
        """ASR models have no token sequence limit."""
        return 0

    @property
    def batch_size(self) -> int:
        """HTTP adapters process audio sequentially."""
        return 1

    @staticmethod
    def _audio_dict_to_wav_bytes(audio_dict: dict) -> bytes:
        array = np.asarray(audio_dict["array"], dtype=np.float32)
        buffer = io.BytesIO()
        sf.write(buffer, array, audio_dict["sampling_rate"], format="WAV", subtype="PCM_16")
        return buffer.getvalue()

    @staticmethod
    def _extract_audio(instance: Any) -> Optional[List[dict]]:
        if hasattr(instance, "args") and len(instance.args) >= 3:
            auxiliary = instance.args[2]
            if isinstance(auxiliary, dict) and "audio" in auxiliary:
                return cast(List[dict], auxiliary["audio"])
        return None

    def _request(self, method: str, url: str, **kwargs: Any) -> http_requests.Response:
        """Issue request, retrying transport errors, throttling, and server errors."""
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.request(
                    method, url, timeout=self.request_timeout, **kwargs
                )
                if response.status_code not in {408, 429} and response.status_code < 500:
                    response.raise_for_status()
                    return response
                response.raise_for_status()
            except http_requests.RequestException as error:
                last_error = error
                status = getattr(error.response, "status_code", None)
                if status is not None and status not in {408, 429} and status < 500:
                    raise RuntimeError(f"HTTP request failed: {method} {url}") from error
                if attempt + 1 < self.max_retries:
                    delay = self.retry_timeout * (attempt + 1)
                    retry_after = getattr(error.response, "headers", {}).get("Retry-After")
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            pass
                    time.sleep(delay)
        raise RuntimeError(f"HTTP request failed after retries: {method} {url}") from last_error

    def _poll(
        self,
        fetch: Callable[[], dict],
        status_of: Callable[[dict], str],
        *,
        pending: set[str],
        succeeded: set[str],
        failed: set[str],
        provider: str,
    ) -> dict:
        deadline = time.monotonic() + self.poll_timeout
        while True:
            payload = fetch()
            status = status_of(payload).lower()
            if status in succeeded:
                return payload
            if status in failed:
                detail = payload.get("error") or payload.get("failure_detail") or status
                raise RuntimeError(f"{provider} transcription failed: {detail}")
            if status not in pending:
                raise RuntimeError(f"{provider} returned unknown job status: {status or '<empty>'}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{provider} transcription timed out after {self.poll_timeout}s")
            time.sleep(min(self.poll_interval, remaining))

    def _retry_empty(self, transcribe: Callable[[], str], provider: str) -> str:
        for attempt in range(self.max_retries):
            text = transcribe().strip()
            if text:
                return text
            if attempt + 1 < self.max_retries:
                logger.warning("%s returned empty transcription; retrying", provider)
                time.sleep(self.retry_timeout * (attempt + 1))
        return ""

    def generate_until(self, requests: List[Any]) -> List[str]:
        results: List[str] = []
        for instance in tqdm(requests, desc=f"Transcribing {self.model_name}", unit="req"):
            audio_dicts = self._extract_audio(instance)
            if not audio_dicts:
                results.append("")
                continue
            texts = [
                self._transcribe_audio(self._audio_dict_to_wav_bytes(item))
                for item in audio_dicts
            ]
            results.append(" ".join(text for text in texts if text))
        return results

    def _transcribe_audio(self, wav_bytes: bytes) -> str:
        raise NotImplementedError

    def loglikelihood(self, requests: List[Any]) -> List[Tuple[float, bool]]:
        return [(0.0, True) for _ in requests]

    def loglikelihood_rolling(self, requests: List[Any]) -> List[float]:
        return [0.0 for _ in requests]
