"""AWS Transcribe adapter for LM Evaluation Harness.

Uploads WAV audio to S3, starts a transcription job, polls it, reads the JSON
result from S3, and removes temporary AWS resources.

Optional dependency: boto3
"""

import io
import json
import logging
import os
import time
import uuid
from contextlib import suppress
from typing import Any, List, Optional, Tuple

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
from tqdm import tqdm

from lm_eval.api.model import LM  # type: ignore[import-untyped]
from lm_eval.api.registry import register_model  # type: ignore[import-untyped]

try:
    import boto3  # type: ignore[import-not-found]
except ImportError:
    boto3 = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@register_model("aws-transcribe")
class AWSTranscribeLM(LM):
    """Asynchronous AWS Transcribe ASR adapter backed by a caller-owned S3 bucket."""

    MULTIMODAL = True

    def __init__(
        self,
        model: Optional[str] = None,
        model_name: Optional[str] = None,
        language: Optional[str] = None,
        region: Optional[str] = None,
        bucket: Optional[str] = None,
        s3_prefix: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        s3_endpoint_url: Optional[str] = None,
        poll_interval: float = 2.0,
        job_timeout: float = 900.0,
        **_kwargs: Any,
    ):
        super().__init__()
        if boto3 is None:
            raise ImportError(
                "boto3 is required for the aws-transcribe adapter. "
                "Install it with: pip install boto3"
            )
        if poll_interval < 0 or not np.isfinite(poll_interval):
            raise ValueError("poll_interval must be finite and non-negative")
        if job_timeout <= 0 or not np.isfinite(job_timeout):
            raise ValueError("job_timeout must be finite and positive")

        self.model_name = model or model_name or os.environ.get("MODEL", "aws-transcribe")
        self.language = (
            language
            or os.environ.get("AWS_TRANSCRIBE_LANGUAGE_CODE")
            or os.environ.get("ASR_LANGUAGE", "ar-SA")
        )
        self.region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        self.bucket = bucket or os.environ.get("AWS_TRANSCRIBE_S3_BUCKET")
        if not self.bucket:
            raise ValueError(
                "No S3 bucket provided. Set AWS_TRANSCRIBE_S3_BUCKET or pass bucket."
            )
        self.s3_prefix = (
            s3_prefix or os.environ.get("AWS_TRANSCRIBE_S3_PREFIX", "balsam-eval")
        ).strip("/")
        self.poll_interval = poll_interval
        self.job_timeout = job_timeout
        self._tokenizer_name = self.model_name

        transcribe_url = endpoint_url or os.environ.get("AWS_TRANSCRIBE_ENDPOINT_URL")
        s3_url = s3_endpoint_url or os.environ.get("AWS_S3_ENDPOINT_URL")
        self.transcribe_client = boto3.client(
            "transcribe", region_name=self.region, endpoint_url=transcribe_url
        )
        self.s3_client = boto3.client("s3", region_name=self.region, endpoint_url=s3_url)

    @property
    def tokenizer_name(self) -> str:
        return self._tokenizer_name

    @property
    def max_sequence_length(self) -> int:
        """ASR models have no token sequence limit."""
        return 0

    @property
    def batch_size(self) -> int:
        """AWS Transcribe jobs are submitted sequentially."""
        return 1

    @staticmethod
    def _audio_dict_to_wav_bytes(audio_dict: dict) -> Tuple[bytes, int]:
        array = np.asarray(audio_dict["array"], dtype=np.float32)
        sample_rate = int(audio_dict["sampling_rate"])
        buffer = io.BytesIO()
        sf.write(buffer, array, sample_rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue(), sample_rate

    @staticmethod
    def _extract_audio(instance: Any) -> Optional[List[dict]]:
        if hasattr(instance, "args") and len(instance.args) >= 3:
            auxiliary = instance.args[2]
            if isinstance(auxiliary, dict) and "audio" in auxiliary:
                audio: List[dict] = auxiliary["audio"]
                return audio
        return None

    def _key(self, name: str) -> str:
        return f"{self.s3_prefix}/{name}" if self.s3_prefix else name

    def _transcribe_audio(self, wav_bytes: bytes, sample_rate: int) -> str:
        job_name = f"balsam-eval-{uuid.uuid4().hex}"
        input_key = self._key(f"{job_name}.wav")
        output_key = self._key(f"{job_name}.json")
        job_started = False

        try:
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=input_key,
                Body=wav_bytes,
                ContentType="audio/wav",
            )
            request: dict[str, Any] = {
                "TranscriptionJobName": job_name,
                "Media": {"MediaFileUri": f"s3://{self.bucket}/{input_key}"},
                "MediaFormat": "wav",
                "MediaSampleRateHertz": sample_rate,
                "LanguageCode": self.language,
                "OutputBucketName": self.bucket,
                "OutputKey": output_key,
            }
            self.transcribe_client.start_transcription_job(**request)
            job_started = True

            deadline = time.monotonic() + self.job_timeout
            while True:
                response = self.transcribe_client.get_transcription_job(
                    TranscriptionJobName=job_name
                )
                job = response["TranscriptionJob"]
                status = job["TranscriptionJobStatus"]
                if status == "COMPLETED":
                    break
                if status == "FAILED":
                    reason = job.get("FailureReason", "unknown reason")
                    raise RuntimeError(f"AWS Transcribe job failed: {reason}")
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"AWS Transcribe job did not finish within {self.job_timeout} seconds"
                    )
                time.sleep(self.poll_interval)

            body = self.s3_client.get_object(Bucket=self.bucket, Key=output_key)["Body"]
            payload = json.loads(body.read())
            transcript = payload["results"]["transcripts"][0]["transcript"]
            return str(transcript).strip()
        finally:
            if job_started:
                with suppress(Exception):
                    self.transcribe_client.delete_transcription_job(
                        TranscriptionJobName=job_name
                    )
            for key in (input_key, output_key):
                with suppress(Exception):
                    self.s3_client.delete_object(Bucket=self.bucket, Key=key)

    def generate_until(self, requests: List[Any]) -> List[str]:
        results: List[str] = []
        for instance in tqdm(requests, desc="Transcribing (AWS Transcribe)", unit="req"):
            audio_items = self._extract_audio(instance)
            if not audio_items:
                results.append("")
                continue
            transcripts = []
            for audio in audio_items:
                wav_bytes, sample_rate = self._audio_dict_to_wav_bytes(audio)
                transcripts.append(self._transcribe_audio(wav_bytes, sample_rate))
            results.append(" ".join(text for text in transcripts if text))
        return results

    def loglikelihood(self, requests: List[Any]) -> List[Tuple[float, bool]]:
        return [(0.0, True) for _ in requests]

    def loglikelihood_rolling(self, requests: List[Any]) -> List[float]:
        return [0.0 for _ in requests]
