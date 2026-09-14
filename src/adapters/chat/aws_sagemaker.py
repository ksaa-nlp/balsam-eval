"""Amazon SageMaker Runtime chat adapter for LM Evaluation Harness.

Optional dependency: boto3
"""

import base64
import copy
import io
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
from lm_eval.api.model import LM  # type: ignore[import-untyped]
from lm_eval.api.registry import register_model  # type: ignore[import-untyped]
from PIL import Image

try:
    import boto3  # type: ignore[import-not-found]
except ImportError:
    boto3 = None  # type: ignore[assignment]


def _parse_messages(prompt: Any) -> List[Dict[str, Any]]:
    value = prompt.prompt if hasattr(prompt, "prompt") else prompt
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    return [{"role": "user", "content": str(value)}]


def _image_data_url(image: Image.Image) -> str:
    output = io.BytesIO()
    converted = image
    image_format = (image.format or "png").lower()
    image_format = {"jpg": "jpeg"}.get(image_format, image_format)
    if image_format not in {"png", "jpeg", "gif", "webp"}:
        image_format = "png"
    if image_format == "jpeg" and image.mode not in {"RGB", "L"}:
        converted = image.convert("RGB")
    elif image_format == "png" and image.mode not in {"1", "L", "LA", "P", "RGB", "RGBA"}:
        converted = image.convert("RGBA" if "A" in image.getbands() else "RGB")
    converted.save(output, format=image_format.upper())
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/{image_format};base64,{encoded}"


def _audio_part(audio: Dict[str, Any]) -> Dict[str, Any]:
    output = io.BytesIO()
    samples = np.asarray(audio["array"], dtype=np.float32)
    sf.write(output, samples, audio["sampling_rate"], format="WAV", subtype="PCM_16")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return {"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}}


def _inject_multimodal(
    messages: List[Dict[str, Any]], auxiliary: Dict[str, Any]
) -> List[Dict[str, Any]]:
    parts = [
        {"type": "image_url", "image_url": {"url": _image_data_url(image)}}
        for image in auxiliary.get("visual") or []
    ]
    parts.extend(_audio_part(audio) for audio in auxiliary.get("audio") or [])
    if not parts:
        return messages

    result = copy.deepcopy(messages)
    target = next((message for message in reversed(result) if message.get("role") == "user"), None)
    if target is None:
        target = {"role": "user", "content": []}
        result.append(target)
    content = target.get("content", "")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    elif not isinstance(content, list):
        content = [{"type": "text", "text": str(content)}]
    target["content"] = parts + content
    return result


@register_model("sagemaker-chat", "sagemaker")
class SageMakerChatLM(LM):
    """JSON chat adapter for caller-managed SageMaker real-time endpoints.

    Multimodal blocks use OpenAI's ``image_url`` and ``input_audio`` shapes;
    acceptance depends entirely on the deployed endpoint handler.
    """

    MULTIMODAL = True

    def __init__(
        self,
        model: Optional[str] = None,
        model_name: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        messages_key: Optional[str] = None,
        response_path: Optional[str] = None,
        content_type: str = "application/json",
        accept: str = "application/json",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **_kwargs: Any,
    ):
        super().__init__()
        if boto3 is None:
            raise ImportError(
                "boto3 is required for the sagemaker adapter. Install it with: pip install boto3"
            )
        self.endpoint_name = (
            endpoint_name
            or model
            or model_name
            or os.environ.get("SAGEMAKER_ENDPOINT_NAME")
            or os.environ.get("MODEL")
        )
        if not self.endpoint_name:
            raise ValueError(
                "No endpoint name provided. Set SAGEMAKER_ENDPOINT_NAME or pass endpoint_name."
            )
        self.region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        self.messages_key = messages_key or os.environ.get("SAGEMAKER_MESSAGES_KEY", "messages")
        self.response_path = response_path or os.environ.get(
            "SAGEMAKER_RESPONSE_PATH", "0.generated_text"
        )
        self.content_type = content_type
        self.accept = accept
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._tokenizer_name = self.endpoint_name
        endpoint_url = endpoint_url or os.environ.get("AWS_SAGEMAKER_ENDPOINT_URL")
        self.client = boto3.client(
            "sagemaker-runtime", region_name=self.region, endpoint_url=endpoint_url
        )

    @property
    def tokenizer_name(self) -> str:
        return self._tokenizer_name

    @property
    def max_sequence_length(self) -> int:
        """SageMaker endpoints do not expose tokenizer limits."""
        return 0

    @property
    def batch_size(self) -> int:
        """SageMaker requests are sent sequentially."""
        return 1

    @staticmethod
    def _request_data(
        instance: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        if hasattr(instance, "args"):
            prompt = instance.args[0]
            kwargs = instance.args[1] if len(instance.args) > 1 else {}
            auxiliary = instance.args[2] if len(instance.args) > 2 else {}
        elif isinstance(instance, dict):
            prompt = instance.get("prompt", "")
            kwargs = instance
            auxiliary = instance.get("auxiliary_args", {})
        else:
            prompt, kwargs, auxiliary = instance, {}, {}
        return (
            _parse_messages(prompt),
            kwargs if isinstance(kwargs, dict) else {},
            auxiliary if isinstance(auxiliary, dict) else {},
        )

    @staticmethod
    def _extract_path(payload: Any, path: str) -> str:
        value = payload
        for part in path.split(".") if path else []:
            if isinstance(value, list):
                value = value[int(part)]
            else:
                value = value[part]
        if isinstance(value, dict) and "content" in value:
            value = value["content"]
        return str(value).strip()

    def _invoke(
        self,
        messages: List[Dict[str, Any]],
        kwargs: Dict[str, Any],
        auxiliary: Dict[str, Any],
    ) -> str:
        parameters: Dict[str, Any] = {
            "max_new_tokens": kwargs.get("max_gen_toks", kwargs.get("max_tokens", self.max_tokens)),
            "temperature": kwargs.get("temperature", self.temperature),
        }
        stop = kwargs.get("until", kwargs.get("stop"))
        if isinstance(stop, str):
            stop = [stop]
        if stop:
            parameters["stop"] = [item for item in stop if item]
        payload = {
            self.messages_key: _inject_multimodal(messages, auxiliary),
            "parameters": parameters,
        }
        response = self.client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType=self.content_type,
            Accept=self.accept,
            Body=json.dumps(payload).encode("utf-8"),
        )
        decoded = json.loads(response["Body"].read())
        return self._extract_path(decoded, self.response_path)

    def generate_until(self, requests: List[Any]) -> List[str]:
        return [self._invoke(*self._request_data(instance)) for instance in requests]

    def loglikelihood(self, requests: List[Any]) -> List[Tuple[float, bool]]:
        return [(0.0, True) for _ in requests]

    def loglikelihood_rolling(self, requests: List[Any]) -> List[float]:
        return [0.0 for _ in requests]

    def apply_chat_template(
        self, chat_history: Any, add_generation_prompt: bool = True
    ) -> str:
        if isinstance(chat_history, str):
            return chat_history
        return json.dumps(chat_history)
