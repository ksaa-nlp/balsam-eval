"""Amazon Bedrock Converse adapter for LM Evaluation Harness.

Optional dependency: boto3
"""

import io
import json
import os
from typing import Any, Dict, List, Optional, Tuple

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


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(block.get("text", "")) if isinstance(block, dict) else str(block)
            for block in content
        ).strip()
    return str(content)


def _image_block(image: Image.Image) -> Dict[str, Any]:
    """Convert a PIL image to a Bedrock Converse image content block."""
    image_format = (image.format or "png").lower()
    image_format = {"jpg": "jpeg"}.get(image_format, image_format)
    if image_format not in {"png", "jpeg", "gif", "webp"}:
        image_format = "png"

    output = io.BytesIO()
    converted = image
    if image_format == "jpeg" and image.mode not in {"RGB", "L"}:
        converted = image.convert("RGB")
    elif image_format == "png" and image.mode not in {"1", "L", "LA", "P", "RGB", "RGBA"}:
        converted = image.convert("RGBA" if "A" in image.getbands() else "RGB")
    converted.save(output, format=image_format.upper())
    return {"image": {"format": image_format, "source": {"bytes": output.getvalue()}}}


@register_model("aws-bedrock", "bedrock")
class BedrockChatLM(LM):
    """Chat adapter using Bedrock Runtime's model-independent Converse API."""

    MULTIMODAL = True

    def __init__(
        self,
        model: Optional[str] = None,
        model_name: Optional[str] = None,
        region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        top_p: Optional[float] = None,
        **_kwargs: Any,
    ):
        super().__init__()
        if boto3 is None:
            raise ImportError(
                "boto3 is required for the bedrock adapter. Install it with: pip install boto3"
            )
        self.model_name = (
            model
            or model_name
            or os.environ.get("BEDROCK_MODEL_ID")
            or os.environ.get("MODEL")
        )
        if not self.model_name:
            raise ValueError("No model ID provided. Set BEDROCK_MODEL_ID or pass model.")
        self.region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self._tokenizer_name = self.model_name
        endpoint_url = endpoint_url or os.environ.get("AWS_BEDROCK_ENDPOINT_URL")
        self.client = boto3.client(
            "bedrock-runtime", region_name=self.region, endpoint_url=endpoint_url
        )

    @property
    def tokenizer_name(self) -> str:
        return self._tokenizer_name

    @property
    def max_sequence_length(self) -> int:
        """Bedrock chat requests do not expose tokenizer limits."""
        return 0

    @property
    def batch_size(self) -> int:
        """Bedrock requests are sent sequentially."""
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

    def _converse(
        self,
        messages: List[Dict[str, Any]],
        kwargs: Dict[str, Any],
        auxiliary: Dict[str, Any],
    ) -> str:
        if auxiliary.get("audio"):
            raise NotImplementedError(
                "Amazon Bedrock Converse does not support audio input"
            )

        system: List[Dict[str, Any]] = []
        conversation: List[Dict[str, Any]] = []
        for message in messages:
            role = message.get("role", "user")
            content = _text(message.get("content", ""))
            if role == "system":
                system.append({"text": content})
            else:
                conversation.append({
                    "role": role if role in {"user", "assistant"} else "user",
                    "content": [{"text": content}],
                })

        images = auxiliary.get("visual") or []
        if images:
            target = next(
                (message for message in reversed(conversation) if message["role"] == "user"),
                None,
            )
            if target is None:
                target = {"role": "user", "content": []}
                conversation.append(target)
            target["content"].extend(_image_block(image) for image in images)

        inference: Dict[str, Any] = {
            "maxTokens": kwargs.get("max_gen_toks", kwargs.get("max_tokens", self.max_tokens)),
            "temperature": kwargs.get("temperature", self.temperature),
        }
        top_p = kwargs.get("top_p", self.top_p)
        if top_p is not None:
            inference["topP"] = top_p
        stop = kwargs.get("until", kwargs.get("stop"))
        if isinstance(stop, str):
            stop = [stop]
        if stop:
            inference["stopSequences"] = [item for item in stop if item]

        request: Dict[str, Any] = {
            "modelId": self.model_name,
            "messages": conversation,
            "inferenceConfig": inference,
        }
        if system:
            request["system"] = system
        response = self.client.converse(**request)
        blocks = response["output"]["message"]["content"]
        return "".join(block.get("text", "") for block in blocks).strip()

    def generate_until(self, requests: List[Any]) -> List[str]:
        return [self._converse(*self._request_data(instance)) for instance in requests]

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
