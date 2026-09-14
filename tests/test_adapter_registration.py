"""Ensure every project adapter is registered during package import."""

import pytest
from lm_eval.api.registry import get_model

import src.adapters  # noqa: F401
from src.adapter_config import API_KEY_ENV_BY_ADAPTER, ASR_ADAPTERS


CHAT_ADAPTERS = {
    "aixplain",
    "anthropic",
    "azure-openai",
    "aws-bedrock",
    "bedrock",
    "cohere",
    "gemini",
    "groq",
    "huggingface-chat",
    "local-adapter",
    "openai",
    "sagemaker",
    "sagemaker-chat",
}


@pytest.mark.parametrize("adapter", sorted(CHAT_ADAPTERS | set(ASR_ADAPTERS)))
def test_adapter_is_registered(adapter):
    assert get_model(adapter)


def test_api_key_map_covers_key_based_custom_adapters():
    keyless_adapters = {"aws-bedrock", "bedrock", "sagemaker", "sagemaker-chat"}
    assert CHAT_ADAPTERS - keyless_adapters <= API_KEY_ENV_BY_ADAPTER.keys()
    assert ASR_ADAPTERS - {"aws-transcribe"} <= API_KEY_ENV_BY_ADAPTER.keys()
