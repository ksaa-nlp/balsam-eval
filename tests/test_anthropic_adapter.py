"""Unit tests for Anthropic adapter with reasoning models support."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from lm_eval.api.registry import get_model

from src.adapters.chat.anthropic import (
    AnthropicAudioLM,
    AnthropicNonRetriableError,
    _is_reasoning_model,
)


class TestAnthropicReasoningModelDetection:
    """Test detection of reasoning/thinking models."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "claude-fable-5-1",
            "claude-haiku-5-5",
            "claude-3-7-sonnet-20250219",
            "claude-opus-4-0",
            "claude-opus-5-0",
            "claude-sonnet-4-0",
            "claude-5-sonnet",
            "claude-3-7-thinking",
            "claude-model-reasoning",
        ],
    )
    def test_reasoning_models_detected(self, model_name: str) -> None:
        assert _is_reasoning_model(model_name) is True

    @pytest.mark.parametrize(
        "model_name",
        [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-2.1",
            "claude-2.0",
        ],
    )
    def test_standard_models_not_detected_as_reasoning(self, model_name: str) -> None:
        assert _is_reasoning_model(model_name) is False

    def test_empty_or_none_model_returns_false(self) -> None:
        assert _is_reasoning_model(None) is False
        assert _is_reasoning_model("") is False

    def test_is_reasoning_env_var_override(self) -> None:
        with patch.dict(os.environ, {"IS_REASONING": "1"}):
            assert _is_reasoning_model("claude-3-5-sonnet-20241022") is True
            assert _is_reasoning_model("") is True

        with patch.dict(os.environ, {"IS_REASONING": "true"}):
            assert _is_reasoning_model("any-model") is True

        with patch.dict(os.environ, {"IS_REASONING": "0"}):
            assert _is_reasoning_model("claude-3-5-sonnet-20241022") is False


class TestAnthropicPayloadCreation:
    """Test payload creation and dynamic parameter stripping."""

    def test_strips_parameters_for_reasoning_model(self) -> None:
        model = object.__new__(AnthropicAudioLM)
        model.model = "claude-haiku-5-5"
        model._max_gen_toks = 4096
        model._force_reasoning = False

        messages = [{"role": "user", "content": "Hello"}]
        gen_kwargs = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_tokens": 1024,
            "until": ["\n\nHuman:"],
        }

        payload = model._create_payload(messages, gen_kwargs=gen_kwargs)

        assert "temperature" not in payload
        assert "top_p" not in payload
        assert "top_k" not in payload
        assert payload.get("max_tokens") == 1024
        assert payload.get("model") == "claude-haiku-5-5"

    def test_preserves_temperature_for_standard_model(self) -> None:
        model = object.__new__(AnthropicAudioLM)
        model.model = "claude-3-5-sonnet-20241022"
        model._max_gen_toks = 4096
        model._force_reasoning = False

        messages = [{"role": "user", "content": "Hello"}]
        gen_kwargs = {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 512,
        }

        payload = model._create_payload(messages, gen_kwargs=gen_kwargs)

        assert payload.get("temperature") == 0.7
        assert payload.get("top_p") == 0.95
        assert payload.get("max_tokens") == 512

    def test_normalizes_string_prompt_to_chat_dict(self) -> None:
        model = object.__new__(AnthropicAudioLM)
        model.model = "claude-haiku-5-5"
        model._max_gen_toks = 4096
        model._force_reasoning = False

        payload = model._create_payload("Simple string prompt", gen_kwargs={})
        assert payload.get("messages") == [
            {"role": "user", "content": [{"type": "text", "text": "Simple string prompt"}]}
        ]


class TestAnthropicParseGenerations:
    """Test parse_generations handling thinking and reasoning blocks."""

    def test_filters_thinking_blocks_safely(self) -> None:
        model = object.__new__(AnthropicAudioLM)

        outputs = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Let me reason through this..."},
                {"type": "text", "text": "Here is the final answer."},
            ],
        }

        results = model.parse_generations([outputs])
        assert results == ["Here is the final answer."]

    def test_handles_multiple_text_blocks(self) -> None:
        model = object.__new__(AnthropicAudioLM)

        outputs = {
            "content": [
                {"type": "thinking", "thinking": "Thinking..."},
                {"type": "text", "text": "Part 1. "},
                {"type": "text", "text": "Part 2."},
            ]
        }

        results = model.parse_generations([outputs])
        assert results == ["Part 1. Part 2."]


class TestAnthropicErrorHandling:
    """Test auto-correction and non-retriable client error handling."""

    def test_auto_corrects_on_deprecation_400(self) -> None:
        async def _test() -> None:
            model = object.__new__(AnthropicAudioLM)
            model.model = "claude-unknown-reasoning"
            model.base_url = "https://api.anthropic.com/v1/messages"
            model.header = {"x-api-key": "test"}
            model._max_gen_toks = 4096
            model._force_reasoning = False
            model.cache_hook = MagicMock()
            model.create_message = lambda m: m
            model.parse_generations = lambda outputs: ["generated text"]

            first_resp = MagicMock()
            first_resp.ok = False
            first_resp.status = 400
            first_resp.text = AsyncMock(
                return_value='{"type":"error","error":{"type":"invalid_request_error","message":"`temperature` is deprecated for this model."}}'
            )

            second_resp = MagicMock()
            second_resp.ok = True
            second_resp.status = 200
            second_resp.json = AsyncMock(return_value={"content": [{"type": "text", "text": "generated text"}]})

            session = MagicMock()
            session.post = MagicMock(
                side_effect=[
                    AsyncMock(
                        __aenter__=AsyncMock(return_value=first_resp),
                        __aexit__=AsyncMock(return_value=False),
                    ),
                    AsyncMock(
                        __aenter__=AsyncMock(return_value=second_resp),
                        __aexit__=AsyncMock(return_value=False),
                    ),
                ]
            )
            sem = asyncio.Semaphore(1)

            result = await model.amodel_call(
                session=session,
                sem=sem,
                messages=[{"role": "user", "content": "hi"}],
                gen_kwargs={"temperature": 1.0},
            )

            assert result == ["generated text"]
            assert model._force_reasoning is True
            assert session.post.call_count == 2

        asyncio.run(_test())

    def test_fails_fast_on_client_error(self) -> None:
        async def _test() -> None:
            model = object.__new__(AnthropicAudioLM)
            model.model = "claude-3-5-sonnet-20241022"
            model.base_url = "https://api.anthropic.com/v1/messages"
            model.header = {"x-api-key": "test"}
            model._max_gen_toks = 4096
            model._force_reasoning = False
            model.create_message = lambda m: m

            resp = MagicMock()
            resp.ok = False
            resp.status = 400
            resp.text = AsyncMock(
                return_value='{"type":"error","error":{"type":"invalid_request_error","message":"max_tokens exceeds limit"}}'
            )

            session = MagicMock()
            session.post = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=resp),
                    __aexit__=AsyncMock(return_value=False),
                )
            )
            sem = asyncio.Semaphore(1)

            with pytest.raises(AnthropicNonRetriableError, match=r"Anthropic client error \(400\)"):
                await model.amodel_call(
                    session=session,
                    sem=sem,
                    messages=[{"role": "user", "content": "hi"}],
                    gen_kwargs={"max_tokens": 999999},
                )

        asyncio.run(_test())


class TestAnthropicModelRegistry:
    """Test registry binds anthropic to AnthropicAudioLM."""

    def test_anthropic_model_registered(self) -> None:
        model_cls = get_model("anthropic")
        assert model_cls is AnthropicAudioLM
