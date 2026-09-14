import pytest

from src.adapters.utils import (
    convert_anthropic_url,
    get_max_tokens_config,
    process_adapter_and_url,
)


@pytest.fixture(autouse=True)
def clear_reasoning_environment(monkeypatch):
    monkeypatch.delenv("IS_REASONING", raising=False)
    monkeypatch.delenv("MAX_TOKENS", raising=False)


@pytest.mark.parametrize(
    ("adapter", "model", "expected"),
    [
        ("openai", "gpt-5", {"max_completion_tokens": 8192}),
        ("openai", "gpt-5.2", {"max_completion_tokens": 128000}),
        (
            "local-chat-completions",
            "deepseek-r1",
            {"max_completion_tokens": 8192, "max_tokens": 8192},
        ),
        ("gemini", "2.0-flash-thinking", {"max_tokens": 8192}),
        ("openai", "gpt-4", {"max_tokens": 1024}),
        ("unknown", "model", {"max_tokens": 1024}),
    ],
)
def test_get_max_tokens_config_detects_adapter_and_model(adapter, model, expected):
    assert get_max_tokens_config(adapter, model) == expected


@pytest.mark.parametrize("adapter", ["openai", "openai-chat-completions"])
def test_reasoning_env_uses_openai_completion_token_name(monkeypatch, adapter):
    monkeypatch.setenv("IS_REASONING", "1")
    monkeypatch.setenv("MAX_TOKENS", "1234")

    assert get_max_tokens_config(adapter, "anything") == {
        "max_completion_tokens": 1234
    }


def test_reasoning_env_uses_standard_token_name_for_other_adapters(monkeypatch):
    monkeypatch.setenv("IS_REASONING", "1")
    monkeypatch.delenv("MAX_TOKENS", raising=False)

    assert get_max_tokens_config("gemini", "anything") == {"max_tokens": 8192}


@pytest.mark.parametrize("value", ["invalid", "0", "-1"])
def test_reasoning_env_rejects_invalid_max_tokens(monkeypatch, value):
    monkeypatch.setenv("IS_REASONING", "1")
    monkeypatch.setenv("MAX_TOKENS", value)

    with pytest.raises(ValueError, match="MAX_TOKENS"):
        get_max_tokens_config("openai", "model")


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (None, "https://api.anthropic.com/v1/messages"),
        ("  ", "https://api.anthropic.com/v1/messages"),
        (
            "https://api.anthropic.com/v1/messages",
            "https://api.anthropic.com/v1/messages",
        ),
        (
            "https://api.anthropic.com/v1/chat/completions",
            "https://api.anthropic.com/v1/messages",
        ),
        (
            "https://api.anthropic.com",
            "https://api.anthropic.com/v1/messages",
        ),
        (
            "https://api.anthropic.com/v1",
            "https://api.anthropic.com/v1/messages",
        ),
        (
            "https://proxy/v1/chat/completions?key=x",
            "https://proxy/v1/chat/completions?key=x",
        ),
        ("https://unknown.example/api", "https://unknown.example/api"),
    ],
)
def test_convert_anthropic_url(url, expected):
    assert convert_anthropic_url(url) == expected


def test_process_adapter_converts_anthropic_without_output(capsys):
    result = process_adapter_and_url(
        "anthropic-chat-completions", "https://api.anthropic.com", verbose=False
    )

    assert result == (
        "anthropic",
        "https://api.anthropic.com/v1/messages",
    )
    assert capsys.readouterr().out == ""


def test_process_adapter_preserves_other_adapters():
    assert process_adapter_and_url("gemini", "https://example", verbose=False) == (
        "gemini",
        "https://example",
    )


def test_process_openai_alias_uses_validating_adapter_and_preserves_url():
    assert process_adapter_and_url(
        "openai-chat-completions", "https://example", verbose=False
    ) == (
        "openai",
        "https://example",
    )


def test_process_native_anthropic_normalizes_public_url_and_preserves_custom_host():
    assert process_adapter_and_url(
        "anthropic", "https://api.anthropic.com/v1/chat/completions", verbose=False
    ) == ("anthropic", "https://api.anthropic.com/v1/messages")
    assert process_adapter_and_url(
        "anthropic-chat-completions", "https://private.example/messages", verbose=False
    ) == ("local-chat-completions", "https://private.example/messages")


def test_anthropic_alias_preserves_custom_openai_compatible_endpoint_exactly():
    endpoint = "https://gateway.example/v1/chat/completions?tenant=one"
    assert process_adapter_and_url(
        "anthropic-chat-completions", endpoint, verbose=False
    ) == ("local-chat-completions", endpoint)
