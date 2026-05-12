"""Automatic speech recognition adapters."""
from src.adapters.asr.openai_asr import OpenAIWhisperLM  # noqa: F401
from src.adapters.asr.google_stt import GoogleSTTLM  # noqa: F401
from src.adapters.asr.azure_stt import AzureSTTLM  # noqa: F401
