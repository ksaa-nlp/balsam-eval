"""Adapters for chat, ASR, and judge model integrations."""
from src.adapters.chat.openai import OpenAIAudioLM  # noqa: F401
from src.adapters.chat.anthropic import AnthropicAudioLM  # noqa: F401
from src.adapters.chat.local import LocalAudioLM  # noqa: F401
from src.adapters.chat.cohere import CohereAudioLM  # noqa: F401
from src.adapters.chat.gemini import GeminiLM  # noqa: F401
from src.adapters.chat.groq import GroqLM  # noqa: F401
from src.adapters.asr.openai_asr import OpenAIWhisperLM  # noqa: F401
from src.adapters.asr.google_stt import GoogleSTTLM  # noqa: F401
from src.adapters.asr.azure_stt import AzureSTTLM  # noqa: F401
from src.adapters.judge.local_model import LocalModelEdited  # noqa: F401
