"""Chat completion adapters with audio support."""
from src.adapters.chat.openai import OpenAIAudioLM  # noqa: F401
from src.adapters.chat.anthropic import AnthropicAudioLM  # noqa: F401
from src.adapters.chat.local import LocalAudioLM  # noqa: F401
from src.adapters.chat.cohere import CohereAudioLM  # noqa: F401
from src.adapters.chat.gemini import GeminiLM  # noqa: F401
from src.adapters.chat.groq import GroqLM  # noqa: F401
