"""Chat completion adapters with audio support."""
from src.adapters.chat.openai import OpenAIAudioLM  # noqa: F401
from src.adapters.chat.anthropic import AnthropicAudioLM  # noqa: F401
from src.adapters.chat.local import LocalAudioLM  # noqa: F401
from src.adapters.chat.cohere import CohereAudioLM  # noqa: F401
from src.adapters.chat.gemini import GeminiLM  # noqa: F401
from src.adapters.chat.groq import GroqLM  # noqa: F401
from src.adapters.chat.azure_openai import AzureOpenAIChatLM  # noqa: F401
from src.adapters.chat.aixplain import AiXplainChatLM  # noqa: F401
from src.adapters.chat.huggingface_chat import HuggingFaceChatLM  # noqa: F401
from src.adapters.chat.aws_bedrock import BedrockChatLM  # noqa: F401
from src.adapters.chat.aws_sagemaker import SageMakerChatLM  # noqa: F401
