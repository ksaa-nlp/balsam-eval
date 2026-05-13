"""Automatic speech recognition adapters."""
from src.adapters.asr.openai_asr import OpenAIWhisperLM  # noqa: F401
from src.adapters.asr.google_stt import GoogleSTTLM  # noqa: F401
from src.adapters.asr.azure_stt import AzureSTTLM  # noqa: F401
from src.adapters.asr.huggingface_asr import HuggingFaceASRLM  # noqa: F401
from src.adapters.asr.nemo_asr import NeMoASRLM  # noqa: F401
from src.adapters.asr.ibm_stt import IBMSTTLM  # noqa: F401
from src.adapters.asr.qwen_asr import QwenASRLM  # noqa: F401
