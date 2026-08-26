"""Automatic speech recognition adapters."""
from src.adapters.asr.openai_asr import OpenAIWhisperLM  # noqa: F401
from src.adapters.asr.google_stt import GoogleSTTLM  # noqa: F401
from src.adapters.asr.azure_stt import AzureSTTLM  # noqa: F401
from src.adapters.asr.huggingface_asr import HuggingFaceASRLM  # noqa: F401
from src.adapters.asr.nemo_asr import NeMoASRLM  # noqa: F401
from src.adapters.asr.ibm_stt import IBMSTTLM  # noqa: F401
from src.adapters.asr.qwen_asr import QwenASRLM  # noqa: F401
from src.adapters.asr.cohere_asr import CohereASRLM  # noqa: F401
from src.adapters.asr.deepgram_stt import DeepgramSTTLM  # noqa: F401
from src.adapters.asr.speechmatics_stt import SpeechmaticsSTTLM  # noqa: F401
from src.adapters.asr.assemblyai_stt import AssemblyAISTTLM  # noqa: F401
from src.adapters.asr.elevenlabs_stt import ElevenLabsSTTLM  # noqa: F401
from src.adapters.asr.gladia_stt import GladiaSTTLM  # noqa: F401
from src.adapters.asr.revai_stt import RevAISTTLM  # noqa: F401
from src.adapters.asr.aws_transcribe import AWSTranscribeLM  # noqa: F401
