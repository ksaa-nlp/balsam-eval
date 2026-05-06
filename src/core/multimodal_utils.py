"""
Utilities for multimodal datasets in Balsam Eval.
"""

import os
from typing import Any, Dict, List, Union

import librosa
import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
from PIL import Image


def doc_to_image(doc: Dict[str, Any]) -> List[Image.Image]:
    """
    Extract PIL Images from a document.

    Args:
        doc: Document dict that may have 'images' field

    Returns:
        List of PIL.Image objects
    """
    images: List[Image.Image] = []

    image_paths = doc.get("images", [])
    for path in image_paths:
        try:
            img = Image.open(path)
            images.append(img)
        except (OSError, IOError) as e:
            print(f"Warning: Failed to load image {path}: {e}")
            continue

    return images


def doc_to_audio(doc: Dict[str, Any]) -> List[Dict]:
    """
    Extract audio data from a document in HuggingFace format.

    Args:
        doc: Document dict that may have 'audio' field

    Returns:
        List of audio dicts: {'array': np.ndarray, 'sampling_rate': int}
    """
    audios: List[Dict] = []

    audio_paths = doc.get("audio", [])
    for path in audio_paths:
        if os.path.exists(path):
            audio_dict = load_audio_file(path)
            if audio_dict is not None:
                audios.append(audio_dict)
        else:
            print(f"Warning: Audio file not found: {path}")

    return audios


def load_audio_file(file_path: str) -> Union[Dict[str, Any], None]:
    """
    Load an audio file and convert it to HuggingFace dataset format.

    Args:
        file_path: Path to the audio file

    Returns:
        Dictionary with 'array' (numpy array of audio samples) and 'sampling_rate' (int)
        Returns None if loading fails
    """
    if not os.path.exists(file_path):
        print(f"Audio file not found: {file_path}")
        return None

    try:
        audio_array, sampling_rate = librosa.load(file_path, sr=None)
        return {"array": audio_array, "sampling_rate": sampling_rate}
    except (OSError, ValueError, RuntimeError) as e:
        print(f"Warning: Failed to load audio with librosa: {e}")
        try:
            audio_array, sampling_rate = sf.read(file_path)
            # Convert to float32 and normalize if needed
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)

            # Handle multi-channel audio by taking first channel
            if len(audio_array.shape) > 1:
                audio_array = audio_array[:, 0]

            return {"array": audio_array, "sampling_rate": sampling_rate}
        except (OSError, ValueError, RuntimeError) as e2:
            print(f"Warning: Failed to load audio with soundfile: {e2}")

    print(
        f"Error: Could not load audio file {file_path}. Please install librosa or soundfile."
    )
    return None


def doc_to_text(doc: Dict[str, Any]) -> str:
    """
    Format the text prompt from a document.

    Args:
        doc: Document dict with 'input' and optional 'instruction' fields

    Returns:
        Formatted text prompt
    """
    # Combine instruction and input
    parts = []

    if doc.get("instruction"):
        parts.append(doc["instruction"])

    if doc.get("input"):
        parts.append(doc["input"])

    return "\n".join(parts)
