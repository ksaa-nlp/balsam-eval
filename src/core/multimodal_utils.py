"""
Utilities for multimodal datasets in Balsam Eval.
Provides doc_to_visual function for lmms-eval compatibility with both images and audio.
"""

import os
from typing import Dict, Any, Sequence, Union

import numpy as np
from PIL import Image
import librosa
import soundfile as sf  # type: ignore[import-untyped]


def doc_to_visual(doc: Dict[str, Any]) -> Sequence[Union[Image.Image, Dict]]:
    """
    Extract PIL Images and audio data from a document for lmms-eval multimodal processing.

    lmms_eval supports both images and audio through doc_to_visual:
    - Images are returned as PIL.Image objects
    - Audio files are returned as HuggingFace format dicts:
      {'array': np.ndarray, 'sampling_rate': int}

    Args:
        doc: Document dict that may have 'images' and 'audio' fields

    Returns:
        Sequence of PIL.Image objects and audio data dicts

    Example:
        >>> doc = {"images": ["/path/to/image1.png"], "audio": ["/path/to/audio1.wav"]}
        >>> visuals = doc_to_visual(doc)
        >>> len(visuals)
        2
    """
    visuals: list[Union[Image.Image, Dict]] = []

    # Handle images
    image_paths = doc.get("images", [])
    for path in image_paths:
        try:
            img = Image.open(path)
            visuals.append(img)
        except (OSError, IOError) as e:
            print(f"Warning: Failed to load image {path}: {e}")
            continue

    # Handle audio files (load and convert to HuggingFace format)
    audio_paths = doc.get("audio", [])
    for path in audio_paths:
        if os.path.exists(path):
            audio_dict = load_audio_file(path)
            if audio_dict is not None:
                visuals.append(audio_dict)
        else:
            print(f"Warning: Audio file not found: {path}")

    return visuals


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
