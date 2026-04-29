"""
Utilities for multimodal datasets in Balsam Eval.
Provides doc_to_visual function for lmms-eval compatibility with both images and audio.
"""

import os
from typing import Dict, Any, Sequence, Union

from PIL import Image


def doc_to_visual(doc: Dict[str, Any]) -> Sequence[Union[Image.Image, str]]:
    """
    Extract PIL Images and audio file paths from a document for lmms-eval multimodal processing.

    lmms_eval supports both images and audio through doc_to_visual:
    - Images are returned as PIL.Image objects
    - Audio files are returned as string paths (lmms_eval detects them by extension)

    Args:
        doc: Document dict that may have 'images' and 'audio' fields

    Returns:
        Sequence of PIL.Image objects and audio file path strings

    Example:
        >>> doc = {"images": ["/path/to/image1.png"], "audio": ["/path/to/audio1.wav"]}
        >>> visuals = doc_to_visual(doc)
        >>> len(visuals)
        2
    """
    visuals: list[Union[Image.Image, str]] = []

    # Handle images
    image_paths = doc.get("images", [])
    for path in image_paths:
        try:
            img = Image.open(path)
            visuals.append(img)
        except (OSError, IOError) as e:
            print(f"Warning: Failed to load image {path}: {e}")
            continue

    # Handle audio files (return as string paths for lmms_eval)
    audio_paths = doc.get("audio", [])
    for path in audio_paths:
        if os.path.exists(path):
            visuals.append(path)
        else:
            print(f"Warning: Audio file not found: {path}")

    return visuals


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
