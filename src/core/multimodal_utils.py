"""
Utilities for multimodal datasets in Balsam Eval.
Provides doc_to_visual function for lmms-eval compatibility.
"""

from pathlib import Path
from typing import List, Dict, Any
from PIL import Image


def doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    """
    Extract PIL Images from a document for lmms-eval multimodal processing.

    Args:
        doc: Document dict that may have an 'images' field with image paths

    Returns:
        List of PIL.Image objects

    Example:
        >>> doc = {"images": ["/path/to/image1.png", "/path/to/image2.jpg"]}
        >>> images = doc_to_visual(doc)
        >>> len(images)
        2
    """
    image_paths = doc.get("images", [])
    images = []

    for path in image_paths:
        try:
            img = Image.open(path)
            images.append(img)
        except Exception as e:
            print(f"Warning: Failed to load image {path}: {e}")
            continue

    return images


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
