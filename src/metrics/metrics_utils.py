"""Shared utilities for metrics implementations.

This module contains only truly shared code used by multiple metrics.
Metric-specific functions should remain in their respective metric files.
"""

import logging
import math
import re
import unicodedata

logger = logging.getLogger(__name__)

try:
    from pyarabic import araby
except ImportError:
    araby = None

# Punctuation handling shared across BLEU and ROUGE metrics.
all_punctuations = "".join(
    chr(i) for i in range(0x110000) if unicodedata.category(chr(i)).startswith("P")
)
OTHERS = """`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!"…"–ـ"""
all_punctuations += "".join(char for char in OTHERS if char not in all_punctuations)
_PUNCTUATION_PATTERN = re.compile(f"([{re.escape(all_punctuations)}])")


def clamp_score(score: float) -> float:
    """Return a finite metric score constrained to the inclusive [0, 1] range."""
    value = float(score)
    if math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, value))


def prepare_text_with_punctuation(
    text: str,
    change_curly_braces: bool = False,
    remove_diacritics: bool = False,
) -> str:
    """Prepare text for evaluation by handling punctuation and special characters.

    This function is used by both BLEU and ROUGE metrics for text normalization.

    Args:
        text: Input text
        change_curly_braces: Whether to change curly braces to square brackets
        remove_diacritics: Whether to remove Arabic diacritics (requires pyarabic)

    Returns:
        Prepared text with normalized punctuation
    """
    if not isinstance(text, str):
        text = str(text)

    # Add spaces around punctuation
    text = _PUNCTUATION_PATTERN.sub(r" \1 ", text)

    if change_curly_braces:
        text = text.replace("{", "[").replace("}", "]")

    if remove_diacritics:
        if araby:
            text = araby.strip_diacritics(text)
        else:
            logger.warning("pyarabic not installed, skipping diacritic removal")

    return text
