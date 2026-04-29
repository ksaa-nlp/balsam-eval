"""Shared utilities for metrics implementations.

This module contains only truly shared code used by multiple metrics.
Metric-specific functions should remain in their respective metric files.
"""

import logging
import re
import sys
import unicodedata

logger = logging.getLogger(__name__)

# Punctuation handling - shared across BLEU and ROUGE metrics
_PUNCT_TABLE = dict.fromkeys(
    i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")
)
ALL_PUNCTUATIONS = "".join(chr(p) for p in _PUNCT_TABLE)
OTHERS = """`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!"…"–ـ"""
ALL_PUNCTUATIONS += "".join([o for o in OTHERS if o not in ALL_PUNCTUATIONS])


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
    text = re.sub("([" + ALL_PUNCTUATIONS + "])", " \\1 ", text)

    if change_curly_braces:
        text = text.replace("{", "[").replace("}", "]")

    if remove_diacritics:
        try:
            from pyarabic import araby

            text = araby.strip_diacritics(text)
        except ImportError:
            logger.warning("pyarabic not installed, skipping diacritic removal")

    return text
