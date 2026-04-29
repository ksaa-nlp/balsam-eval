"""Accuracy metric implementation for MCQ evaluation."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from lmms_eval.api import registry as le_registry
from lmms_eval.api.registry import register_aggregation, register_metric

from src.metrics_registry import BaseMetric, MetricConfig, get_metrics_registry

logger = logging.getLogger(__name__)


def extract_first_word_or_line(text: str) -> str:
    """Extract the first word or short line from text.

    For MCQ answers, handles cases like "Answer: A" → "A"
    or "The answer is Paris" → "Paris".

    Args:
        text: Input text

    Returns:
        Extracted word or line
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.strip()
    if not text:
        logger.debug("extract_first_word_or_line: Empty text after strip")
        return ""

    first_line = text.split("\n")[0].strip()
    # Remove all non-alphanumeric characters from the end
    first_line = re.sub(r"[^\w\s]+$", "", first_line, flags=re.UNICODE)

    logger.debug("extract_first_word_or_line: first_line='%s'", first_line)

    # Handle common patterns like "Answer: A" or "The answer is: Paris"
    colon_match = re.match(r"^([^:]+):\s*([^\s]+)", first_line, re.IGNORECASE)
    if colon_match:
        prefix = colon_match.group(1).strip().lower()
        if any(
            word in prefix
            for word in [
                "answer",
                "response",
                "result",
                "choice",
                "option",
                "الإجابة",
                "الجواب",
            ]
        ):
            extracted = colon_match.group(2).strip()
            logger.debug(
                "extract_first_word_or_line: Extracted after colon pattern: '%s'",
                extracted,
            )
            extracted = re.sub(r"[^\w\s]", "", extracted, flags=re.UNICODE).strip()
            return extracted

    # If line is short (≤3 words), return it as-is
    if len(first_line.split()) <= 3:
        logger.debug(
            "extract_first_word_or_line: Returning short line (≤3 words): '%s'",
            first_line,
        )
        return first_line

    # Otherwise, extract just the first word
    first_word = first_line.split()[0] if first_line.split() else first_line
    first_word = re.sub(r"[^\w\s]", "", first_word, flags=re.UNICODE).strip()

    logger.debug("extract_first_word_or_line: Extracted first word: '%s'", first_word)
    return first_word


def normalize_text(
    text: str,
    mcq_mapping: Optional[dict[str, str]] = None,
) -> str:
    """Normalize text for comparison, with special handling for MCQ answers.

    Args:
        text: The text to normalize (could be a letter or full answer text)
        mcq_mapping: Dict mapping letters to full option text

    Returns:
        Normalized text or empty string if text is empty/None
    """
    logger.debug(
        "normalize_text called with text='%s', mcq_mapping=%s", text, mcq_mapping
    )

    # Handle None or empty cases first
    if text is None or text == "":
        logger.debug("normalize_text: Received None or empty string")
        return ""

    if not isinstance(text, str):
        text = str(text)

    text = text.strip()

    # If mcq_mapping is provided, check if text is a letter key
    if mcq_mapping and text.upper() in mcq_mapping:
        logger.debug("normalize_text: Mapping letter '%s' to full text", text)
        mapped_text: str = mcq_mapping[text.upper()]
        return mapped_text

    # Remove all punctuation and extra whitespace
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip().lower()

    logger.debug("normalize_text: Normalized to '%s'", text)
    return text


def compute_accuracy(items: List[Tuple[Any, Any]]) -> float:
    """Compute accuracy score from reference and prediction pairs.

    Args:
        items: List of (reference, prediction) tuples

    Returns:
        Accuracy score between 0 and 1
    """
    total = 0
    correct = 0

    for ref, pred in items:
        if ref is None or pred is None:
            continue

        # Normalize both for comparison
        ref_norm = normalize_text(ref)
        pred_norm = normalize_text(pred)

        if ref_norm and pred_norm:
            total += 1
            if ref_norm == pred_norm:
                correct += 1

    return correct / total if total > 0 else 0.0


# Register aggregation function
if "accuracy" not in le_registry.AGGREGATION_REGISTRY:
    register_aggregation("accuracy")(compute_accuracy)


# Register metric function
if "accuracy" not in le_registry.METRIC_REGISTRY:
    register_metric(
        metric="accuracy",
        higher_is_better=True,
        output_type="generate_until",
        aggregation="accuracy",
    )(lambda items: items)


def process_results(doc: Dict[str, Any], results: Any) -> Dict[str, List[str]]:
    """Process results for accuracy evaluation.

    Extracts reference and prediction from document and model results,
    then prepares them for accuracy computation.

    Args:
        doc: Document containing reference output
        results: Model predictions (list or single value)

    Returns:
        Dictionary with accuracy data containing [reference, prediction]
    """
    preds = results[0] if isinstance(results, list) else results
    golds = doc["output"]

    # Extract first word/line for MCQ answers
    pred_extracted = extract_first_word_or_line(preds)
    gold_extracted = extract_first_word_or_line(golds)

    return {"accuracy": [gold_extracted, pred_extracted]}


class AccuracyMetric(BaseMetric):
    """Accuracy metric for YAML/task export.

    This metric class integrates with the metrics registry to provide
    accuracy evaluation for MCQ-based tasks.
    """

    def get_doc_to_text(self, original_doc_to_text: str) -> str:
        """Get the doc_to_text template for accuracy metric.

        Args:
            original_doc_to_text: Original doc_to_text template

        Returns:
            Original template (accuracy doesn't modify the prompt)
        """
        return original_doc_to_text

    def get_generation_kwargs(self) -> Dict[str, Any]:
        """Get generation kwargs for accuracy metric.

        Returns:
            Generation parameters (no sampling, stop on empty string)
        """
        return {"do_sample": False, "until": [""]}


# Register in custom registry
_accuracy_config = MetricConfig(
    name="accuracy",
    higher_is_better=True,
    aggregation_name="accuracy",
    process_results=process_results,
)
get_metrics_registry().register("accuracy", AccuracyMetric(_accuracy_config))
