"""Accuracy metric implementation for MCQ evaluation."""

import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz
from lm_eval.api import registry as le_registry
from lm_eval.api.registry import register_aggregation, register_metric

from src.metrics_registry import BaseMetric, MetricConfig, get_metrics_registry
from src.metrics.metrics_utils import clamp_score

logger = logging.getLogger(__name__)

MCQ_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ARABIC_MCQ_LABELS = ("أ", "ب", "ج", "د", "هـ", "و")


def extract_first_word_or_line(text: str, reference: Optional[str] = None) -> str:
    """Extract an answer line without truncating multi-word answers.

    For MCQ answers, handles cases like "Answer: A" → "A"
    and can identify a reference answer at the start of a response line.

    Args:
        text: Input text
        reference: Optional expected answer used to locate a verbose answer prefix

    Returns:
        Extracted word or line
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.strip()
    if not text:
        logger.debug("extract_first_word_or_line: Empty text after strip")
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    first_line = lines[0] if lines else ""
    # Remove all non-alphanumeric characters from the end
    first_line = re.sub(r"[^\w\s]+$", "", first_line, flags=re.UNICODE)

    logger.debug("extract_first_word_or_line: first_line='%s'", first_line)

    if reference:
        reference_norm = normalize_text(reference)
        for line in lines:
            candidates = [line.lstrip("-*• \t")]
            if ":" in line:
                candidates.append(line.split(":", 1)[1].strip())
            for candidate in candidates:
                candidate_norm = normalize_text(candidate)
                if candidate_norm == reference_norm or candidate_norm.startswith(
                    f"{reference_norm} "
                ):
                    return reference

    # Handle common patterns like "Answer: A" or "The answer is: Paris"
    colon_match = re.match(r"^([^:]+):\s*(.+)$", first_line, re.IGNORECASE)
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

    logger.debug("extract_first_word_or_line: Returning first line: '%s'", first_line)
    return first_line


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

    # Ignore Arabic vocalization and tatweel, which do not change the answer.
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.replace("ـ", "")

    # Remove all punctuation and extra whitespace
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip().lower()

    logger.debug("normalize_text: Normalized to '%s'", text)
    return text


def resolve_mcq_answer(answer: str, options: Any) -> str:
    """Resolve a choice label to its option text when MCQ options are present."""
    if not isinstance(options, list) or not options:
        return answer

    normalized_answer = normalize_text(answer)
    for option in options:
        if normalized_answer == normalize_text(option):
            return str(option)

    label = answer.strip().rstrip(".)،:").strip()
    leading_label = re.match(
        r"^([A-Za-zأاإآبجدهـو])(?:\s*[).:،-]\s*|\s+)", answer.strip()
    )
    embedded_label = re.search(
        r"(?<!\w)([A-Za-zأاإآبجدهـو])\s*[).:،-](?:\s|$)", answer
    )
    label_match = leading_label or embedded_label
    if label_match:
        label = label_match.group(1)

    label_upper = label.upper()
    if len(label_upper) == 1 and label_upper in MCQ_LABELS:
        index = MCQ_LABELS.index(label_upper)
        if index < len(options):
            return str(options[index])

    if label.isdigit():
        index = int(label) - 1
        if 0 <= index < len(options):
            return str(options[index])

    for index, arabic_label in enumerate(ARABIC_MCQ_LABELS):
        equivalent_labels = {arabic_label}
        if arabic_label == "أ":
            equivalent_labels.update({"ا", "إ", "آ"})
        elif arabic_label == "هـ":
            equivalent_labels.add("ه")
        if label in equivalent_labels and index < len(options):
            return str(options[index])

    return answer


def compute_accuracy(
    items: List[Tuple[Any, Any]],
    fuzzy_threshold: float = 0.85,
    use_fuzzy: bool = False,
) -> float:
    """Compute accuracy score from reference and prediction pairs.

    Args:
        items: List of (reference, prediction) tuples
        fuzzy_threshold: Similarity threshold for fuzzy matching (0-1)
        use_fuzzy: Whether to use fuzzy matching or exact matching

    Returns:
        Accuracy score between 0 and 1
    """
    total = 0
    correct = 0

    for ref, pred in items:
        if ref is None:
            continue

        # Normalize both for comparison
        ref_norm = normalize_text(ref)
        pred_norm = normalize_text(pred)

        if not ref_norm:
            continue
        total += 1

        if pred_norm:
            if use_fuzzy:
                # Use fuzzy matching with RapidFuzz
                similarity_ratio = fuzz.ratio(ref_norm, pred_norm) / 100.0

                # Also check partial ratio for cases where one string contains the other
                partial_ratio = fuzz.partial_ratio(ref_norm, pred_norm) / 100.0

                # Consider it a match if either ratio meets the threshold
                if similarity_ratio >= fuzzy_threshold or partial_ratio >= fuzzy_threshold:
                    correct += 1
                    logger.debug(
                        "Fuzzy match: ref='%s', pred='%s', similarity=%.2f, partial=%.2f",
                        ref_norm,
                        pred_norm,
                        similarity_ratio,
                        partial_ratio,
                    )
            else:
                # Use exact matching
                if ref_norm == pred_norm:
                    correct += 1

    return clamp_score(correct / total) if total > 0 else 0.0


def compute_fuzzy_accuracy(items: List[Tuple[Any, Any]]) -> float:
    """Compute fuzzy accuracy score with default threshold.

    This is a wrapper that uses fuzzy matching with a default threshold of 0.85.

    Args:
        items: List of (reference, prediction) tuples

    Returns:
        Accuracy score between 0 and 1
    """
    return compute_accuracy(items, fuzzy_threshold=0.85, use_fuzzy=True)


# Register aggregation function
if "accuracy" not in le_registry.AGGREGATION_REGISTRY:
    register_aggregation("accuracy")(compute_accuracy)

if "fuzzy_accuracy" not in le_registry.AGGREGATION_REGISTRY:
    register_aggregation("fuzzy_accuracy")(compute_fuzzy_accuracy)


# Register metric function
if "accuracy" not in le_registry.METRIC_REGISTRY:
    register_metric(
        metric="accuracy",
        higher_is_better=True,
        output_type="generate_until",
        aggregation="accuracy",
    )(lambda items: items)

if "fuzzy_accuracy" not in le_registry.METRIC_REGISTRY:
    register_metric(
        metric="fuzzy_accuracy",
        higher_is_better=True,
        output_type="generate_until",
        aggregation="fuzzy_accuracy",
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
    preds = results[0] if isinstance(results, list) and results else ""
    golds = doc["output"]

    # Extract first word/line for MCQ answers
    pred_extracted = extract_first_word_or_line(preds, reference=golds)
    pred_extracted = resolve_mcq_answer(pred_extracted, doc.get("mcq"))
    gold_extracted = extract_first_word_or_line(golds)

    return {"accuracy": [gold_extracted, pred_extracted]}


def process_results_fuzzy(doc: Dict[str, Any], results: Any) -> Dict[str, List[str]]:
    """Process results for fuzzy accuracy evaluation.

    This function processes results specifically for fuzzy matching,
    which is more lenient with typos and minor variations.

    Args:
        doc: Document containing reference output
        results: Model predictions (list or single value)

    Returns:
        Dictionary with fuzzy_accuracy data containing [reference, prediction]
    """
    preds = results[0] if isinstance(results, list) and results else ""
    golds = doc["output"]

    # Extract first word/line for MCQ answers
    pred_extracted = extract_first_word_or_line(preds, reference=golds)
    pred_extracted = resolve_mcq_answer(pred_extracted, doc.get("mcq"))
    gold_extracted = extract_first_word_or_line(golds)

    return {"fuzzy_accuracy": [gold_extracted, pred_extracted]}


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
            Template with labeled choices when the document contains MCQ options
        """
        return (
            f"{original_doc_to_text}\n"
            "{% if mcq %}\n"
            "الخيارات:\n"
            "{% for option in mcq %}"
            '{{ "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[loop.index0] }}. {{ option }}\n'
            "{% endfor %}"
            "{% endif %}"
        )

    def get_generation_kwargs(self) -> Dict[str, Any]:
        """Get generation kwargs for accuracy metric.

        Returns:
            Generation parameters (no sampling, stop on empty string)
        """
        return {"do_sample": False, "until": []}


# Register in custom registry
_accuracy_config = MetricConfig(
    name="accuracy",
    higher_is_better=True,
    aggregation_name="accuracy",
    process_results=process_results,
)
get_metrics_registry().register("accuracy", AccuracyMetric(_accuracy_config))

# Register fuzzy accuracy in custom registry
_fuzzy_accuracy_config = MetricConfig(
    name="fuzzy_accuracy",
    higher_is_better=True,
    aggregation_name="fuzzy_accuracy",
    process_results=process_results_fuzzy,
)
get_metrics_registry().register("fuzzy_accuracy", AccuracyMetric(_fuzzy_accuracy_config))
