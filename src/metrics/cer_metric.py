"""CER (Character Error Rate) metric implementation for ASR evaluation."""

import json
from typing import Any, Dict, List

from lmms_eval.api import registry as le_registry
from lmms_eval.api.registry import register_aggregation, register_metric

from src.metrics_registry import BaseMetric, MetricConfig, get_metrics_registry

try:
    import jiwer
except ImportError:
    jiwer = None


def compute_cer_score(
    references: List[str],
    predictions: List[str],
) -> float:
    """Compute CER score for references and predictions.

    Args:
        references: List of reference texts
        predictions: List of prediction texts

    Returns:
        Average CER score (lower is better)
    """
    if jiwer is None:
        return 1.0

    total_cer = 0.0
    valid = 0

    for ref, pred in zip(references, predictions):
        if not pred or not ref:
            continue

        try:
            cer = jiwer.cer(ref, pred)
            total_cer += cer
            valid += 1
        except Exception:
            continue

    return total_cer / valid if valid else 1.0


def compute_cer_aggregation(items: List[Any]) -> float:
    """Aggregate CER scores from reference and prediction pairs.

    Args:
        items: List of [reference, prediction] lists

    Returns:
        Average CER score
    """
    refs = [r[0] if isinstance(r, (list, tuple)) else r for r in items]
    preds = [p[1] if isinstance(p, (list, tuple)) else p for p in items]

    return compute_cer_score(references=refs, predictions=preds)


# Register aggregation function
if "cer_aggregation" not in le_registry.AGGREGATION_REGISTRY:
    register_aggregation("cer_aggregation")(compute_cer_aggregation)


# Register metric function
if "cer" not in le_registry.METRIC_REGISTRY:
    register_metric(
        metric="cer",
        higher_is_better=False,  # Lower CER is better
        output_type="generate_until",
        aggregation="cer_aggregation",
    )(lambda items: items)


def extract_text_from_prediction(pred: str) -> str:
    """Extract actual text content from various prediction formats.

    Handles:
    - JSON format: {"text": "..."}
    - Quoted strings: "..."
    - Plain text

    Args:
        pred: Raw prediction string

    Returns:
        Extracted text content
    """
    if not isinstance(pred, str):
        return str(pred) if pred else ""

    pred = pred.strip()

    # Try to parse as JSON first
    try:
        parsed = json.loads(pred)
        if isinstance(parsed, dict):
            # Handle {"text": "..."} format
            if "text" in parsed:
                return parsed["text"]
            # Handle other dict formats - return first string value
            for v in parsed.values():
                if isinstance(v, str):
                    return v
        elif isinstance(parsed, str):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Remove surrounding quotes if present
    if pred.startswith('"') and pred.endswith('"'):
        pred = pred[1:-1]
    elif pred.startswith("'") and pred.endswith("'"):
        pred = pred[1:-1]

    return pred


def process_results(doc: Dict[str, Any], results: Any) -> Dict[str, List[str]]:
    """Process results for CER evaluation.

    Extracts reference and prediction from document and model results.

    Args:
        doc: Document containing reference output
        results: Model predictions (list or single value)

    Returns:
        Dictionary with CER data containing [reference, prediction]
    """
    preds = results[0] if isinstance(results, list) else results
    golds = doc["output"]

    # Extract actual text from prediction
    cleaned_pred = extract_text_from_prediction(preds)

    return {"cer": [golds, cleaned_pred]}


class CERMetric(BaseMetric):
    """CER metric for YAML/task export.

    This metric class integrates with the metrics registry to provide
    CER evaluation for ASR (Automatic Speech Recognition) tasks.
    """

    def get_doc_to_text(self, original_doc_to_text: str) -> str:
        """Get the doc_to_text template for CER metric.

        Args:
            original_doc_to_text: Original doc_to_text template

        Returns:
            Original template (CER doesn't modify the prompt)
        """
        return original_doc_to_text

    def get_generation_kwargs(self) -> Dict[str, Any]:
        """Get generation kwargs for CER metric.

        Returns:
            Generation parameters (no sampling, stop on empty string)
        """
        return {"do_sample": False, "until": [""]}


# Register in custom registry
_cer_config = MetricConfig(
    name="cer",
    higher_is_better=False,  # Lower CER is better
    aggregation_name="cer_aggregation",
    process_results=process_results,
)
get_metrics_registry().register("cer", CERMetric(_cer_config))
