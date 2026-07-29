"""WER (Word Error Rate) metric implementation for ASR evaluation."""

from typing import Any, Dict, List
import jiwer

from lm_eval.api import registry as le_registry
from lm_eval.api.registry import register_aggregation, register_metric

from src.metrics_registry import BaseMetric, MetricConfig, get_metrics_registry
from src.metrics.asr_utils import process_results_asr




def compute_wer_score(
    references: List[str],
    predictions: List[str],
) -> float:
    """Compute WER score for references and predictions.

    Args:
        references: List of reference texts
        predictions: List of prediction texts

    Returns:
        Average WER score (lower is better)
    """

    if len(references) != len(predictions):
        raise ValueError("references and predictions must have the same length")

    total_wer = 0.0
    valid = 0

    for ref, pred in zip(references, predictions):
        if not ref:
            continue
        total_wer += jiwer.wer(ref, pred or "")
        valid += 1

    return total_wer / valid if valid else 1.0


def compute_wer_aggregation(items: List[Any]) -> float:
    """Aggregate WER scores from reference and prediction pairs.

    Args:
        items: List of [reference, prediction] lists

    Returns:
        Average WER score
    """
    if not all(isinstance(item, (list, tuple)) and len(item) == 2 for item in items):
        raise ValueError("WER aggregation items must be [reference, prediction] pairs")
    refs = [item[0] for item in items]
    preds = [item[1] for item in items]

    return compute_wer_score(references=refs, predictions=preds)


# Register aggregation function
if "wer_aggregation" not in le_registry.AGGREGATION_REGISTRY:
    register_aggregation("wer_aggregation")(compute_wer_aggregation)


# Register metric function
if "wer" not in le_registry.METRIC_REGISTRY:
    register_metric(
        metric="wer",
        higher_is_better=False,  # Lower WER is better
        output_type="generate_until",
        aggregation="wer_aggregation",
    )(lambda items: items)


def process_results(doc: Dict[str, Any], results: Any) -> Dict[str, List[str]]:
    """Process results for WER evaluation.

    Extracts reference and prediction from document and model results.

    Args:
        doc: Document containing reference output
        results: Model predictions (list or single value)

    Returns:
        Dictionary with WER data containing [reference, prediction]
    """
    return process_results_asr(doc, results, "wer")


class WERMetric(BaseMetric):
    """WER metric for YAML/task export.

    This metric class integrates with the metrics registry to provide
    WER evaluation for ASR (Automatic Speech Recognition) tasks.
    """

    def get_doc_to_text(self, original_doc_to_text: str) -> str:
        """Get the doc_to_text template for WER metric.

        Args:
            original_doc_to_text: Original doc_to_text template

        Returns:
            Original template (WER doesn't modify the prompt)
        """
        return original_doc_to_text

    def get_generation_kwargs(self) -> Dict[str, Any]:
        """Get generation kwargs for WER metric.

        Returns:
            Generation parameters (no sampling, stop on empty string)
        """
        return {"do_sample": False, "until": []}


# Register in custom registry
_wer_config = MetricConfig(
    name="wer",
    higher_is_better=False,  # Lower WER is better
    aggregation_name="wer_aggregation",
    process_results=process_results,
)
get_metrics_registry().register("wer", WERMetric(_wer_config))
