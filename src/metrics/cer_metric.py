"""CER (Character Error Rate) metric implementation for ASR evaluation."""

from typing import Any, Dict, List
import jiwer

from lmms_eval.api import registry as le_registry
from lmms_eval.api.registry import register_aggregation, register_metric

from src.metrics_registry import BaseMetric, MetricConfig, get_metrics_registry
from src.metrics.asr_utils import process_results_asr



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
    total_cer = 0.0
    valid = 0

    for ref, pred in zip(references, predictions):
        if not pred or not ref:
            continue

        try:
            cer = jiwer.cer(ref, pred)
            total_cer += cer
            valid += 1
        except (TypeError, ValueError, RuntimeError):
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


def process_results(doc: Dict[str, Any], results: Any) -> Dict[str, List[str]]:
    """Process results for CER evaluation.

    Extracts reference and prediction from document and model results.

    Args:
        doc: Document containing reference output
        results: Model predictions (list or single value)

    Returns:
        Dictionary with CER data containing [reference, prediction]
    """
    return process_results_asr(doc, results, "cer")


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
