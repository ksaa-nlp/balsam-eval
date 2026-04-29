"""Template metric for implementing custom evaluation metrics.

This file serves as a template for creating new metrics. Copy this file
and customize the TODO sections to implement your custom metric logic.

The metric is fully wired into lmms_eval and metrics_registry systems.
"""

from typing import Any, Dict, List

from lmms_eval.api import registry as le_registry
from lmms_eval.api.registry import register_aggregation, register_metric

from src.metrics_registry import BaseMetric, MetricConfig, get_metrics_registry


def compute_new_metric_aggregation(items: List[Any]) -> float:  # pylint: disable=unused-argument
    """Aggregate metric results into a final score.

    TODO: Implement your aggregation logic here.
    This function receives a list of metric items and should return
    a single score value.

    Args:
        _items: List of metric items (typically reference, prediction pairs)

    Returns:
        Aggregated score (typically a float between 0 and 1, or 0 to 100)

    Example:
        >>> def compute_new_metric_aggregation(items):
        ...     total = 0
        ...     count = 0
        ...     for ref, pred in items:
        ...         if ref is not None and pred is not None:
        ...             # Your scoring logic here
        ...             score = float(ref == pred)  # Simple exact match
        ...             total += score
        ...             count += 1
        ...     return total / count if count > 0 else 0.0
    """
    # TODO: Implement aggregation logic
    return 0.0


# Register aggregation function
if "new_metric_agg" not in le_registry.AGGREGATION_REGISTRY:
    register_aggregation("new_metric_agg")(compute_new_metric_aggregation)


# Register metric function
if "new_metric" not in le_registry.METRIC_REGISTRY:
    register_metric(
        metric="new_metric",
        higher_is_better=True,
        output_type="generate_until",
        aggregation="new_metric_agg",
    )(lambda items: items)


def process_results(doc: Dict[str, Any], results: Any) -> Dict[str, List[Any]]:
    """Process results for metric evaluation.

    TODO: Implement extraction logic to prepare data for aggregation.
    Extract the reference and prediction from the document and results.

    Args:
        doc: Document containing reference output and other metadata
        results: Model predictions (list or single value)

    Returns:
        Dictionary with metric data containing [reference, prediction]

    Example:
        >>> def process_results(doc, results):
        ...     preds = results[0] if isinstance(results, list) else results
        ...     golds = doc["output"]
        ...     return {"new_metric": [golds, preds]}
    """
    # TODO: Implement extraction logic
    preds = results[0] if isinstance(results, list) else results
    golds = doc.get("output", "")
    return {"new_metric": [golds, preds]}


class NewMetric(BaseMetric):
    """Template metric class for YAML/task export.

    TODO: Customize this class if your metric needs to modify prompts
    or generation parameters.

    This metric class integrates with the metrics registry to provide
    custom evaluation metrics for tasks.
    """

    def get_doc_to_text(self, original_doc_to_text: str) -> str:
        """Get the doc_to_text template for this metric.

        TODO: Modify the prompt if your metric requires specific instructions.
        Most metrics can return the original template unchanged.

        Args:
            original_doc_to_text: Original doc_to_text template

        Returns:
            Modified or original template

        Example:
            >>> def get_doc_to_text(self, original_doc_to_text: str) -> str:
            ...     # Add special instructions for your metric
            ...     return f"{original_doc_to_text}\\nPlease respond in JSON format."
        """
        # TODO: Optionally modify doc prompt
        return original_doc_to_text

    def get_generation_kwargs(self) -> Dict[str, Any]:
        """Get generation kwargs for this metric.

        TODO: Adjust generation parameters if needed.
        Most metrics use default parameters (no sampling, stop on empty).

        Returns:
            Generation parameters

        Example:
            >>> def get_generation_kwargs(self) -> Dict[str, Any]:
            ...     return {
            ...         "do_sample": False,
            ...         "until": ["\\n"],
            ...         "temperature": 0.7,
            ...         "max_new_tokens": 100,
            ...     }
        """
        # TODO: Adjust generation parameters
        return {"do_sample": False, "until": [""]}


# Register in custom registry
_new_metric_config = MetricConfig(
    name="new_metric",
    higher_is_better=True,
    aggregation_name="new_metric_agg",
    process_results=process_results,
)
get_metrics_registry().register("new_metric", NewMetric(_new_metric_config))
