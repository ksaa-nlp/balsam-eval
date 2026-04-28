"""Blank metric template for balsam-eval.

Fully wired into lmms_eval + metrics_registry.
"""

from lmms_eval.api import registry as le_registry
from lmms_eval.api.registry import register_aggregation, register_metric

from src.metrics_registry import BaseMetric, MetricConfig, get_metrics_registry


# Register aggregation function
if "new_metric_agg" not in le_registry.AGGREGATION_REGISTRY:
    @register_aggregation("new_metric_agg")
    def new_metric_aggregation(items):
        """Aggregate multiple results into one final score.

        Args:
            items: List of metric items

        Returns:
            Aggregated score
        """
        # TODO: Implement aggregation logic
        return 0.0


# Register metric function
if "new_metric" not in le_registry.METRIC_REGISTRY:
    @register_metric(
        metric="new_metric",
        higher_is_better=True,
        output_type="generate_until",
        aggregation="new_metric_agg",
    )
    def new_metric(items):
        """Return data items for aggregation.

        Args:
            items: Metric items

        Returns:
            Items for aggregation
        """
        # TODO: Prepare items for aggregation
        return items


def process_results(doc, results):
    """Extract reference & prediction text from sample.

    Args:
        doc: Document with reference
        results: Model predictions

    Returns:
        Dictionary with metric data
    """
    # TODO: Implement extraction logic
    return {"new_metric": ["", ""]}


class NewMetric(BaseMetric):
    """BaseMetric subclass for YAML/task export."""

    def get_doc_to_text(self, original_doc_to_text: str) -> str:
        """Get doc_to_text template.

        Args:
            original_doc_to_text: Original template

        Returns:
            Original template (no modification)
        """
        # TODO: Optionally modify doc prompt
        return original_doc_to_text

    def get_generation_kwargs(self) -> dict:
        """Get generation kwargs.

        Returns:
            Generation parameters
        """
        # TODO: Adjust generation parameters
        return {}


# Register in custom registry
config = MetricConfig(
    name="new_metric",
    aggregation_name="new_metric_agg",
    higher_is_better=True,
    process_results=process_results,
)
get_metrics_registry().register("new_metric", NewMetric(config))
