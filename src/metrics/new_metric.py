"""
Blank metric template for balsam-eval.
Fully wired into lm_eval + metrics_registry.
"""

from lm_eval.api.registry import register_metric, register_aggregation
from lm_eval.api import registry as le_registry
from src.metrics_registry import BaseMetric, MetricConfig, get_metrics_registry


# ----------------------------
# ⚙️ Aggregation function
# ----------------------------
if "new_metric_agg" not in le_registry.AGGREGATION_REGISTRY:
    @register_aggregation("new_metric_agg")
    def new_metric_aggregation(items):
        """Aggregate multiple results into one final score."""
        # TODO: implement aggregation logic
        return 0.0


# ----------------------------
# 🧮 Metric function
# ----------------------------
if "new_metric" not in le_registry.METRIC_REGISTRY:
    @register_metric(
        metric="new_metric",
        higher_is_better=True,
        output_type="generate_until",
        aggregation="new_metric_agg",
    )
    def new_metric(items):
        """Return data items for aggregation."""
        # TODO: prepare items for aggregation
        return items


# ----------------------------
# 🧾 Process results
# ----------------------------
def process_results(doc, results):
    """Extract reference & prediction text from sample."""
    # TODO: implement extraction logic
    return {"new_metric": ["", ""]}


# ----------------------------
# 🧩 YAML Export Integration
# ----------------------------
class NewMetric(BaseMetric):
    """BaseMetric subclass for YAML/task export."""

    def get_doc_to_text(self, original_doc_to_text: str) -> str:
        # TODO: optionally modify doc prompt
        return original_doc_to_text

    def get_generation_kwargs(self):
        # TODO: adjust generation parameters
        return {}


# ----------------------------
# 🔗 Register in custom registry
# ----------------------------
config = MetricConfig(
    name="new_metric",
    aggregation_name="new_metric_agg",
    higher_is_better=True,
    process_results=process_results,
)
get_metrics_registry().register("new_metric", NewMetric(config))
