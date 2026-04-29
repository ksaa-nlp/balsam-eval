"""ROUGE metric implementation for text evaluation."""

from typing import Any, Dict, List, Tuple

from lmms_eval.api import registry as le_registry
from lmms_eval.api.registry import register_aggregation, register_metric

from src.metrics_registry import BaseMetric, MetricConfig, get_metrics_registry
from src.metrics.metrics_utils import prepare_text_with_punctuation


def compute_rouge_scores(
    references: List[str],
    predictions: List[str],
    prepare_refs: bool = False,
    prepare_preds: bool = True,
) -> Dict[str, float]:
    """Compute ROUGE scores for references and predictions.

    Args:
        references: List of reference texts
        predictions: List of prediction texts
        prepare_refs: Whether to prepare reference texts
        prepare_preds: Whether to prepare prediction texts

    Returns:
        Dictionary with rouge1, rouge2, rougeL, rougeLsum scores
    """
    try:
        import evaluate

        rouge = evaluate.load("rouge")
    except ImportError:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

    def tokenizer(x):
        return x.split()

    refs = [
        prepare_text_with_punctuation(r, prepare_refs, False).strip()
        for r in references
    ]
    preds = [
        prepare_text_with_punctuation(p, prepare_preds, False).strip()
        for p in predictions
    ]

    total = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}
    for ref, pred in zip(refs, preds):
        score = rouge.compute(
            references=[ref], predictions=[pred], tokenizer=tokenizer
        )
        for k in total:
            total[k] += score[k]
    count = len(refs) if refs else 1
    return {k: v / count for k, v in total.items()}


def compute_rouge_aggregation(items: List[Tuple[Any, Any]]) -> Dict[str, float]:
    """Aggregate ROUGE scores from reference and prediction pairs.

    Args:
        items: List of (reference, prediction) tuples

    Returns:
        Dictionary with rouge1, rouge2, rougeL, rougeLsum scores
    """
    refs = [r for r, _ in items]
    preds = [p for _, p in items]

    return compute_rouge_scores(
        references=refs, predictions=preds, prepare_refs=False, prepare_preds=True
    )


# Register aggregation function
if "rouge" not in le_registry.AGGREGATION_REGISTRY:
    register_aggregation("rouge")(compute_rouge_aggregation)


# Register metric function
if "rouge" not in le_registry.METRIC_REGISTRY:
    register_metric(
        metric="rouge",
        higher_is_better=True,
        output_type="generate_until",
        aggregation="rouge",
    )(lambda items: items)


def process_results(doc: Dict[str, Any], results: Any) -> Dict[str, List[str]]:
    """Process results for ROUGE evaluation.

    Extracts reference and prediction from document and model results.

    Args:
        doc: Document containing reference output
        results: Model predictions (list or single value)

    Returns:
        Dictionary with ROUGE data containing [reference, prediction]
    """
    preds = results[0] if isinstance(results, list) else results
    golds = doc["output"]
    return {"rouge": [golds, preds]}


class RougeMetric(BaseMetric):
    """ROUGE metric for YAML/task export.

    This metric class integrates with the metrics registry to provide
    ROUGE evaluation for text generation tasks.
    """

    def get_doc_to_text(self, original_doc_to_text: str) -> str:
        """Get the doc_to_text template for ROUGE metric.

        Args:
            original_doc_to_text: Original doc_to_text template

        Returns:
            Original template (ROUGE doesn't modify the prompt)
        """
        return original_doc_to_text

    def get_generation_kwargs(self) -> Dict[str, Any]:
        """Get generation kwargs for ROUGE metric.

        Returns:
            Generation parameters (no sampling, stop on empty string)
        """
        return {"do_sample": False, "until": [""]}


# Register in custom registry
_rouge_config = MetricConfig(
    name="rouge",
    higher_is_better=True,
    aggregation_name="rouge",
    process_results=process_results,
)
get_metrics_registry().register("rouge", RougeMetric(_rouge_config))
