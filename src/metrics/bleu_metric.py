"""BLEU metric implementation for text evaluation."""

from typing import Any, Dict, List, Tuple

import evaluate

from lmms_eval.api import registry as le_registry
from lmms_eval.api.registry import register_aggregation, register_metric

from src.metrics_registry import BaseMetric, MetricConfig, get_metrics_registry
from src.metrics.metrics_utils import prepare_text_with_punctuation

def compute_bleu_score(
    references: List[str],
    predictions: List[str],
    prepare_refs: bool = False,
    prepare_preds: bool = True,
    remove_diacritics: bool = True,
) -> float:
    """Compute BLEU score for references and predictions.

    Args:
        references: List of reference texts
        predictions: List of prediction texts
        prepare_refs: Whether to prepare reference texts
        prepare_preds: Whether to prepare prediction texts
        remove_diacritics: Whether to remove Arabic diacritics

    Returns:
        Average BLEU score
    """
    bleu = evaluate.load("bleu")

    def tokenizer(x):
        return x.split()

    refs = [
        prepare_text_with_punctuation(r, prepare_refs, remove_diacritics).strip()
        for r in references
    ]
    preds = [
        prepare_text_with_punctuation(p, prepare_preds, remove_diacritics).strip()
        for p in predictions
    ]

    total = 0.0
    valid = 0
    for ref, pred in zip(refs, preds):
        if not pred or not ref:
            continue
        try:
            score = bleu.compute(
                references=[ref], predictions=[pred], tokenizer=tokenizer
            )
            total += score["bleu"]
            valid += 1
        except ZeroDivisionError:
            continue
    return total / valid if valid else 0.0


def compute_bleu_aggregation(items: List[Tuple[Any, Any]]) -> float:
    """Aggregate BLEU scores from reference and prediction pairs.

    Args:
        items: List of (reference, prediction) tuples

    Returns:
        Average BLEU score
    """
    refs = [r[0] if isinstance(r, (list, tuple)) else r for r, _ in items]
    preds = [p for _, p in items]

    return compute_bleu_score(
        references=refs,
        predictions=preds,
        prepare_refs=False,
        prepare_preds=True,
        remove_diacritics=True,
    )


# Register aggregation function
if "custom_bleu" not in le_registry.AGGREGATION_REGISTRY:
    register_aggregation("custom_bleu")(compute_bleu_aggregation)


# Register metric function
if "bleu" not in le_registry.METRIC_REGISTRY:
    register_metric(
        metric="bleu",
        higher_is_better=True,
        output_type="generate_until",
        aggregation="custom_bleu",
    )(lambda items: items)


def process_results(doc: Dict[str, Any], results: Any) -> Dict[str, List[str]]:
    """Process results for BLEU evaluation.

    Extracts reference and prediction from document and model results.

    Args:
        doc: Document containing reference output
        results: Model predictions (list or single value)

    Returns:
        Dictionary with BLEU data containing [reference, prediction]
    """
    preds = results[0] if isinstance(results, list) else results
    golds = doc["output"]
    return {"bleu": [golds, preds]}


class BleuMetric(BaseMetric):
    """BLEU metric for YAML/task export.

    This metric class integrates with the metrics registry to provide
    BLEU evaluation for text generation tasks.
    """

    def get_doc_to_text(self, original_doc_to_text: str) -> str:
        """Get the doc_to_text template for BLEU metric.

        Args:
            original_doc_to_text: Original doc_to_text template

        Returns:
            Original template (BLEU doesn't modify the prompt)
        """
        return original_doc_to_text

    def get_generation_kwargs(self) -> Dict[str, Any]:
        """Get generation kwargs for BLEU metric.

        Returns:
            Generation parameters (no sampling, stop on empty string)
        """
        return {"do_sample": False, "until": [""]}


# Register in custom registry
_bleu_config = MetricConfig(
    name="bleu",
    higher_is_better=True,
    aggregation_name="custom_bleu",
    process_results=process_results,
)
get_metrics_registry().register("bleu", BleuMetric(_bleu_config))
