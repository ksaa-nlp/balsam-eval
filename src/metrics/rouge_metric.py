"""ROUGE metric implementation for text evaluation."""

import re
import sys
import unicodedata

import evaluate
from lmms_eval.api import registry as le_registry
from lmms_eval.api.registry import register_aggregation, register_metric

from src.metrics_registry import BaseMetric, MetricConfig, get_metrics_registry

rouge = evaluate.load("rouge")

# Punctuation handling
PUNCT_TABLE = dict.fromkeys(
    i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")
)
ALL_PUNCTUATIONS = "".join(chr(p) for p in PUNCT_TABLE)
OTHERS = """`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!"…"–ـ"""
ALL_PUNCTUATIONS += "".join([o for o in OTHERS if o not in ALL_PUNCTUATIONS])


def prepare_texts(text: str, change_curly_braces: bool = False) -> str:
    """Prepare text for ROUGE evaluation.

    Args:
        text: Input text
        change_curly_braces: Whether to change curly braces

    Returns:
        Prepared text
    """
    text = re.sub("([" + ALL_PUNCTUATIONS + "])", " \\1 ", text)
    if change_curly_braces:
        text = text.replace("{", "[").replace("}", "]")
    return text


# Register aggregation
if "rouge" not in le_registry.AGGREGATION_REGISTRY:
    @register_aggregation("rouge")
    def rouge_aggregation(items):
        """Aggregate ROUGE scores.

        Args:
            items: List of (reference, prediction) tuples

        Returns:
            Dictionary with ROUGE scores
        """
        def tokenizer(x):
            return x.split()

        refs = [prepare_texts(r, False).strip() for r, _ in items]
        preds = [prepare_texts(p, True).strip() for _, p in items]

        total = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}
        for ref, pred in zip(refs, preds):
            score = rouge.compute(references=[ref], predictions=[pred], tokenizer=tokenizer)
            for k in total.keys():
                total[k] += score[k]
        count = len(refs) if refs else 1
        return {k: v / count for k, v in total.items()}


# Register metric
if "rouge" not in le_registry.METRIC_REGISTRY:
    @register_metric(
        metric="rouge",
        higher_is_better=True,
        output_type="generate_until",
        aggregation="rouge",
    )
    def rouge_fn(items):
        """Return ROUGE metric items."""
        return items


def process_results(doc, results):
    """Process results for ROUGE evaluation.

    Args:
        doc: Document containing reference
        results: Model predictions

    Returns:
        Dictionary with ROUGE data
    """
    preds = results[0] if isinstance(results, list) else results
    golds = doc["output"]
    return {"rouge": [golds, preds]}


class RougeMetric(BaseMetric):
    """ROUGE metric for YAML/task export."""

    def get_doc_to_text(self, original_doc_to_text: str) -> str:
        """Get doc_to_text template.

        Args:
            original_doc_to_text: Original template

        Returns:
            Original template (no modification)
        """
        return original_doc_to_text

    def get_generation_kwargs(self) -> dict:
        """Get generation kwargs.

        Returns:
            Generation parameters
        """
        return {"do_sample": False, "until": [""]}


# Register in custom registry
config = MetricConfig(
    name="rouge",
    higher_is_better=True,
    aggregation_name="rouge",
    process_results=process_results,
)
get_metrics_registry().register("rouge", RougeMetric(config))
