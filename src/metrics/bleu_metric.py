"""BLEU metric implementation for text evaluation."""

import re
import sys
import unicodedata

import evaluate
import pyarabic.araby as araby
from lmms_eval.api import registry as le_registry
from lmms_eval.api.registry import register_aggregation, register_metric

from src.metrics_registry import BaseMetric, MetricConfig, get_metrics_registry

bleu = evaluate.load("bleu")

# Punctuation handling
PUNCT_TABLE = dict.fromkeys(
    i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")
)
ALL_PUNCTUATIONS = "".join(chr(p) for p in PUNCT_TABLE)
OTHERS = """`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!"…"–ـ"""
ALL_PUNCTUATIONS += "".join([o for o in OTHERS if o not in ALL_PUNCTUATIONS])


def prepare_texts(text: str, change_curly_braces: bool, remove_diacritics: bool) -> str:
    """Prepare text for BLEU evaluation.

    Args:
        text: Input text
        change_curly_braces: Whether to change curly braces
        remove_diacritics: Whether to remove Arabic diacritics

    Returns:
        Prepared text
    """
    text = re.sub("([" + ALL_PUNCTUATIONS + "])", " \\1 ", text)
    if change_curly_braces:
        text = text.replace("{", "[").replace("}", "]")
    if remove_diacritics:
        text = araby.strip_diacritics(text)
    return text


# Register aggregation
if "custom_bleu" not in le_registry.AGGREGATION_REGISTRY:
    @register_aggregation("custom_bleu")
    def bleu_aggregation(items):
        """Aggregate BLEU scores.

        Args:
            items: List of (reference, prediction) tuples

        Returns:
            Average BLEU score
        """
        def tokenizer(x):
            return x.split()

        refs = [r[0] if isinstance(r, (list, tuple)) else r for r, _ in items]
        preds = [p for _, p in items]

        refs = [prepare_texts(r, False, True).strip() for r in refs]
        preds = [prepare_texts(p, True, True).strip() for p in preds]

        total, valid = 0.0, 0
        for ref, pred in zip(refs, preds):
            if not pred or not ref:
                continue
            try:
                score = bleu.compute(references=[ref], predictions=[pred], tokenizer=tokenizer)
                total += score["bleu"]
                valid += 1
            except ZeroDivisionError:
                continue
        return total / valid if valid else 0.0


# Register metric
if "bleu" not in le_registry.METRIC_REGISTRY:
    @register_metric(
        metric="bleu",
        higher_is_better=True,
        output_type="generate_until",
        aggregation="custom_bleu",
    )
    def bleu_metric(items):
        """Return BLEU metric items."""
        return items


def process_results(doc, results):
    """Process results for BLEU evaluation.

    Args:
        doc: Document containing reference
        results: Model predictions

    Returns:
        Dictionary with BLEU data
    """
    preds, golds = results[0], doc["output"]
    return {"bleu": [golds, preds]}


class BleuMetric(BaseMetric):
    """BLEU metric for YAML/task export."""

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
    name="bleu",
    higher_is_better=True,
    aggregation_name="custom_bleu",
    process_results=process_results,
)
get_metrics_registry().register("bleu", BleuMetric(config))
