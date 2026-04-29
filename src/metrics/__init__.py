"""Metrics package for evaluation metrics.

This package provides implementations of various evaluation metrics
including accuracy, BLEU, ROUGE, and extensible templates for custom metrics.

All metrics are automatically registered with both lmms_eval and the
custom metrics_registry system.

Usage:
    >>> from src.metrics import AccuracyMetric, process_results as accuracy_process_results
    >>> from src.metrics import BleuMetric, process_results as bleu_process_results
    >>> from src.metrics import RougeMetric, process_results as rouge_process_results
    >>> from src.metrics.metrics_utils import prepare_text_with_punctuation, ALL_PUNCTUATIONS
"""

# Metric implementations
from .accuracy_metric import (
    AccuracyMetric,
    process_results as accuracy_process_results,
)
from .bleu_metric import BleuMetric, process_results as bleu_process_results
from .new_metric import NewMetric, process_results as new_metric_process_results
from .rouge_metric import RougeMetric, process_results as rouge_process_results

# Shared utilities (only truly shared code)
from .metrics_utils import (
    ALL_PUNCTUATIONS,
    prepare_text_with_punctuation,
)

__all__ = [
    # Metric classes
    "AccuracyMetric",
    "BleuMetric",
    "RougeMetric",
    # Process results functions
    "accuracy_process_results",
    "bleu_process_results",
    "rouge_process_results",
    # Shared utilities (only truly shared)
    "prepare_text_with_punctuation",
    "ALL_PUNCTUATIONS",
]
