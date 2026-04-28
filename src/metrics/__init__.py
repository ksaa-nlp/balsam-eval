"""Metrics package for evaluation metrics."""

from .impl.bleu_metric import BleuMetric, process_results as bleu_process_results
from .impl.rouge_metric import RougeMetric, process_results as rouge_process_results
from .impl.accuracy_metric import AccuracyMetric, process_results as accuracy_process_results
from .impl.new_metric import NewMetric, process_results as new_metric_process_results

__all__ = [
    "BleuMetric",
    "bleu_process_results",
    "RougeMetric",
    "rouge_process_results",
    "AccuracyMetric",
    "accuracy_process_results",
    "NewMetric",
    "new_metric_process_results",
]
