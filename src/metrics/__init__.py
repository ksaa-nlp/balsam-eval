"""Metrics package initialization."""

# Import all metrics to register them
from . import rouge_metric
from . import bleu_metric
from . import accuracy_metric

__all__ = ['rouge_metric', 'bleu_metric', 'accuracy_metric']
