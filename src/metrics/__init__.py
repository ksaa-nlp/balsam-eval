"""Metrics package for evaluation metrics."""

import importlib
import os
from pathlib import Path

# Auto-discover and import all metric implementations
_impl_dir = Path(__file__).parent / "impl"
if _impl_dir.exists():
    for _metric_file in _impl_dir.glob("*.py"):
        if _metric_file.name != "__init__.py":
            _module_name = f"src.metrics.impl.{_metric_file.stem}"
            try:
                importlib.import_module(_module_name)
            except ImportError:
                pass  # Skip if there are import errors

# Explicit imports for backward compatibility
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
