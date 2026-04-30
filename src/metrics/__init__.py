"""Metrics package for evaluation metrics.

This package provides implementations of various evaluation metrics
including accuracy, BLEU, ROUGE, WER, CER, LLM-as-Judge, and extensible templates for custom metrics.

All metrics are automatically registered with both lmms_eval and the
custom metrics_registry system.

Usage:
    >>> from src.metrics import AccuracyMetric, process_results as accuracy_process_results
    >>> from src.metrics import BleuMetric, process_results as bleu_process_results
    >>> from src.metrics import RougeMetric, process_results as rouge_process_results
    >>> from src.metrics import WERMetric, process_results as wer_process_results
    >>> from src.metrics import CERMetric, process_results as cer_process_results
    >>> from src.metrics import MCQLLMJudgeMetric, process_results_mcq_llm_judge
    >>> from src.metrics import GenerativeLLMJudgeMetric, process_results_generative_llm_judge
    >>> from src.metrics.metrics_utils import prepare_text_with_punctuation, all_punctuations
"""

# Metric implementations
from .accuracy_metric import (
    AccuracyMetric,
    process_results as accuracy_process_results,
)
from .bleu_metric import BleuMetric, process_results as bleu_process_results
from .cer_metric import CERMetric, process_results as cer_process_results
from .llm_judge_metric import (
    GenerativeLLMJudgeMetric,
    MCQLLMJudgeMetric,
    process_results_generative as process_results_generative_llm_judge,
    process_results_mcq as process_results_mcq_llm_judge,
)
from .new_metric import NewMetric, process_results as new_metric_process_results
from .rouge_metric import RougeMetric, process_results as rouge_process_results
from .wer_metric import WERMetric, process_results as wer_process_results

# Shared utilities (only truly shared code)
from .metrics_utils import (
    all_punctuations,
    prepare_text_with_punctuation,
)

__all__ = [
    # Metric classes
    "AccuracyMetric",
    "BleuMetric",
    "RougeMetric",
    "WERMetric",
    "CERMetric",
    "MCQLLMJudgeMetric",
    "GenerativeLLMJudgeMetric",
    # Process results functions
    "accuracy_process_results",
    "bleu_process_results",
    "rouge_process_results",
    "wer_process_results",
    "cer_process_results",
    "process_results_mcq_llm_judge",
    "process_results_generative_llm_judge",
    # Shared utilities (only truly shared)
    "prepare_text_with_punctuation",
    "all_punctuations",
]
