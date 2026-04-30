"""LLM-as-Judge metric implementation for integration with lmms_eval.

This metric allows LLM judge to be called like other metrics during evaluation,
rather than as a post-processing step.

Supports both MCQ and Generative tasks with proper scoring.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import lmms_eval.api.registry as le_registry
from lmms_eval.api.registry import register_aggregation, register_metric

from src.llm_judger.base_llm_judge import ModelConfig
from src.llm_judger.generative_llm_judge import GenerativeLLMJudge
from src.llm_judger.mcq_llm_judge import MCQLLMJudge
from src.metrics_registry import BaseMetric, MetricConfig, get_metrics_registry

logger = logging.getLogger(__name__)

# Global cache for LLM judge instances to avoid reinitializing
_llm_judge_cache: Dict[str, Any] = {}


def get_llm_judge_config() -> Dict[str, Optional[str]]:
    """Get LLM judge configuration from environment variables.

    Returns:
        Dictionary with model_name, provider, and api_key
    """
    return {
        "model_name": os.environ.get("LLM_JUDGE_MODEL"),
        "provider": os.environ.get("LLM_JUDGE_PROVIDER"),
        "api_key": os.environ.get("LLM_JUDGE_API_KEY"),
    }


def get_or_create_mcq_judge() -> MCQLLMJudge:
    """Get or create MCQ LLM judge instance from cache.

    Returns:
        MCQLLMJudge instance
    """
    cache_key = "mcq"

    if cache_key not in _llm_judge_cache:
        config = get_llm_judge_config()

        if not all([config["model_name"], config["provider"], config["api_key"]]):
            raise ValueError(
                "LLM judge not configured. Please set LLM_JUDGE_MODEL, "
                "LLM_JUDGE_PROVIDER, and LLM_JUDGE_API_KEY environment variables."
            )

        logger.info(
            "Initializing MCQ LLM Judge with model=%s, provider=%s",
            config["model_name"],
            config["provider"],
        )

        _llm_judge_cache[cache_key] = MCQLLMJudge(
            model_configs=[
                ModelConfig(
                    name=config["model_name"],
                    provider=config["provider"],  # type: ignore[arg-type]
                    api_key=config["api_key"],
                )
            ],
            custom_prompt=None,
            aggregation_method="mean",
            threshold=0.5,
        )

    return _llm_judge_cache[cache_key]


def get_or_create_generative_judge() -> GenerativeLLMJudge:
    """Get or create Generative LLM judge instance from cache.

    Returns:
        GenerativeLLMJudge instance
    """
    cache_key = "generative"

    if cache_key not in _llm_judge_cache:
        config = get_llm_judge_config()

        if not all([config["model_name"], config["provider"], config["api_key"]]):
            raise ValueError(
                "LLM judge not configured. Please set LLM_JUDGE_MODEL, "
                "LLM_JUDGE_PROVIDER, and LLM_JUDGE_API_KEY environment variables."
            )

        logger.info(
            "Initializing Generative LLM Judge with model=%s, provider=%s",
            config["model_name"],
            config["provider"],
        )

        _llm_judge_cache[cache_key] = GenerativeLLMJudge(
            model_configs=[
                ModelConfig(
                    name=config["model_name"],
                    provider=config["provider"],  # type: ignore[arg-type]
                    api_key=config["api_key"],
                )
            ],
            custom_prompt=None,
            aggregation_method="mean",
            threshold=0.5,
        )

    return _llm_judge_cache[cache_key]


def compute_mcq_llm_judge(items: List[List[Any]]) -> float:
    """Compute MCQ LLM judge score.

    Args:
        items: List of [reference, prediction] pairs

    Returns:
        Average LLM judge score (0-1)
    """
    if not items:
        return 0.0

    try:
        judge = get_or_create_mcq_judge()
    except ValueError as e:
        logger.warning("MCQ LLM Judge not available: %s", e)
        return 0.0

    scores = []
    for ref, pred in items:
        if not ref or not pred:
            continue

        try:
            result = judge.evaluate_answer(
                question="",  # MCQ doesn't need the question context
                reference_answer=str(ref),
                given_answer=str(pred),
                context=None,
            )
            scores.append(result["overall_score"])
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Error evaluating with MCQ LLM judge: %s", e)
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0


def compute_generative_llm_judge(items: List[List[Any]]) -> float:
    """Compute Generative LLM judge score.

    Args:
        items: List of [reference, prediction] pairs

    Returns:
        Average LLM judge score (0-1)
    """
    if not items:
        return 0.0

    try:
        judge = get_or_create_generative_judge()
    except ValueError as e:
        logger.warning("Generative LLM Judge not available: %s", e)
        return 0.0

    scores = []
    for ref, pred in items:
        if not ref or not pred:
            continue

        try:
            result = judge.evaluate_answer(
                question="",  # Question not needed if already in context
                reference_answer=str(ref),
                given_answer=str(pred),
                context=None,
            )
            scores.append(result["overall_score"])
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Error evaluating with Generative LLM judge: %s", e)
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0


# Register aggregation functions
if "mcq_llm_judge" not in le_registry.AGGREGATION_REGISTRY:
    register_aggregation("mcq_llm_judge")(compute_mcq_llm_judge)

if "generative_llm_judge" not in le_registry.AGGREGATION_REGISTRY:
    register_aggregation("generative_llm_judge")(compute_generative_llm_judge)


# Register metrics
if "mcq_llm_judge" not in le_registry.METRIC_REGISTRY:
    register_metric(
        metric="mcq_llm_judge",
        higher_is_better=True,
        output_type="generate_until",
        aggregation="mcq_llm_judge",
    )(lambda items: items)

if "generative_llm_judge" not in le_registry.METRIC_REGISTRY:
    register_metric(
        metric="generative_llm_judge",
        higher_is_better=True,
        output_type="generate_until",
        aggregation="generative_llm_judge",
    )(lambda items: items)


def process_results_mcq(doc: Dict[str, Any], results: Any) -> Dict[str, List[str]]:
    """Process results for MCQ LLM judge evaluation.

    Args:
        doc: Document containing reference output
        results: Model predictions (list or single value)

    Returns:
        Dictionary with mcq_llm_judge data containing [reference, prediction]
    """
    preds = results[0] if isinstance(results, list) else results
    golds = doc["output"]

    return {"mcq_llm_judge": [golds, preds]}


def process_results_generative(
    doc: Dict[str, Any], results: Any
) -> Dict[str, List[str]]:
    """Process results for Generative LLM judge evaluation.

    Args:
        doc: Document containing reference output
        results: Model predictions (list or single value)

    Returns:
        Dictionary with generative_llm_judge data containing [reference, prediction]
    """
    preds = results[0] if isinstance(results, list) else results
    golds = doc["output"]

    return {"generative_llm_judge": [golds, preds]}


class MCQLLMJudgeMetric(BaseMetric):
    """MCQ LLM Judge metric for YAML/task export."""

    def get_doc_to_text(self, original_doc_to_text: str) -> str:
        """Get the doc_to_text template for MCQ LLM judge metric.

        Args:
            original_doc_to_text: Original doc_to_text template

        Returns:
            Original template (LLM judge doesn't modify the prompt)
        """
        return original_doc_to_text

    def get_generation_kwargs(self) -> Dict[str, Any]:
        """Get generation kwargs for MCQ LLM judge metric.

        Returns:
            Generation parameters
        """
        return {"do_sample": False, "until": [""]}


class GenerativeLLMJudgeMetric(BaseMetric):
    """Generative LLM Judge metric for YAML/task export."""

    def get_doc_to_text(self, original_doc_to_text: str) -> str:
        """Get the doc_to_text template for Generative LLM judge metric.

        Args:
            original_doc_to_text: Original doc_to_text template

        Returns:
            Original template (LLM judge doesn't modify the prompt)
        """
        return original_doc_to_text

    def get_generation_kwargs(self) -> Dict[str, Any]:
        """Get generation kwargs for Generative LLM judge metric.

        Returns:
            Generation parameters
        """
        return {"do_sample": True, "temperature": 0.7, "until": ["<|endoftext|>"]}


# Register in custom registry
_mcq_judge_config = MetricConfig(
    name="mcq_llm_judge",
    higher_is_better=True,
    aggregation_name="mcq_llm_judge",
    process_results=process_results_mcq,
)
get_metrics_registry().register("mcq_llm_judge", MCQLLMJudgeMetric(_mcq_judge_config))

_generative_judge_config = MetricConfig(
    name="generative_llm_judge",
    higher_is_better=True,
    aggregation_name="generative_llm_judge",
    process_results=process_results_generative,
)
get_metrics_registry().register(
    "generative_llm_judge", GenerativeLLMJudgeMetric(_generative_judge_config)
)
