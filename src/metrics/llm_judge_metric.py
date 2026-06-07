"""LLM-as-judge metrics for evaluation (generative and MCQ).

Integrates the src/llm_judger system as proper lm-eval metrics.
- llm-as-judge:     uses GenerativeLLMJudge (0-3 scale, normalized to 0-1)
- mcq-llm-as-judge: uses MCQLLMJudge (binary 0/1, with MCQ answer normalization)

Requires env vars: JUDGE_MODEL, JUDGE_PROVIDER, JUDGE_API_KEY.
"""

import logging
import os
import re
from statistics import mean
from typing import Any, Dict, List, Tuple

from lm_eval.api import registry as le_registry
from lm_eval.api.registry import register_aggregation, register_metric
from tqdm import tqdm

from src.llm_judger.base_llm_judge import ModelConfig
from src.llm_judger.generative_llm_judge import GenerativeLLMJudge
from src.llm_judger.mcq_llm_judge import MCQLLMJudge
from src.metrics_registry import BaseMetric, MetricConfig, get_metrics_registry

logger = logging.getLogger(__name__)


def _parse_csv_env(name: str) -> list[str]:
    """Split a comma-separated env var into a trimmed list."""
    raw = os.getenv(name, "")
    return [v.strip() for v in raw.split(",") if v.strip()] if raw else []


def _get_judge_configs() -> list[ModelConfig]:
    """Build ModelConfig list from env vars.

    Supports comma-separated values for multiple judges.
    A single API key is broadcast to all judges.
    """
    models = _parse_csv_env("JUDGE_MODEL")
    providers = _parse_csv_env("JUDGE_PROVIDER")
    api_keys = _parse_csv_env("JUDGE_API_KEY")

    if not models or not providers:
        logger.warning(
            "LLM judge not configured — set JUDGE_MODEL, JUDGE_PROVIDER, "
            "JUDGE_API_KEY env vars."
        )
        return []

    n = max(len(models), len(providers))
    if len(models) == 1:
        models *= n
    if len(providers) == 1:
        providers *= n
    if len(api_keys) <= 1:
        api_keys = (api_keys or [""]) * n

    return [
        ModelConfig(
            name=models[i],
            provider=providers[i],  # type: ignore[arg-type]
            api_key=api_keys[i] or None,
        )
        for i in range(n)
    ]


def _create_generative_judge():
    configs = _get_judge_configs()
    if not configs:
        return None

    return GenerativeLLMJudge(
        model_configs=configs,
        aggregation_method="mean",
        threshold=0.5,
    )


def _create_mcq_judge():
    configs = _get_judge_configs()
    if not configs:
        return None

    return MCQLLMJudge(
        model_configs=configs,
        aggregation_method="mean",
        threshold=0.5,
    )


def _normalize_mcq_answer(answer: str, mcq_options: list) -> str:
    """Convert a single-letter answer (A/B/C/D) to its full text."""
    if not answer or not mcq_options:
        return answer or ""
    answer = str(answer).strip()
    mapping: Dict[str, str] = {chr(65 + i): str(opt) for i, opt in enumerate(mcq_options)}

    if len(answer) == 1 and answer.upper() in mapping:
        return mapping[answer.upper()]

    match = re.match(r"^([A-Za-z])\)", answer)
    if match and match.group(1).upper() in mapping:
        return mapping[match.group(1).upper()]

    return answer


# ---------------------------------------------------------------------------
# Generative LLM-as-judge
# ---------------------------------------------------------------------------

def compute_llm_judge_aggregation(items: List[Tuple[str, str, str]]) -> float:
    """Score (question, gold, prediction) tuples with GenerativeLLMJudge."""
    judge = _create_generative_judge()
    if judge is None:
        return 0.0

    scores: list[float] = []
    for question, gold, pred in tqdm(items, desc="LLM-as-judge", unit="sample"):
        if not gold or not pred:
            continue
        result = judge.evaluate_answer(
            question=question, reference_answer=gold, given_answer=str(pred),
        )
        scores.append(result["overall_score"])

    if not scores:
        logger.warning("LLM judge produced no scores.")
        return 0.0

    avg = round(mean(scores), 4)
    logger.info("LLM-as-judge average: %.4f (%d samples)", avg, len(scores))
    return avg


if "llm_as_judge_agg" not in le_registry.AGGREGATION_REGISTRY:
    register_aggregation("llm_as_judge_agg")(compute_llm_judge_aggregation)

if "llm_as_judge" not in le_registry.METRIC_REGISTRY:
    register_metric(
        metric="llm_as_judge",
        higher_is_better=True,
        output_type="generate_until",
        aggregation="llm_as_judge_agg",
    )(lambda items: items)


def process_results(doc: Dict[str, Any], results: Any) -> Dict[str, Any]:
    """Collect (question, gold, prediction) for generative judge."""
    pred = results[0] if isinstance(results, list) else results
    gold = doc.get("output", "")
    question = doc.get("input", "")
    instruction = doc.get("instruction", "")
    if instruction:
        question = f"{instruction}\n{question}"
    return {"llm_as_judge": (question, gold, pred)}


class LLMJudgeMetric(BaseMetric):
    """Generative LLM-as-judge metric class."""

    def get_doc_to_text(self, original_doc_to_text: str) -> str:
        return original_doc_to_text

    def get_generation_kwargs(self) -> Dict[str, Any]:
        return {"do_sample": False, "until": []}


_llm_judge_config = MetricConfig(
    name="llm_as_judge",
    higher_is_better=True,
    aggregation_name="llm_as_judge_agg",
    process_results=process_results,
)
get_metrics_registry().register("llm_as_judge", LLMJudgeMetric(_llm_judge_config))


# ---------------------------------------------------------------------------
# MCQ LLM-as-judge
# ---------------------------------------------------------------------------

def compute_mcq_llm_judge_aggregation(items: List[Tuple[str, str, str, list]]) -> float:
    """Score (question, gold, prediction, mcq_options) tuples with MCQLLMJudge."""
    judge = _create_mcq_judge()
    if judge is None:
        return 0.0

    scores: list[float] = []
    for question, gold, pred, mcq_options in tqdm(
        items, desc="MCQ-LLM-as-judge", unit="sample"
    ):
        if not gold or not pred:
            continue
        gold = _normalize_mcq_answer(gold, mcq_options)
        pred = _normalize_mcq_answer(str(pred), mcq_options)
        result = judge.evaluate_answer(
            question=question, reference_answer=gold, given_answer=pred,
        )
        scores.append(result["overall_score"])

    if not scores:
        logger.warning("MCQ LLM judge produced no scores.")
        return 0.0

    avg = round(mean(scores), 4)
    logger.info("MCQ-LLM-as-judge average: %.4f (%d samples)", avg, len(scores))
    return avg


if "mcq_llm_as_judge_agg" not in le_registry.AGGREGATION_REGISTRY:
    register_aggregation("mcq_llm_as_judge_agg")(compute_mcq_llm_judge_aggregation)

if "mcq_llm_as_judge" not in le_registry.METRIC_REGISTRY:
    register_metric(
        metric="mcq_llm_as_judge",
        higher_is_better=True,
        output_type="generate_until",
        aggregation="mcq_llm_as_judge_agg",
    )(lambda items: items)


def mcq_process_results(doc: Dict[str, Any], results: Any) -> Dict[str, Any]:
    """Collect (question, gold, prediction, mcq_options) for MCQ judge."""
    pred = results[0] if isinstance(results, list) else results
    gold = doc.get("output", "")
    question = doc.get("input", "")
    instruction = doc.get("instruction", "")
    if instruction:
        question = f"{instruction}\n{question}"
    mcq_options = doc.get("mcq", [])
    return {"mcq_llm_as_judge": (question, gold, pred, mcq_options)}


class MCQLLMJudgeMetric(BaseMetric):
    """MCQ LLM-as-judge metric class."""

    def get_doc_to_text(self, original_doc_to_text: str) -> str:
        return original_doc_to_text

    def get_generation_kwargs(self) -> Dict[str, Any]:
        return {"do_sample": False, "until": []}


_mcq_llm_judge_config = MetricConfig(
    name="mcq_llm_as_judge",
    higher_is_better=True,
    aggregation_name="mcq_llm_as_judge_agg",
    process_results=mcq_process_results,
)
get_metrics_registry().register(
    "mcq_llm_as_judge", MCQLLMJudgeMetric(_mcq_llm_judge_config)
)
