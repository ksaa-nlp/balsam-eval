"""Unified LLM-as-judge metric for evaluation.

Auto-detects MCQ vs generative based on the dataset doc's ``mcq`` field
and selects the appropriate judge prompt. Also forwards per-doc
``custom_prompt`` to the judge when present in the dataset file.

Cloud builds use JUDGE_CONFIGS_B64. Local runs may use the legacy
JUDGE_MODEL, JUDGE_PROVIDER, and JUDGE_API_KEY variables.
"""

import base64
import json
import logging
import os
import re
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from lm_eval.api import registry as le_registry
from lm_eval.api.registry import register_aggregation, register_metric
from tqdm import tqdm

from src.llm_judger.base_llm_judge import ModelConfig
from src.llm_judger.generative_llm_judge import GenerativeLLMJudge
from src.llm_judger.mcq_llm_judge import MCQLLMJudge
from src.metrics_registry import BaseMetric, MetricConfig, get_metrics_registry

logger = logging.getLogger(__name__)

# (question, gold, pred, mcq_options | None, custom_prompt | None)
JudgeItem = Tuple[str, str, str, Optional[list], Optional[str]]
JudgeCacheKey = Tuple[str, str, str, Tuple[str, ...], Optional[str]]
_JUDGE_SCORE_CACHE: dict[JudgeCacheKey, Optional[float]] = {}


def _parse_csv_env(name: str) -> list[str]:
    """Split a comma-separated env var into a trimmed list."""
    raw = os.getenv(name, "")
    return [v.strip() for v in raw.split(",") if v.strip()] if raw else []


def _get_judge_configs() -> list[ModelConfig]:
    """Build ModelConfig list from env vars.

    Supports comma-separated values for multiple judges.
    A single API key is broadcast to all judges.
    """
    encoded_configs = os.getenv("JUDGE_CONFIGS_B64")
    if encoded_configs:
        try:
            raw_configs = json.loads(base64.b64decode(encoded_configs, validate=True))
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError("JUDGE_CONFIGS_B64 must contain valid base64 JSON") from exc

        if not isinstance(raw_configs, list) or not raw_configs:
            raise ValueError("JUDGE_CONFIGS_B64 must contain a non-empty JSON array")

        configs: list[ModelConfig] = []
        for index, raw_config in enumerate(raw_configs):
            if not isinstance(raw_config, dict):
                raise ValueError(f"Judge config at index {index} must be an object")

            model = raw_config.get("model")
            provider = raw_config.get("provider")
            if not isinstance(model, str) or not model.strip():
                raise ValueError(f"Judge config at index {index} requires model")
            if not isinstance(provider, str) or not provider.strip():
                raise ValueError(f"Judge config at index {index} requires provider")

            api_key_env = raw_config.get("apiKeyEnv")
            if api_key_env is not None and not isinstance(api_key_env, str):
                raise ValueError(f"Judge config at index {index} has invalid apiKeyEnv")
            base_url = raw_config.get("baseUrl")
            if base_url is not None and not isinstance(base_url, str):
                raise ValueError(f"Judge config at index {index} has invalid baseUrl")
            custom_prompt = raw_config.get("customPrompt")
            if custom_prompt is not None and not isinstance(custom_prompt, str):
                raise ValueError(f"Judge config at index {index} has invalid customPrompt")

            configs.append(
                ModelConfig(
                    name=model.strip(),
                    provider=provider.strip(),  # type: ignore[arg-type]
                    api_key=os.getenv(api_key_env) if api_key_env else None,
                    endpoint_url=base_url,
                    custom_prompt=custom_prompt,
                )
            )
        return configs

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
    invalid = [
        name
        for name, values in (
            ("JUDGE_MODEL", models),
            ("JUDGE_PROVIDER", providers),
            ("JUDGE_API_KEY", api_keys),
        )
        if values and len(values) not in (1, n)
    ]
    if invalid:
        raise ValueError(
            f"Judge configuration lengths must be 1 or {n}: {', '.join(invalid)}"
        )
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


_GENERATIVE_JUDGE: Optional[GenerativeLLMJudge] = None
_MCQ_JUDGE: Optional[MCQLLMJudge] = None


def _get_generative_judge() -> Optional[GenerativeLLMJudge]:
    global _GENERATIVE_JUDGE  # pylint: disable=global-statement
    if _GENERATIVE_JUDGE is None:
        configs = _get_judge_configs()
        if not configs:
            return None
        _GENERATIVE_JUDGE = GenerativeLLMJudge(
            model_configs=configs, aggregation_method="mean", threshold=0.5,
        )
    return _GENERATIVE_JUDGE


def _get_mcq_judge() -> Optional[MCQLLMJudge]:
    global _MCQ_JUDGE  # pylint: disable=global-statement
    if _MCQ_JUDGE is None:
        configs = _get_judge_configs()
        if not configs:
            return None
        _MCQ_JUDGE = MCQLLMJudge(
            model_configs=configs, aggregation_method="mean", threshold=0.5,
        )
    return _MCQ_JUDGE


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
# Unified LLM-as-judge
# ---------------------------------------------------------------------------

def _judge_cache_key(item: JudgeItem) -> JudgeCacheKey:
    question, gold, pred, mcq_options, custom_prompt = item
    options = tuple(str(option) for option in (mcq_options or []))
    return str(question), str(gold), str(pred), options, custom_prompt


def _score_judge_item(item: JudgeItem) -> Optional[float]:
    question, gold, pred, mcq_options, custom_prompt = item
    if not gold or not pred:
        return None

    ref = gold
    answer = str(pred)

    if mcq_options:
        mcq_judge = _get_mcq_judge()
        if mcq_judge is None:
            return None
        ref = _normalize_mcq_answer(ref, mcq_options)
        answer = _normalize_mcq_answer(answer, mcq_options)
        result = mcq_judge.evaluate_answer(
            question=question,
            reference_answer=ref,
            given_answer=answer,
            custom_prompt=custom_prompt,
        )
    else:
        gen_judge = _get_generative_judge()
        if gen_judge is None:
            return None
        result = gen_judge.evaluate_answer(
            question=question,
            reference_answer=ref,
            given_answer=answer,
            custom_prompt=custom_prompt,
        )

    return float(result["overall_score"])


def compute_llm_judge_aggregation(items: List[JudgeItem]) -> float:
    """Score items once and reuse scores for per-question result export."""
    keys = [_judge_cache_key(item) for item in items]
    uncached = [item for item, key in zip(items, keys) if key not in _JUDGE_SCORE_CACHE]

    for item in tqdm(uncached, desc="LLM-as-judge", unit="sample", disable=not uncached):
        key = _judge_cache_key(item)
        _JUDGE_SCORE_CACHE[key] = _score_judge_item(item)

    scores: list[float] = []
    for key in keys:
        score = _JUDGE_SCORE_CACHE[key]
        if score is not None:
            scores.append(score)

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
    """Collect judge data, auto-detecting MCQ vs generative from the doc."""
    pred = results[0] if isinstance(results, list) and results else ""
    gold = doc.get("output", "")
    question = doc.get("input", "")
    instruction = doc.get("instruction", "")
    if instruction:
        question = f"{instruction}\n{question}"
    mcq_options = doc.get("mcq") or None
    custom_prompt = doc.get("custom_prompt") or None
    return {"llm_as_judge": (question, gold, pred, mcq_options, custom_prompt)}


class LLMJudgeMetric(BaseMetric):
    """Unified LLM-as-judge metric class."""

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
