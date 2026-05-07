"""LLM Judge operations for evaluation results."""

import logging
import traceback
from statistics import mean
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from tqdm import tqdm

from src.llm_judger.base_llm_judge import ModelConfig
from src.llm_judger.generative_llm_judge import GenerativeLLMJudge
from src.llm_judger.mcq_llm_judge import MCQLLMJudge

if TYPE_CHECKING:
    from src.processors.task_operations import TaskOperations

logger = logging.getLogger(__name__)


class LLMJudgeProcessor:
    """Handles LLM judge processing of evaluation results."""

    def __init__(
        self,
        task_operations: "TaskOperations",
        llm_judge_api_keys: list[str] | None = None,
        llm_judge_models: list[str] | None = None,
        llm_judge_providers: list[str] | None = None,
    ):
        """Initialize LLM judge processor.

        Args:
            task_operations: TaskOperations instance for task utilities
            llm_judge_api_keys: API keys for LLM judge models (comma-separated in env)
            llm_judge_models: Names of LLM judge models (comma-separated in env)
            llm_judge_providers: Providers of LLM judge models (comma-separated in env)
        """
        self.task_operations = task_operations
        self.model_configs = self._build_model_configs(
            llm_judge_models or [],
            llm_judge_providers or [],
            llm_judge_api_keys or [],
        )

    @staticmethod
    def _build_model_configs(
        models: list[str],
        providers: list[str],
        api_keys: list[str],
    ) -> list[ModelConfig]:
        """Build ModelConfig list from parallel lists, broadcasting single values."""
        if not models or not providers:
            return []

        n = max(len(models), len(providers))

        if len(models) == 1 and n > 1:
            models = models * n
        if len(providers) == 1 and n > 1:
            providers = providers * n

        api_keys_resolved: list[str | None]
        if not api_keys:
            api_keys_resolved = [None] * n
        elif len(api_keys) == 1 and n > 1:
            api_keys_resolved = [api_keys[0]] * n
        else:
            api_keys_resolved = [k for k in api_keys]  # pylint: disable=unnecessary-comprehension

        if len(models) != len(providers) or len(providers) != len(api_keys_resolved):
            raise ValueError(
                f"JUDGE_MODEL ({len(models)}), JUDGE_PROVIDER ({len(providers)}), "
                f"and JUDGE_API_KEY ({len(api_keys_resolved)}) must have the same "
                f"number of comma-separated values (or a single value to apply to all)"
            )

        return [
            ModelConfig(name=m, provider=p, api_key=k)  # type: ignore[arg-type]
            for m, p, k in zip(models, providers, api_keys_resolved)
        ]

    def should_process_llm_judge(self) -> bool:
        """Check if LLM judge processing should be performed.

        Returns:
            True if at least one model config is present
        """
        return bool(self.model_configs)

    def process_results_with_llm_judge(
        self,
        results_data: Dict[str, Any],
        llm_judge,  # Can be MCQLLMJudge or GenerativeLLMJudge
        is_mcq: bool = False,
        task_filter: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Process results with LLM judge.

        Args:
            results_data: The evaluation results
            llm_judge: The LLM judge instance (MCQLLMJudge or GenerativeLLMJudge)
            is_mcq: Whether this is for MCQ tasks (affects prefix in output keys)
            task_filter: Optional list of task names to process. If None, process all tasks.

        Returns:
            Updated results with LLM judge scores
        """
        processed_data = results_data.copy()

        samples = processed_data.get("samples", {})
        processed_data.setdefault("results", {})

        all_scores: list[float] = []
        all_scores_raw: list[float] = []

        # Filter samples if task_filter is provided
        if task_filter:
            filtered_samples = [
                (task_key, sample)
                for task_key, sample_list in samples.items()
                if task_key in task_filter
                for sample in sample_list
            ]
            logger.info(
                "Filtering to %s tasks, %s total samples",
                len(task_filter),
                len(filtered_samples),
            )
        else:
            filtered_samples = [
                (task_key, sample)
                for task_key, sample_list in samples.items()
                for sample in sample_list
            ]

        taskwise_scores: dict[str, list[float]] = {}
        taskwise_scores_raw: dict[str, list[float]] = {}

        prefix = "mcq_" if is_mcq else ""
        pbar_desc = "Evaluating (MCQ)" if is_mcq else "Evaluating (Gen)"

        for sample_key, sample in tqdm(filtered_samples, desc=pbar_desc, unit="sample"):
            logger.debug("Processing sample: %s", sample_key)
            if not isinstance(sample, dict):
                logger.warning(
                    "Sample %s is not a dict, skipping. Type: %s",
                    sample_key,
                    type(sample),
                )
                continue

            responses = sample.get("filtered_resps", [])
            logger.debug("Sample %s: Found %s filtered responses", sample_key, len(responses))
            if not responses:
                logger.warning("Sample %s: No filtered responses found, skipping", sample_key)
                continue

            response = responses[0]
            expected_output = sample.get("doc", {}).get("output", "")
            question = sample.get("doc", {}).get("input", "")

            logger.debug("Sample %s:", sample_key)
            logger.debug("  Question length: %s chars", len(question))
            logger.debug("  Expected output length: %s chars", len(expected_output))
            logger.debug("  Response type: %s", type(response))
            logger.debug(
                "  Response value: %s",
                repr(response) if len(str(response)) < 200 else repr(str(response)[:200]),
            )

            # For MCQ tasks, normalize answers to text BEFORE sending to judge
            mcq_options = sample.get("doc", {}).get("mcq", [])
            logger.debug("Sample %s: MCQ options: %s", sample_key, mcq_options)

            if is_mcq and mcq_options:
                # Create letter-to-text mapping
                mcq_mapping = {chr(65 + i): opt for i, opt in enumerate(mcq_options)}
                logger.debug("Sample %s: MCQ mapping: %s", sample_key, mcq_mapping)

                # Normalize both the model's response and the expected output
                original_response = response
                original_expected = expected_output

                logger.debug(
                    "Sample %s: Before normalization - Response type: %s, Expected type: %s",
                    sample_key,
                    type(response),
                    type(expected_output),
                )
                logger.debug(
                    "Sample %s: Before normalization - Response: %s, Expected: %s",
                    sample_key,
                    repr(original_response),
                    repr(original_expected),
                )

                response = self.task_operations.normalize_mcq_answer(response, mcq_mapping)
                expected_output = self.task_operations.normalize_mcq_answer(
                    expected_output, mcq_mapping
                )

                logger.debug("Sample %s: MCQ Normalization:", sample_key)
                logger.debug("  Response: '%s' → '%s'", original_response, response)
                logger.debug("  Expected: '%s' → '%s'", original_expected, expected_output)

            if expected_output and response:
                try:
                    logger.debug("Sample %s: Calling LLM judge...", sample_key)
                    logger.debug("  Question length: %s", len(question))
                    logger.debug("  Reference answer length: %s", len(expected_output))
                    logger.debug("  Given answer length: %s", len(response))

                    evaluation_result = llm_judge.evaluate_answer(
                        question=question,
                        reference_answer=expected_output,
                        given_answer=response,
                        context=None,
                        test_id=f"{sample_key}_{len(all_scores)}",
                        metadata={"task": sample_key},
                    )

                    logger.debug(
                        "Sample %s: LLM judge result keys: %s",
                        sample_key,
                        list(evaluation_result.keys()),
                    )

                    normalized_score = evaluation_result["overall_score"]
                    raw_score = evaluation_result["overall_raw_score"]
                    explanation = evaluation_result["aggregated_explanation"]

                    logger.debug(
                        "Sample %s: Scores - Normalized: %s, Raw: %s",
                        sample_key,
                        normalized_score,
                        raw_score,
                    )

                    sample[f"{prefix}llm_score"] = normalized_score
                    sample[f"{prefix}llm_score_raw"] = raw_score
                    sample[f"{prefix}llm_explanation"] = explanation
                    sample[f"{prefix}llm_judge_details"] = {
                        "model_results": evaluation_result["model_results"],
                        "aggregation_method": evaluation_result["aggregation_method"],
                        "passed": evaluation_result["overall_passed"],
                    }

                    all_scores.append(normalized_score)
                    all_scores_raw.append(raw_score)
                    taskwise_scores.setdefault(sample_key, []).append(normalized_score)
                    taskwise_scores_raw.setdefault(sample_key, []).append(raw_score)

                except Exception as e: # pylint: disable=broad-except
                    logger.error("Sample %s: ❌ LLM evaluation failed", sample_key)
                    logger.error("  Error type: %s", type(e).__name__)
                    logger.error("  Error message: %s", str(e))
                    logger.error("  Question length: %s", len(question))
                    logger.error(
                        "  Expected output type: %s, value: %s",
                        type(expected_output),
                        repr(expected_output)[:100],
                    )
                    logger.error(
                        "  Response type: %s, value: %s",
                        type(response),
                        repr(response)[:100],
                    )
                    logger.error("  Traceback: %s", traceback.format_exc())

                    sample[f"{prefix}llm_score"] = None
                    sample[f"{prefix}llm_score_raw"] = None
                    sample[f"{prefix}llm_explanation"] = f"Error during LLM evaluation: {str(e)}"
                    sample[f"{prefix}llm_judge_details"] = {"error": str(e)}
            else:
                logger.warning("Sample %s: Missing expected_output or response", sample_key)
                logger.warning(
                    "  expected_output: %s",
                    repr(expected_output)[:50] if expected_output else "MISSING",
                )
                logger.warning(
                    "  response: %s",
                    repr(response)[:50] if response else "MISSING",
                )

        # Aggregate scores per task
        for task_key, scores in taskwise_scores.items():
            if scores:
                avg_score = round(mean(scores), 4)
                avg_score_raw = round(mean(taskwise_scores_raw[task_key]), 4)

                # Ensure the task result exists and is a dict, create if needed
                if task_key not in processed_data["results"]:
                    processed_data["results"][task_key] = {"alias": task_key}
                    logger.debug("Created new result entry for task '%s'", task_key)
                elif not isinstance(processed_data["results"][task_key], dict):
                    logger.warning(
                        "Task '%s' has non-dict result, converting to dict", task_key
                    )
                    processed_data["results"][task_key] = {"alias": task_key}

                processed_data["results"][task_key][
                    f"{prefix}llm_judge_score,none"
                ] = avg_score
                processed_data["results"][task_key][
                    f"{prefix}llm_judge_score_stderr,none"
                ] = 0.0
                processed_data["results"][task_key][
                    f"{prefix}llm_judge_score_raw,none"
                ] = avg_score_raw
                processed_data["results"][task_key][
                    f"{prefix}llm_judge_score_raw_stderr,none"
                ] = 0.0
                processed_data["results"][task_key][f"{prefix}llm_as_judge"] = {
                    "average_score": avg_score,
                    "average_score_raw": avg_score_raw,
                    "num_samples": len(scores),
                }

                logger.debug(
                    "Task '%s': Added LLM judge scores (avg=%s, avg_raw=%s, samples=%s)",
                    task_key,
                    avg_score,
                    avg_score_raw,
                    len(scores),
                )

        # Add overall statistics
        if all_scores:
            processed_data[f"overall_{prefix}llm_as_judge"] = round(mean(all_scores), 4)
            processed_data[f"overall_{prefix}llm_as_judge_stats"] = {
                "total_samples": len(all_scores),
                "average_score": round(mean(all_scores), 4),
                "min_score": min(all_scores),
                "max_score": max(all_scores),
            }

        if all_scores_raw:
            processed_data[f"overall_{prefix}llm_as_judge_raw"] = round(
                mean(all_scores_raw), 4
            )
            processed_data[f"overall_{prefix}llm_as_judge_raw_stats"] = {
                "total_samples": len(all_scores_raw),
                "average_score_raw": round(mean(all_scores_raw), 4),
                "min_score_raw": min(all_scores_raw),
                "max_score_raw": max(all_scores_raw),
            }

        return processed_data

    def process_generation_tasks(
        self, results_data: Dict[str, Any], generation_tasks: list[str]
    ) -> Dict[str, Any]:
        """Process generation tasks with LLM judge.

        Args:
            results_data: Evaluation results
            generation_tasks: List of generation task names to process

        Returns:
            Updated results with LLM judge scores
        """
        if not generation_tasks or not self.should_process_llm_judge():
            logger.info("No generation tasks or LLM judge config, skipping.")
            return results_data

        logger.info(
            "Processing %s generation tasks with GenerativeLLMJudge...",
            len(generation_tasks),
        )

        llm_judge_generation = GenerativeLLMJudge(
            model_configs=self.model_configs,
            custom_prompt=None,
            aggregation_method="mean",
            threshold=0.5,
        )

        return self.process_results_with_llm_judge(
            results_data=results_data,
            llm_judge=llm_judge_generation,
            is_mcq=False,
            task_filter=generation_tasks,
        )

    def process_mcq_tasks(
        self, results_data: Dict[str, Any], mcq_tasks: list[str]
    ) -> Dict[str, Any]:
        """Process MCQ tasks with LLM judge.

        Args:
            results_data: Evaluation results
            mcq_tasks: List of MCQ task names to process

        Returns:
            Updated results with LLM judge scores
        """
        if not mcq_tasks or not self.should_process_llm_judge():
            logger.info("No MCQ tasks or LLM judge config, skipping.")
            return results_data

        logger.info("Processing %s MCQ tasks with MCQLLMJudge...", len(mcq_tasks))

        llm_judge_mcq = MCQLLMJudge(
            model_configs=self.model_configs,
            custom_prompt=None,
            aggregation_method="mean",
            threshold=0.5,
        )

        return self.process_results_with_llm_judge(
            results_data=results_data,
            llm_judge=llm_judge_mcq,
            is_mcq=True,
            task_filter=mcq_tasks,
        )
