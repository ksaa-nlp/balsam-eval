"""This module contains the code for running an evaluation job."""

import os
import sys
import json
import logging
import traceback
from pathlib import Path
from typing import Any, List, Literal, Optional, Dict
from statistics import mean

import lm_eval.evaluator
import lm_eval.tasks
import lm_eval.models  # Register all lm_eval models (openai-chat-completions, etc.)
import requests
from tqdm import tqdm

from src.helpers import normalize_string
from src.llm_judger.base_llm_judge import ModelConfig
from src.llm_judger.mcq_llm_judge import MCQLLMJudge
from src.llm_judger.generative_llm_judge import GenerativeLLMJudge
from src.db_operations import JobStatus, add_results_to_db, update_status
from src.gemini_adapter import GeminiLM
from src.groq_adapter import GroqLM
import src.metrics

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if os.environ.get("ENV", "PROD") == "local":
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


_original_relative_to = Path.relative_to


def _safe_relative_to(self, *args, **kwargs):
    try:
        return _original_relative_to(self, *args, **kwargs)
    except ValueError as e:
        if "not in the subpath of" in str(e):
            return self
        raise e


Path.relative_to = _safe_relative_to


requests.post = lambda url, timeout=5000, **kwargs: requests.request(
    method="POST", url=url, timeout=timeout, **kwargs
)


class EvaluationJob:
    """
    Construct an evaluation job.
    A job is identified by a unique job ID, which should exist
    as an environment variable before running it.

    Args:
        tasks (`List[str]`): The tasks to evaluate, these names correspond
        to YAML files that should exist in the current working directory.
    """

    def __init__(
        self,
        tasks: List[str],
        model_args: dict[str, Any],
        category_name: str,
        tasks_mapper_dict: Optional[dict] = None,
        adapter: Literal[
            "local-chat-completions",
            "openai-chat-completions",
            "anthropic-chat-completions",
            "gemini",
        ] = "local-chat-completions",
        job_id: Optional[str] = None,
        api_host: Optional[str] = None,
        server_token: Optional[str] = None,
        benchmark_id: Optional[str] = None,
        llm_judge_model: Optional[str] = None,
        llm_judge_provider: Optional[str] = None,
        llm_judge_api_key: Optional[str] = None,
        task_id: Optional[str] = None,
    ):
        self.model_args = model_args or {}
        self.tasks: List[str] = tasks
        self.adapter = adapter
        self.job_id = job_id
        self.category_name = category_name
        self.tasks_mapper_dict = tasks_mapper_dict
        self.benchmark_id = benchmark_id
        self.api_host = api_host
        self.server_token = server_token
        self.llm_judge_model = llm_judge_model
        self.llm_judge_provider = llm_judge_provider
        self.llm_judge_api_key = llm_judge_api_key
        self.task_id = task_id

        # FIX #2: Add eos_string to model_args if not present to avoid EOS warning
        if "eos_string" not in self.model_args:
            self.model_args["eos_string"] = "<|endoftext|>"
            logger.info("Added default eos_string='<|endoftext|>' to model_args")

        API_KEY = os.getenv("API_KEY")
        if API_KEY and (
            self.adapter == "openai-chat-completions"
            or self.adapter == "local-chat-completions"
        ):
            os.environ["OPENAI_API_KEY"] = API_KEY

        if API_KEY and self.adapter == "anthropic-chat-completions":
            os.environ["ANTHROPIC_API_KEY"] = API_KEY

        if API_KEY and self.adapter == "gemini":
            os.environ["GOOGLE_API_KEY"] = API_KEY

    def _extract_api_error_message(self, exception: Exception):
        """Extract the full API error object from various exception types."""
        try:
            if hasattr(exception, "response") and exception.response is not None:
                try:
                    error_data = exception.response.json()
                    if "error" in error_data:
                        return error_data
                except (json.JSONDecodeError, ValueError):
                    pass

            error_str = str(exception)

            if "API request failed with error message:" in error_str:
                import re

                json_match = re.search(r"\{(?:[^{}]|{[^{}]*})*\}", error_str)
                if json_match:
                    try:
                        error_json = json.loads(json_match.group())
                        if "error" in error_json:
                            return error_json
                    except json.JSONDecodeError:
                        pass

            if hasattr(exception, "args") and exception.args:
                for arg in exception.args:
                    if isinstance(arg, dict) and "error" in arg:
                        return arg
                    elif isinstance(arg, str):
                        try:
                            parsed = json.loads(arg)
                            if "error" in parsed:
                                return parsed
                        except json.JSONDecodeError:
                            pass

            return {
                "error": {
                    "message": str(exception),
                    "type": type(exception).__name__,
                    "param": None,
                    "code": "unknown",
                }
            }

        except Exception as e:
            logger.error(f"Error while extracting API error message: {e}")
            return {
                "error": {
                    "message": str(exception),
                    "type": "extraction_error",
                    "param": None,
                    "code": "unknown",
                }
            }

    def __call__(self):
        """Run the evaluation job."""
        if self.api_host and self.job_id and self.server_token:
            update_status(
                api_host=self.api_host,
                job_id=self.job_id,
                server_token=self.server_token,
                status=JobStatus.RUNNING,
            )
        try:
            logger.info("=" * 80)
            logger.info(f"Starting evaluation job for category: {self.category_name}")
            logger.info(f"Tasks: {self.tasks}")
            logger.info(f"Adapter: {self.adapter}")
            logger.info(f"Model args: {self.model_args}")
            logger.info("=" * 80)

            temp_dir = Path(".temp").resolve()

            logger.info("Calling lm_eval.evaluator.simple_evaluate...")
            results = lm_eval.evaluator.simple_evaluate(
                model=self.adapter,
                model_args=self.model_args,
                tasks=self.tasks,
                apply_chat_template=True,
                task_manager=lm_eval.tasks.TaskManager(include_path=str(temp_dir)),
                batch_size=1,
            )

            logger.info("✅ simple_evaluate completed successfully")
            logger.info("Exporting results to %s.json", self.category_name)

            results = self._add_task_to_results(results=results)

            if not results:
                logger.warning("No results found for the evaluation job.")
                if self.api_host and self.job_id and self.server_token:
                    update_status(
                        api_host=self.api_host,
                        job_id=self.job_id,
                        server_token=self.server_token,
                        status=JobStatus.FAILED,
                        error_message="No results found for the evaluation job.",
                    )
                return

            # Detect which tasks are MCQ (accuracy) and which are generation
            mcq_tasks, generation_tasks = self._separate_mcq_and_generation_tasks(
                results
            )

            updated_results = results

            # Run LLM judge for generation tasks if needed
            if (
                generation_tasks
                and self.llm_judge_api_key
                and self.llm_judge_model
                and self.llm_judge_provider
            ):
                logger.info(
                    f"Processing {len(generation_tasks)} generation tasks with GenerativeLLMJudge..."
                )
                llm_judge_generation = GenerativeLLMJudge(
                    model_configs=[
                        ModelConfig(
                            name=self.llm_judge_model,
                            provider=self.llm_judge_provider,
                            api_key=self.llm_judge_api_key,
                        )
                    ],
                    custom_prompt=None,  # Use default prompt for generation (0-3 scale)
                    aggregation_method="mean",
                    threshold=0.5,
                )
                updated_results = self.process_results_with_llm_judge(
                    results_data=updated_results,
                    llm_judge=llm_judge_generation,
                    is_mcq=False,
                    task_filter=generation_tasks,
                )

            # Run LLM judge for MCQ tasks if needed
            if (
                mcq_tasks
                and self.llm_judge_api_key
                and self.llm_judge_model
                and self.llm_judge_provider
            ):
                logger.info(
                    f"Processing {len(mcq_tasks)} MCQ tasks with MCQLLMJudge..."
                )
                llm_judge_mcq = MCQLLMJudge(
                    model_configs=[
                        ModelConfig(
                            name=self.llm_judge_model,
                            provider=self.llm_judge_provider,
                            api_key=self.llm_judge_api_key,
                        )
                    ],
                    custom_prompt=None,  # Use default MCQ prompt (binary 0-1 scoring)
                    aggregation_method="mean",
                    threshold=0.5,
                )
                updated_results = self.process_results_with_llm_judge(
                    results_data=updated_results,
                    llm_judge=llm_judge_mcq,
                    is_mcq=True,
                    task_filter=mcq_tasks,
                )

            if not generation_tasks and not mcq_tasks:
                logger.info(
                    "No LLM judge processing needed (no judge config or no tasks)."
                )

            if (
                "config" in updated_results
                and "model_args" in updated_results["config"]
                and "api_key" in updated_results["config"]["model_args"]
            ):
                updated_results["config"]["model_args"].pop("api_key")

            if self.task_id is None:
                self._export_results(updated_results)
            else:
                self._export_results_tasks(updated_results)

            if self.api_host and self.job_id and self.server_token:
                update_status(
                    api_host=self.api_host,
                    job_id=self.job_id,
                    server_token=self.server_token,
                    status=JobStatus.COMPLETED,
                )
            logger.info("✅ Evaluation job completed successfully")

        except Exception as e:
            logger.error(f"❌ Error in evaluation job: {e}")
            logger.error(traceback.format_exc())
            api_error_data = self._extract_api_error_message(e)

            if self.api_host and self.job_id and self.server_token:
                error_message = (
                    json.dumps(api_error_data)
                    if isinstance(api_error_data, dict)
                    else str(api_error_data)
                )
                update_status(
                    api_host=self.api_host,
                    job_id=self.job_id,
                    server_token=self.server_token,
                    status=JobStatus.FAILED,
                    error_message=error_message,
                )

            error_message = (
                json.dumps(api_error_data)
                if isinstance(api_error_data, dict)
                else str(api_error_data)
            )
            raise Exception(error_message) from e

    def _add_task_to_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        FIX #1: Enhanced task mapping with better fallback logic

        Task names follow pattern: "Task_Category_Type_Metric_Hash"
        Example: "Program Execution_Program Execution_generative_rouge_458b3abaf4"

        Based on task_v2.py generation code:
        - Task: "Program Execution" (FIRST part) ← This is what we want!
        - Category: "Program Execution" (duplicated)
        - Type: "generative" or "mcq"
        - Metric: "rouge", "bleu", "accuracy"
        - Hash: 10-character unique ID
        """
        if "results" not in results:
            logger.warning("No 'results' key found in results dictionary")
            return results

        for task_name, task_result in results["results"].items():
            if isinstance(task_result, dict):
                if "task" not in task_result:
                    # First try: use explicit task_id if provided
                    if self.task_id:
                        task_result["task"] = self.task_id
                        logger.info(
                            f"Task ID set from explicit task_id: {self.task_id}"
                        )
                    # Second try: use tasks_mapper if available
                    elif self.tasks_mapper_dict and task_name in self.tasks_mapper_dict:
                        task_result["task"] = self.tasks_mapper_dict[task_name]
                        logger.info(
                            f"Task ID mapped: {task_name} → {task_result['task']}"
                        )
                    # Third try: parse from task_name (NEW LOGIC - extract first part)
                    else:
                        parts = task_name.split("_")
                        if len(parts) >= 5:
                            # First part is the task name
                            extracted_task = parts[0]
                            task_result["task"] = extracted_task
                            logger.info(
                                f"Task ID extracted from name: {task_name} → {extracted_task}"
                            )
                        else:
                            # Fallback: use the task_name as-is
                            task_result["task"] = task_name
                            logger.warning(
                                f"Task ID fallback to full name: {task_name}"
                            )

        return results

    def _separate_mcq_and_generation_tasks(
        self, results: Dict[str, Any]
    ) -> tuple[list[str], list[str]]:
        """
        Separate tasks into MCQ (accuracy metric) and generation tasks.

        Returns:
            tuple: (mcq_tasks, generation_tasks) - lists of task names
        """
        mcq_tasks = []
        generation_tasks = []

        configs = results.get("configs", {})

        for task_name, task_config in configs.items():
            metric_list = task_config.get("metric_list", [])
            if metric_list:
                # Get the first metric
                first_metric = metric_list[0]
                metric_name = (
                    first_metric.get("metric")
                    if isinstance(first_metric, dict)
                    else first_metric
                )

                if metric_name == "accuracy":
                    mcq_tasks.append(task_name)
                    logger.debug(
                        f"Task '{task_name}' identified as MCQ (accuracy metric)"
                    )
                else:
                    generation_tasks.append(task_name)
                    logger.debug(
                        f"Task '{task_name}' identified as generation (metric: {metric_name})"
                    )
            else:
                # No metric specified, default to generation
                generation_tasks.append(task_name)
                logger.debug(
                    f"Task '{task_name}' has no metric, defaulting to generation"
                )

        logger.info(
            f"Separated tasks: {len(mcq_tasks)} MCQ, {len(generation_tasks)} generation"
        )
        return mcq_tasks, generation_tasks

    def _normalize_mcq_answer(self, answer: str, mcq_mapping: dict) -> str:
        """
        Normalize MCQ answer to full text using the mapping.
        Converts letter answers (A, B, C, D) to their full text equivalents.

        Args:
            answer: The answer to normalize (could be "A", "B", or full text)
            mcq_mapping: Dict mapping letters to full option text (e.g., {"A": "نعم", "B": "لا"})

        Returns:
            Normalized answer as full text
        """
        if not answer or not mcq_mapping:
            return answer

        answer_stripped = answer.strip()

        # Check if it's a single letter
        if len(answer_stripped) == 1 and answer_stripped.upper() in mcq_mapping:
            normalized = mcq_mapping[answer_stripped.upper()]
            logger.debug(f"Converted letter '{answer_stripped}' to '{normalized}'")
            return normalized

        # Check for "A)" format
        import re

        match = re.match(r"^([A-Za-z])\)", answer_stripped)
        if match:
            letter = match.group(1).upper()
            if letter in mcq_mapping:
                normalized = mcq_mapping[letter]
                logger.debug(f"Converted '{letter})' to '{normalized}'")
                return normalized

        # If it's already full text, return as-is
        return answer

    def _calculate_average_scores(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate average scores across all tasks."""
        average_scores = {}
        all_scores = {}

        if "results" not in results:
            return average_scores

        for task_name, task_result in results["results"].items():
            if not isinstance(task_result, dict):
                continue

            for key, value in task_result.items():
                if key.endswith(",none"):
                    # Handle ROUGE (dictionary with multiple scores)
                    if isinstance(value, dict):
                        # For ROUGE, use rougeLsum as the representative score
                        if "rougeLsum" in value:
                            metric_name = key.replace(",none", "")
                            if metric_name not in all_scores:
                                all_scores[metric_name] = []
                            all_scores[metric_name].append(value["rougeLsum"])
                            logger.debug(
                                f"Task '{task_name}': Added ROUGE rougeLsum={value['rougeLsum']}"
                            )
                    # Handle numeric values (BLEU, accuracy, etc.)
                    elif isinstance(value, (int, float)):
                        metric_name = key.replace(",none", "")
                        if metric_name not in all_scores:
                            all_scores[metric_name] = []
                        all_scores[metric_name].append(value)
                        logger.debug(f"Task '{task_name}': Added {metric_name}={value}")

        for metric_name, scores in all_scores.items():
            if scores:
                average_scores[metric_name] = round(mean(scores), 4)
                logger.info(
                    f"Average {metric_name}: {average_scores[metric_name]} (from {len(scores)} tasks)"
                )

        return average_scores

    def _export_results(self, results: dict[str, Any]):
        average_scores = self._calculate_average_scores(results)
        results_with_averages = {**results, "average_scores": average_scores}

        with open(
            os.path.join(".results", f"{normalize_string(self.category_name)}.json"),
            "w",
            encoding="UTF-8",
        ) as fp:
            json.dump(results_with_averages, fp, ensure_ascii=False)

        logger.info(f"Results exported to {normalize_string(self.category_name)}.json")

        if self.api_host and self.job_id and self.server_token and self.benchmark_id:
            if "results" in results_with_averages:
                for key, value in results_with_averages["results"].items():
                    add_results_to_db(
                        api_host=self.api_host,
                        job_id=self.job_id,
                        task_id=value["task"],
                        server_token=self.server_token,
                        result=value,
                        category_name=self.category_name,
                        benchmark_id=self.benchmark_id,
                    )

    def _export_results_tasks(self, results: dict[str, Any]):
        average_scores = self._calculate_average_scores(results)
        results_with_averages = {**results, "average_scores": average_scores}

        filename = f"{normalize_string(self.task_id)}.json"

        with open(filename, "w", encoding="UTF-8") as fp:
            json.dump(results_with_averages, fp, ensure_ascii=False)

        logger.info(f"Results exported to {filename}")

        if self.api_host and self.job_id and self.server_token and self.benchmark_id:
            if "results" in results_with_averages:
                for key, value in results_with_averages["results"].items():
                    if self.task_id is None:
                        logger.warning(f"No mapping found for task '{key}'")
                        raise ValueError("No task_id or tasks_mapper provided.")
                    add_results_to_db(
                        api_host=self.api_host,
                        job_id=self.job_id,
                        task_id=normalize_string(self.task_id),
                        server_token=self.server_token,
                        result=value,
                        category_name=self.category_name,
                        benchmark_id=self.benchmark_id,
                    )

    def process_results_with_llm_judge(
        self,
        results_data: Dict[str, Any],
        llm_judge,  # Can be MCQLLMJudge or GenerativeLLMJudge
        is_mcq: bool = False,
        task_filter: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Process results with LLM judge.

        Args:
            results_data: The evaluation results
            llm_judge: The LLM judge instance (MCQLLMJudge or GenerativeLLMJudge)
            is_mcq: Whether this is for MCQ tasks (affects prefix in output keys)
            task_filter: Optional list of task names to process. If None, process all tasks.
        """
        processed_data = results_data.copy()

        samples = processed_data.get("samples", {})
        processed_data.setdefault("results", {})

        all_scores = []
        all_scores_raw = []

        # Filter samples if task_filter is provided
        if task_filter:
            filtered_samples = [
                (task_key, sample)
                for task_key, sample_list in samples.items()
                if task_key in task_filter
                for sample in sample_list
            ]
            logger.info(
                f"Filtering to {len(task_filter)} tasks, {len(filtered_samples)} total samples"
            )
        else:
            filtered_samples = [
                (task_key, sample)
                for task_key, sample_list in samples.items()
                for sample in sample_list
            ]

        taskwise_scores = {}
        taskwise_scores_raw = {}

        prefix = "mcq_" if is_mcq else ""
        pbar_desc = "Evaluating (MCQ)" if is_mcq else "Evaluating (Gen)"
        for sample_key, sample in tqdm(filtered_samples, desc=pbar_desc, unit="sample"):
            if not isinstance(sample, dict):
                continue

            responses = sample.get("filtered_resps", [])
            if not responses:
                continue

            response = responses[0]
            expected_output = sample.get("doc", {}).get("output", "")
            question = sample.get("doc", {}).get("input", "")

            # For MCQ tasks, normalize answers to text BEFORE sending to judge
            # This way the judge just compares text to text (e.g., "لا" vs "لا" instead of "B" vs "لا")
            mcq_options = sample.get("doc", {}).get("mcq", [])
            if is_mcq and mcq_options:
                # Create letter-to-text mapping
                mcq_mapping = {chr(65 + i): opt for i, opt in enumerate(mcq_options)}

                # Normalize both the model's response and the expected output
                original_response = response
                original_expected = expected_output

                response = self._normalize_mcq_answer(response, mcq_mapping)
                expected_output = self._normalize_mcq_answer(
                    expected_output, mcq_mapping
                )

                logger.debug(f"MCQ Normalization:")
                logger.debug(f"  Response: '{original_response}' → '{response}'")
                logger.debug(f"  Expected: '{original_expected}' → '{expected_output}'")

                # Now both are text, judge can simply compare them
                # No need to pass full prompt with options anymore!

            if expected_output and response:
                try:
                    evaluation_result = llm_judge.evaluate_answer(
                        question=question,
                        reference_answer=expected_output,
                        given_answer=response,
                        context=None,
                        test_id=f"{sample_key}_{len(all_scores)}",
                        metadata={"task": sample_key},
                    )

                    normalized_score = evaluation_result["overall_score"]
                    raw_score = evaluation_result["overall_raw_score"]
                    explanation = evaluation_result["aggregated_explanation"]

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

                except Exception as e:
                    sample[f"{prefix}llm_score"] = None
                    sample[f"{prefix}llm_score_raw"] = None
                    sample[f"{prefix}llm_explanation"] = (
                        f"Error during LLM evaluation: {str(e)}"
                    )
                    sample[f"{prefix}llm_judge_details"] = {"error": str(e)}

        for task_key in taskwise_scores:
            if taskwise_scores[task_key]:
                avg_score = round(mean(taskwise_scores[task_key]), 4)
                avg_score_raw = round(mean(taskwise_scores_raw[task_key]), 4)

                processed_data["results"].setdefault(task_key, {"alias": task_key})

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
                    "num_samples": len(taskwise_scores[task_key]),
                }

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
