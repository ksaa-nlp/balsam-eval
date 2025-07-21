"""This module contains the code for running an evaluation job."""
import os
from src.llm_as_a_judge import LLMJudge, ModelConfig
from src.db_operations import JobStatus, add_results_to_db, update_status
from src.utils import normalize_string
import requests
import lm_eval
from typing import Any, List, Literal, Optional, Dict
import traceback
import logging
import json
from statistics import mean

# This import is necessary for the rouge metric to work and for gemini adapter to be available
from . import metric  # noqa: F401
from src.gemini_adapter import GeminiLM  # noqa: F401

# Download necessary resources
import nltk
nltk.download('punkt_tab')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if os.environ.get("ENV", "PROD") == "local":
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# More robust request handling with explicit timeout
requests.post = lambda url, timeout=5000, **kwargs: requests.request(
    method="POST", url=url, timeout=timeout, **kwargs
)


class EvaluatationJob:
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
        task_id: str,
        model_args: dict[str, Any],
        category_name: str,
        benchmark_id: str,
        adapter: Literal[
            "local-chat-completions",
            "openai-chat-completions",
            "anthropic-chat-completions",
            "gemini"
        ] = "local-chat-completions",
        output_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        job_id: Optional[str] = None,
        api_host: Optional[str] = None,
        server_token: Optional[str] = None,
        llm_judge_model: Optional[str] = None,
        llm_judge_provider: Optional[str] = None,
        llm_judge_api_key: Optional[str] = None,
    ):
        self.model_args = model_args or {}
        self.tasks: List[str] = tasks
        self.task_id = task_id
        self.adapter = adapter
        self.job_id = job_id
        self.category_name = category_name
        self.benchmark_id = benchmark_id
        self.job_id = job_id
        self.api_host = api_host
        self.server_token = server_token
        self.llm_judge_model = llm_judge_model
        self.llm_judge_provider = llm_judge_provider
        self.llm_judge_api_key = llm_judge_api_key

        if output_dir:
            self.output_path = os.path.join(
                output_dir, normalize_string(output_path or "results"))
        else:
            self.output_path = normalize_string(output_path or "results")

    def run(self):
        """Run a simple evaluation job."""
        if self.api_host and self.job_id and self.server_token:
            update_status(api_host=self.api_host, job_id=self.job_id,
                          server_token=self.server_token, status=JobStatus.RUNNING)
        try:
            results = lm_eval.simple_evaluate(
                model=self.adapter,
                model_args=self.model_args,
                tasks=self.tasks,
                apply_chat_template=True,
                task_manager=lm_eval.tasks.TaskManager(include_path=".temp"),
            )
            logger.info("Exporting results to %s.json", self.output_path)

            results = self._add_task_to_results(
                results=results, task_id=self.task_id)

            llm_judge = None
            if self.llm_judge_api_key and self.llm_judge_model and self.llm_judge_provider:
                # Initialize LLMJudge
                llm_judge = LLMJudge(
                    model_configs=[
                        ModelConfig(
                            name=self.llm_judge_model,
                            provider=self.llm_judge_provider,
                            api_key=self.llm_judge_api_key
                        )
                    ],
                    aggregation_method="mean",
                    threshold=0.5
                )

            if not results:
                logger.warning("No results found for the evaluation job.")
                if self.api_host and self.job_id and self.server_token:
                    update_status(api_host=self.api_host, job_id=self.job_id,
                                  server_token=self.server_token, status=JobStatus.FAILED,
                                  error_message="No results found for the evaluation job.")
                return

            if llm_judge and isinstance(llm_judge, LLMJudge):
                logger.info("Processing results with LLM judge...")
                updated_results = self.process_results_with_llm_judge(
                    results_data=results,
                    llm_judge=llm_judge)
            else:
                logger.info("Skipping LLM judge processing.")
                updated_results = results

            # Export the results to a file, and add them to the database
            self._export_results(updated_results)

        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error("An error occurred while running the job: %s", e)
            if self.api_host and self.job_id and self.server_token:
                update_status(api_host=self.api_host, job_id=self.job_id,
                              server_token=self.server_token, status=JobStatus.FAILED, error_message=str(e))
            raise e

    def _add_task_to_results(self, results: dict[str, Any], task_id: str):
        """Add task_id to each result in the results dictionary."""
        if "results" not in results:
            logger.warning("No 'results' key found in results dictionary")
            return results

        for task_name in results["results"]:
            if isinstance(results["results"][task_name], dict):
                results["results"][task_name]["task"] = task_id
            else:
                logger.warning(
                    f"Task result for '{task_name}' is not a dictionary, skipping task_id addition")

        return results

    def _calculate_average_scores(self, results: dict[str, Any]) -> dict[str, float]:
        """Calculate the average scores of the model for ROUGE or other metrics."""
        total_scores = {}
        task_counts = {}  # Track how many tasks contribute to each metric

        for task in results.get("results", {}):
            task_results = results["results"][task]

            # Handle different metric types
            for key, value in task_results.items():
                if key.endswith(",none") and not key.endswith("_stderr,none"):
                    metric_name = key.replace(",none", "")

                    if isinstance(value, dict):
                        # Handle nested metrics like ROUGE
                        for sub_metric, sub_value in value.items():
                            if isinstance(sub_value, (int, float)):
                                if sub_metric not in total_scores:
                                    total_scores[sub_metric] = 0.0
                                    task_counts[sub_metric] = 0
                                total_scores[sub_metric] += sub_value
                                task_counts[sub_metric] += 1
                    elif isinstance(value, (int, float)):
                        # Handle simple metrics like accuracy
                        if metric_name not in total_scores:
                            total_scores[metric_name] = 0.0
                            task_counts[metric_name] = 0
                        total_scores[metric_name] += value
                        task_counts[metric_name] += 1

        # Calculate averages for all metrics
        average_scores = {}
        for metric_name, total_score in total_scores.items():
            count = task_counts[metric_name]
            if count > 0:
                average_scores[metric_name] = total_score / count
            else:
                average_scores[metric_name] = 0.0

        logger.info("Average Scores: %s", average_scores)
        return average_scores

    def _export_results(self, results: dict[str, Any], is_temp: bool = False):
        """Export the results to a JSON file in the current working directory."""
        # Add average scores to results for export
        average_scores = self._calculate_average_scores(results)
        results_with_averages = {**results, "average_scores": average_scores}
        # Save results to a JSON file
        with open(f"{self.output_path}.json", "w", encoding="UTF-8") as fp:
            json.dump(results_with_averages, fp, ensure_ascii=False)

        logger.info(f"Results exported to {self.output_path}.json")

        # Add results to the database
        if self.api_host and self.job_id and self.server_token:
            if "results" in results_with_averages:
                for key, value in results_with_averages["results"].items():
                    add_results_to_db(
                        api_host=self.api_host,
                        job_id=self.job_id,
                        task_id=self.task_id,
                        server_token=self.server_token,
                        result=value,
                        category_name=self.category_name,
                        benchmark_id=self.benchmark_id)
            else:
                logger.error("No results found in the evaluation job.")

    def process_results_with_llm_judge(
        self,
        results_data: Dict[str, Any],
        llm_judge: 'LLMJudge',
    ) -> Dict[str, Any]:
        """
        Process results dictionary using LLMJudge to evaluate and score responses.

        Args:
            results_data: Dictionary containing samples and results in the same format as the JSON files
            llm_judge: Initialized LLMJudge instance

        Returns:
            Modified results_data dictionary with LLM judge scores added
        """
        # Make a copy to avoid modifying the original
        processed_data = results_data.copy()

        samples = processed_data.get("samples", {})
        processed_data.setdefault("results", {})

        all_scores = []
        all_scores_raw = []

        # Flatten all samples for processing
        all_samples = [
            (task_key, sample)
            for task_key, sample_list in samples.items()
            for sample in sample_list
        ]

        taskwise_scores = {}
        taskwise_scores_raw = {}

        for sample_key, sample in all_samples:
            if not isinstance(sample, dict):
                continue

            responses = sample.get("filtered_resps", [])
            if not responses:
                continue

            response = responses[0]
            expected_output = sample.get("doc", {}).get("output", "")
            question = sample.get("doc", {}).get("input", "")

            if expected_output and response:
                try:
                    # Use LLMJudge to evaluate the answer
                    evaluation_result = llm_judge.evaluate_answer(
                        question=question,
                        reference_answer=expected_output,
                        given_answer=response,
                        context=None,
                        test_id=f"{sample_key}_{len(all_scores)}",
                        metadata={"task": sample_key}
                    )

                    # Extract scores from the evaluation result
                    normalized_score = evaluation_result["overall_score"]
                    raw_score = evaluation_result["overall_raw_score"]
                    explanation = evaluation_result["aggregated_explanation"]

                    # Add scores to the sample
                    sample["llm_score"] = normalized_score
                    sample["llm_score_raw"] = raw_score
                    sample["llm_explanation"] = explanation
                    sample["llm_judge_details"] = {
                        "model_results": evaluation_result["model_results"],
                        "aggregation_method": evaluation_result["aggregation_method"],
                        "passed": evaluation_result["overall_passed"]
                    }

                    # Collect scores for aggregation
                    all_scores.append(normalized_score)
                    all_scores_raw.append(raw_score)
                    taskwise_scores.setdefault(
                        sample_key, []).append(normalized_score)
                    taskwise_scores_raw.setdefault(
                        sample_key, []).append(raw_score)

                except Exception as e:
                    # Handle evaluation errors
                    sample["llm_score"] = None
                    sample["llm_score_raw"] = None
                    sample["llm_explanation"] = f"Error during LLM evaluation: {str(e)}"
                    sample["llm_judge_details"] = {"error": str(e)}

        # Calculate task-wise averages
        for task_key in taskwise_scores:
            if taskwise_scores[task_key]:  # Check if list is not empty
                avg_score = round(mean(taskwise_scores[task_key]), 4)
                avg_score_raw = round(mean(taskwise_scores_raw[task_key]), 4)

                processed_data["results"].setdefault(
                    task_key, {"alias": task_key})
                processed_data["results"][task_key]["llm_as_judge"] = {
                    "average_score": avg_score,
                    "average_score_raw": avg_score_raw,
                    "num_samples": len(taskwise_scores[task_key])
                }

        # Calculate overall averages
        if all_scores:
            processed_data["overall_llm_as_judge"] = round(mean(all_scores), 4)
            processed_data["overall_llm_as_judge_stats"] = {
                "total_samples": len(all_scores),
                "average_score": round(mean(all_scores), 4),
                "min_score": min(all_scores),
                "max_score": max(all_scores)
            }

        if all_scores_raw:
            processed_data["overall_llm_as_judge_raw"] = round(
                mean(all_scores_raw), 4)
            processed_data["overall_llm_as_judge_raw_stats"] = {
                "total_samples": len(all_scores_raw),
                "average_score_raw": round(mean(all_scores_raw), 4),
                "min_score_raw": min(all_scores_raw),
                "max_score_raw": max(all_scores_raw)
            }

        return processed_data
