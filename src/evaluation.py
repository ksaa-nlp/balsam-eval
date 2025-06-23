"""This module contains the code for running an evaluation job."""
# from . import gemini_adapter

from statistics import mean
from tqdm import tqdm
from typing import Dict, Any, List
import json
import logging
import os
from enum import Enum
import re
import traceback
from typing import Any, List, Literal, Optional
import lm_eval
import requests

# flake8: noqa
from src.gemini_adapter import GeminiLM
from src.llm_as_a_judge import LLMJudge, ModelConfig

# This import is necessary for the aixplain adapter to work.
from . import aixplain

# This import is necessary for the rouge metric to work.
from . import metric
from dotenv import load_dotenv

load_dotenv()

API_HOST = os.getenv("API_HOST")
SERVER_TOKEN = os.getenv("SERVER_TOKEN", "none")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# More robust request handling with explicit timeout
requests.post = lambda url, timeout=5000, **kwargs: requests.request(
    method="POST", url=url, timeout=timeout, **kwargs
)


class JobStatus(Enum):
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


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
        adapter: Literal[
            "local-completions",
            "local-chat-completions",
            "openai-completions",
            "openai-chat-completions",
            "anthropic-chat",
            "anthropic-chat-completions",
            "anthropic-completions",
            "dummy",
            "gguf",
            "ggml",
            "hf-audiolm-qwen",
            "hf-multimodal",
            "hf-auto",
            "steered",
            "hf",
            "huggingface",
            "watsonx_llm",
            "mamba_ssm",
            "nemo_lm",
            "sparseml",
            "deepsparse",
            "neuronx",
            "ipex",
            "openvino",
            "sglang",
            "textsynth",
            "vllm",
            "vllm-vlm"
        ] = "local-chat-completions",
        output_path: Optional[str] = None,
    ):
        self.model_args = model_args or {}
        self.tasks: List[str] = tasks
        self.task_id = task_id
        self.adapter = adapter
        self.job_id = os.getenv("JOB_ID")

        sanitized_path = re.sub(
            r"\s", "_", (output_path or "results").lower()).replace(".", "_")
        self.output_path = sanitized_path

    def run(self):
        """Run a simple evaluation job."""
        self._update_status(JobStatus.RUNNING)
        try:
            results = lm_eval.simple_evaluate(
                model=self.adapter,
                model_args=self.model_args,
                tasks=self.tasks,
                apply_chat_template=True,
                # apply_chat_template=False,
                task_manager=lm_eval.tasks.TaskManager(include_path=".temp"),
            )
            logger.info("Exporting results to %s.json", self.output_path)

            # Configure your LLM judge
            model_configs = [
                ModelConfig(
                    name=os.getenv("JUDGE_MODEL", "gemini"),
                    provider="gemini",
                    api_key=os.getenv("JUDGE_API_KEY")
                )
            ]

            # Initialize LLMJudge
            llm_judge = LLMJudge(
                model_configs=model_configs,
                aggregation_method="mean",
                threshold=0.5
            )

            updated_results = self.process_results_with_llm_judge(
                results_data=results,
                llm_judge=llm_judge,
                show_progress=True
            )

            # Export the results to a file, and add them to the database
            self._export_results(updated_results)

        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error("An error occurred while running the job: %s", e)
            self._update_status(JobStatus.FAILED, str(e))
            raise e

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
                                full_metric_name = f"{metric_name}_{sub_metric}"
                                if full_metric_name not in total_scores:
                                    total_scores[full_metric_name] = 0.0
                                    task_counts[full_metric_name] = 0
                                total_scores[full_metric_name] += sub_value
                                task_counts[full_metric_name] += 1
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
        self.output_path = self.output_path.replace("/", "-")
        if is_temp:
            self.output_path = f"sss/{self.output_path}"
        # Save results to a JSON file
        with open(f"{self.output_path}.json", "w", encoding="UTF-8") as fp:
            json.dump(results_with_averages, fp, ensure_ascii=False)

        logger.info(
            "Results exported to %s", os.path.abspath(
                f"{self.output_path}.json")
        )

        # Add results to the database
        self._add_results_to_db(results)

    def _add_results_to_db(self, results: dict[str, Any]):
        """Calls a webhook to add the results to the database."""
        logger.info("Adding the results to the database")

        # Calculate average scores
        average_scores = self._calculate_average_scores(
            results)  # Returns a dictionary
        logger.info(
            "Average scores for job with ID %s: %s", self.job_id, average_scores
        )

        if not API_HOST:
            logger.warning("API_HOST is not set, skipping status update")
            return

        # Prepare payload
        payload = {
            "id": self.job_id,
            "taskId": self.task_id,
            "results": json.dumps(results, ensure_ascii=False),
            "status": JobStatus.COMPLETED.value,
            "average": average_scores,  # Pass the dictionary
        }

        # Send request
        webhook_url = f"https://{API_HOST}/api/webhook/job"
        logger.info("Sending request to %s", webhook_url)
        response = requests.post(
            webhook_url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "x-server-token": SERVER_TOKEN,
            },
            timeout=20,
        )
        response.raise_for_status()
        logger.info("Results added to the database successfully")

    def _update_status(self, status: JobStatus, error_message: Optional[str] = None):
        """Update the status of the job."""
        if not API_HOST:
            logger.warning("API_HOST is not set, skipping status update")
            return

        logger.info("Updating the status of the job to %s", status.value)
        webhook_url = f"https://{API_HOST}/api/webhook/job"
        logger.info("Sending request to %s", webhook_url)
        response = requests.post(
            webhook_url,
            json={
                "id": self.job_id,
                "status": status.value,
                "error": error_message or "",
            },
            headers={
                "Content-Type": "application/json",
                "x-server-token": SERVER_TOKEN,
            },
            timeout=20,
        )
        response.raise_for_status()
        logger.info("Job status updated successfully")

    def process_results_with_llm_judge(
        self,
        results_data: Dict[str, Any],
        llm_judge: 'LLMJudge',
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Process results dictionary using LLMJudge to evaluate and score responses.

        Args:
            results_data: Dictionary containing samples and results in the same format as the JSON files
            llm_judge: Initialized LLMJudge instance
            show_progress: Whether to show progress bar during evaluation

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

        # Setup progress bar
        iterator = tqdm(all_samples, desc="Evaluating with LLMJudge",
                        unit="question") if show_progress else all_samples

        for sample_key, sample in iterator:
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
