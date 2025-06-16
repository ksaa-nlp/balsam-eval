"""This module contains the code for running an evaluation job."""
# from . import gemini_adapter

import json
import logging
import os
from enum import Enum
import traceback
from typing import Any, List, Literal, Optional
import lm_eval
import requests

# flake8: noqa
from src.gemini_adapter import GeminiLM

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
        self.output_path = output_path or "results"

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

            # Export the results to a file, and add them to the database
            self._export_results(results)

        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error("An error occurred while running the job: %s", e)
            self._update_status(JobStatus.FAILED, str(e))
            raise e

    def _calculate_average_scores(self, results: dict[str, Any]) -> dict[str, float]:
        """Calculate the average scores of the model for ROUGE or other metrics."""
        total_scores = {}
        total_tasks = 0

        for task in results.get("configs", {}):
            # Extract metrics for the current task
            metrics = [
                metric["metric"]
                for metric in results["configs"][task].get("metric_list", [])
            ]
            logger.info("Metrics found for task %s: %s", task, metrics)

            for m in metrics:
                logger.info("Inspecting metric: %s", m)
                if m == "rouge":
                    # Initialize ROUGE metrics if not already initialized
                    if not total_scores:
                        total_scores = {
                            "rouge1": 0.0,
                            "rouge2": 0.0,
                            "rougeL": 0.0,
                            "rougeLsum": 0.0,
                        }
                    try:
                        # Aggregate ROUGE scores
                        logger.info("for the total score intialization")
                        logger.info(
                            "Available keys for task '%s': %s",
                            task,
                            list(results["results"][task].keys()),
                        )
                        total_scores["rouge1"] += results["results"][task][
                            "rouge,none"
                        ]["rouge1"]
                        total_scores["rouge2"] += results["results"][task][
                            "rouge,none"
                        ]["rouge2"]
                        total_scores["rougeL"] += results["results"][task][
                            "rouge,none"
                        ]["rougeL"]
                        total_scores["rougeLsum"] += results["results"][task][
                            "rouge,none"
                        ]["rougeLsum"]
                    except KeyError as e:
                        logger.warning(
                            "ROUGE score '%s' not found in task '%s': %s", m, task, e
                        )
                else:
                    # Handle other metrics
                    if m not in total_scores:
                        total_scores[m] = 0.0
                    try:
                        total_scores[m] += results["results"][task][f"{m},none"]
                    except KeyError as e:
                        logger.warning(
                            "Metric '%s' not found in task '%s': %s", m, task, e
                        )

                total_tasks += 1

        if total_tasks == 0:
            logger.error("No tasks found to calculate scores")
            return {key: 0.0 for key in total_scores}

        # Calculate averages for all scores
        average_scores = {
            key: total_scores[key] / total_tasks for key in total_scores}

        logger.info("Average Scores: %s", average_scores)
        return average_scores

    def _export_results(self, results: dict[str, Any]):
        """Export the results to a JSON file in the current working directory."""
        # Add average scores to results for export
        average_scores = self._calculate_average_scores(results)
        results_with_averages = {**results, "average_scores": average_scores}
        self.output_path = self.output_path.replace("/", "-")
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
