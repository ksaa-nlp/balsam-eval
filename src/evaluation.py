"""This module contains the code for running an evaluation job."""

import json
import logging
import os
from enum import Enum
import traceback
from typing import Any, List, Optional

import lm_eval  # type: ignore
import requests  # type: ignore

# This import is necessary for the aixplain adapter to work.
from . import aixplain

# This import is necessary for the rouge metric to work.
from . import metric

API_HOST = os.getenv("API_HOST", "none")
SERVER_TOKEN = os.getenv("SERVER_TOKEN", "none")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Adapter(str, Enum):
    """The adapter to use for the evaluation job."""

    AIXPLAIN = "aixplain"
    # OPENAI = "openai-completions"
    OPENAI_CHAT = "openai-chat-completions"
    # LOCAL_COMPLETIONS = "local-completions"
    LOCAL_CHAT_COMPLETIONS = "local-chat-completions"
    GGUF = "gguf"

    @staticmethod
    def from_str(model: str) -> "Adapter":
        if model == "aixplain":
            return Adapter.AIXPLAIN
        elif model == "openai-chat-completions":
            return Adapter.OPENAI_CHAT
        # elif model == "openai-completions":
        #     return Adapter.OPENAI
        # elif model == "local-completions":
        #     return Adapter.LOCAL_COMPLETIONS
        elif model == "local-chat-completions":
            return Adapter.LOCAL_CHAT_COMPLETIONS
        elif model == "gguf":
            return Adapter.GGUF
        raise ValueError(f"Model {model} is not supported")


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
        base_url (`str`): The base URL of the model.
        tasks (`List[str]`): The tasks to evaluate, these names correspond
        to YAML files that should exist in the current working directory.
        api_key (`str`): The API key to use for calling the model endpoint.
    """

    def __init__(
        self,
        tasks: List[str],
        adapter: Adapter,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        output_path: Optional[str] = None,
    ):
        self.model_args = {}

        if adapter == Adapter.AIXPLAIN:
            self.model_args = {"model": model}
            os.environ["TEAM_API_KEY"] = api_key

        elif adapter in (
            Adapter.OPENAI_CHAT,
            Adapter.LOCAL_CHAT_COMPLETIONS,
        ):
            if base_url is None or model is None:
                raise ValueError(
                    "The base URL must be provided for the OpenAI adapter."
                )
            # Check if the base URL ends with a slash
            if base_url[-1] == "/":
                base_url = base_url[:-1]

            elif adapter in (Adapter.OPENAI_CHAT, Adapter.LOCAL_CHAT_COMPLETIONS):
                self.model_args = {
                    "base_url": base_url + "/v1/chat/completions",
                    "model": model,
                }

            os.environ["OPENAI_API_KEY"] = api_key

        elif adapter == Adapter.GGUF:
            if base_url is None or model is None:
                raise ValueError("The base URL must be provided for the GGUF adapter.")
            self.model_args = {
                "base_url": base_url,
                "model": model,
            }
            os.environ["OPENAI_API_KEY"] = api_key

        self.tasks = tasks
        self.adapter = adapter
        self.job_id = os.getenv("JOB_ID")
        self.output_path = output_path or "results"

    def run(self):
        """Run a simple evaluation job."""
        self._update_status(JobStatus.RUNNING)
        print(self.model_args)
        try:
            results = lm_eval.simple_evaluate(
                model=self.adapter,
                model_args=self.model_args,
                tasks=self.tasks,
                verbosity="ERROR",
                apply_chat_template=(
                    True
                    if self.adapter
                    in (Adapter.OPENAI_CHAT, Adapter.LOCAL_CHAT_COMPLETIONS)
                    else False
                ),
                # apply_chat_template=True,
                # We set the include path to the current directory,
                # so that the script can find the data files.
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

    def _calculate_average_score(self, results: dict[str, Any]) -> float:
        """Calculate the average score of the model."""
        # Get the metrics for each task
        total_score = 0
        total_tasks = 0
        for task in results["configs"]:
            # Extract the metrics in the task.
            metrics = [
                metric["metric"] for metric in results["configs"][task]["metric_list"]
            ]
            logger.info("Metrics found for task %s: %s", task, metrics)
            for m in metrics:
                total_score += results["results"][task][f"{m},none"]
            total_tasks += 1
        return total_score / total_tasks

    def _export_results(self, results: dict[str, Any]):
        """Export the results to a JSON file in the current working directory."""
        self._add_results_to_db(results)
        with open(f"{self.output_path}.json", "w", encoding="UTF-8") as fp:
            json.dump(results, fp, ensure_ascii=False)

        logger.info(os.path.abspath(f"{self.output_path}.json"))

    def _add_results_to_db(self, results: dict[str, Any]):
        """Calls a webhook to add the results to the database."""
        logger.info("Adding the results to the database")
        # Request to webhook to add the results to the database
        webhook_url = f"https://{API_HOST}/api/webhook/job"
        logger.info("Sending request to %s", webhook_url)
        average = self._calculate_average_score(results)
        logger.info("Average score for job with ID %s: %s", self.job_id, average)

        if API_HOST == "none":
            logger.warning("API_HOST is not set, skipping status update")
            return

        response = requests.post(
            webhook_url,
            json={
                "id": self.job_id,
                "results": json.dumps(results, ensure_ascii=False),
                "status": JobStatus.COMPLETED.value,
                "average": average,
            },
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
        if API_HOST == "none":
            logger.warning("API_HOST is not set, skipping status update")
            return

        logger.info("Updating the status of the job to %s", status.value)
        # Request to webhook to update the status of the job
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
