"""Evaluation job orchestration - refactored for cleaner code."""

import json
import logging
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, cast

import lmms_eval.evaluator
import lmms_eval.models  # Register all lmms_eval models
import requests

# Import custom metrics package to auto-register all metrics
import src.metrics  # Registers all metrics in src.metrics.impl.*  # pylint: disable=unused-import

from src.adapter_utils import get_max_tokens_config
from src.db_operations import JobStatus, update_status
from src.processors.llm_judge_operations import LLMJudgeProcessor
from src.processors.result_processing import ResultProcessor
from src.processors.task_operations import TaskOperations

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if os.environ.get("ENV", "PROD") == "local":
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Compatibility patches
_original_relative_to = Path.relative_to


def _safe_relative_to(self, *args, **kwargs):
    """Safely handle Path.relative_to with ValueError."""
    try:
        return _original_relative_to(self, *args, **kwargs)
    except ValueError as e:
        if "not in the subpath of" in str(e):
            return self
        raise e


Path.relative_to = _safe_relative_to  # type: ignore[method-assign]

# Increase default timeout for requests.post
requests.post = lambda url, timeout=5000, **kwargs: requests.request(
    method="POST", url=url, timeout=timeout, **kwargs
)


class EvaluationJob:
    """Orchestrates evaluation jobs with cleaner separation of concerns."""

    def __init__(
        self,
        tasks: List[str],
        model_args: dict[str, Any],
        category_name: str,
        *,
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
        """Initialize evaluation job.

        Args:
            tasks: List of task names to evaluate
            model_args: Model configuration arguments
            category_name: Name of the category
            tasks_mapper_dict: Optional dictionary mapping task names to task IDs
            adapter: Model adapter type
            job_id: Optional job ID for database updates
            api_host: Optional API host URL
            server_token: Optional server token for authentication
            benchmark_id: Optional benchmark ID
            llm_judge_model: Optional LLM judge model name
            llm_judge_provider: Optional LLM judge provider
            llm_judge_api_key: Optional LLM judge API key
            task_id: Optional explicit task ID
        """
        self.model_args = model_args or {}
        self.tasks: List[str] = tasks
        self.adapter = adapter
        self.job_id = job_id
        self.category_name = category_name
        self.benchmark_id = benchmark_id
        self.api_host = api_host
        self.server_token = server_token

        # Initialize helper classes
        self.task_ops = TaskOperations(tasks_mapper_dict, task_id)
        self.result_processor = ResultProcessor(
            category_name=category_name,
            api_host=api_host,
            job_id=job_id,
            server_token=server_token,
            benchmark_id=benchmark_id,
            task_id=task_id,
        )
        self.llm_judge_processor = LLMJudgeProcessor(
            task_operations=self.task_ops,
            llm_judge_api_key=llm_judge_api_key,
            llm_judge_model=llm_judge_model,
            llm_judge_provider=llm_judge_provider,
        )

        # Setup model args
        if "eos_string" not in self.model_args:
            self.model_args["eos_string"] = ""
            logger.info("Added default eos_string='' to model_args")

        # Set API key environment variables
        api_key = os.getenv("API_KEY")
        if api_key and self.adapter in ["openai-chat-completions", "local-chat-completions"]:
            os.environ["OPENAI_API_KEY"] = api_key
        if api_key and self.adapter == "anthropic-chat-completions":
            os.environ["ANTHROPIC_API_KEY"] = api_key
        if api_key and self.adapter == "gemini":
            os.environ["GOOGLE_API_KEY"] = api_key

    def __call__(self) -> None:
        """Run the evaluation job."""
        # Update status to running
        if self.api_host and self.job_id and self.server_token:
            update_status(
                api_host=self.api_host,
                job_id=self.job_id,
                server_token=self.server_token,
                status=JobStatus.RUNNING,
            )

        try:
            self._run_evaluation()
        except Exception as e:  # pylint: disable=broad-exception-caught
            self._handle_error(e)

    def _run_evaluation(self) -> None:
        """Execute the evaluation workflow."""
        logger.info("=" * 80)
        logger.info("Starting evaluation job for category: %s", self.category_name)
        logger.info("Tasks: %s", self.tasks)
        logger.info("Adapter: %s", self.adapter)
        logger.info("Model args: %s", self.model_args)
        logger.info("=" * 80)

        # Run lmms_eval evaluation
        results = self._run_lmms_eval()

        # Add task information to results
        results = self.task_ops.add_task_to_results(results)

        if not results:
            logger.warning("No results found for the evaluation job.")
            self._update_status(JobStatus.FAILED, "No results found for the evaluation job.")
            return

        # Separate MCQ and generation tasks
        mcq_tasks, generation_tasks = self.task_ops.separate_mcq_and_generation_tasks(results)

        # Process with LLM judge if configured
        updated_results = results
        updated_results = self.llm_judge_processor.process_generation_tasks(
            updated_results, generation_tasks
        )
        updated_results = self.llm_judge_processor.process_mcq_tasks(
            updated_results, mcq_tasks
        )

        # Remove API key from results
        self._sanitize_results(updated_results)

        # Export results
        self.result_processor.export_results(updated_results)

        # Update status to completed
        if self.api_host and self.job_id and self.server_token:
            update_status(
                api_host=self.api_host,
                job_id=self.job_id,
                server_token=self.server_token,
                status=JobStatus.COMPLETED,
            )

        logger.info("✅ Evaluation job completed successfully")

    def _run_lmms_eval(self) -> Dict[str, Any]:
        """Run the lmms_eval evaluation.

        Returns:
            Evaluation results dictionary
        """
        temp_dir = Path(".temp").resolve()

        logger.info("Calling lmms_eval.evaluator.simple_evaluate...")

        # Convert gen_kwargs dict to string format for lmms_eval
        gen_kwargs_dict = get_max_tokens_config(self.adapter, self.model_args["model"])
        gen_kwargs_str = ",".join(f"{k}={v}" for k, v in gen_kwargs_dict.items())

        # Convert model_args dict to string format for lmms_eval
        model_args_str = ",".join(f"{k}={v}" for k, v in self.model_args.items())

        # Map old adapter names to new lmms_eval model names
        model_mapping = {
            "openai-chat-completions": "openai",
            "local-chat-completions": "openai_compatible",
            "gemini": "gemini_api",
            "anthropic-chat-completions": "claude",
            "groq": "openai_compatible",
        }
        model = model_mapping.get(self.adapter, self.adapter)

        results = cast(
            dict[str, Any],
            lmms_eval.evaluator.simple_evaluate(
                model=model,
                model_args=model_args_str,
                tasks=self.tasks,
                apply_chat_template=False,
                task_manager=lmms_eval.tasks.TaskManager(
                    include_path=str(temp_dir), include_defaults=False
                ),
                batch_size=1,
                gen_kwargs=gen_kwargs_str,
            ),
        )

        logger.info("✅ simple_evaluate completed successfully")
        logger.info("Exporting results to %s.json", self.category_name)

        return results

    def _sanitize_results(self, results: Dict[str, Any]) -> None:
        """Remove sensitive information from results.

        Args:
            results: Results dictionary to sanitize (modified in-place)
        """
        if (
            "config" in results
            and "model_args" in results["config"]
            and isinstance(results["config"]["model_args"], dict)
            and "api_key" in results["config"]["model_args"]
        ):
            results["config"]["model_args"].pop("api_key")

    def _update_status(
        self, status: JobStatus, error_message: Optional[str] = None
    ) -> None:
        """Update job status.

        Args:
            status: New job status
            error_message: Optional error message
        """
        if self.api_host and self.job_id and self.server_token:
            update_status(
                api_host=self.api_host,
                job_id=self.job_id,
                server_token=self.server_token,
                status=status,
                error_message=error_message,
            )

    def _handle_error(self, exception: Exception) -> None:
        """Handle evaluation error.

        Args:
            exception: The exception that occurred
        """
        logger.error("❌ Error in evaluation job: %s", exception)
        logger.error(traceback.format_exc())

        api_error_data = self._extract_api_error_message(exception)

        if self.api_host and self.job_id and self.server_token:
            error_message = (
                json.dumps(api_error_data)
                if isinstance(api_error_data, dict)
                else str(api_error_data)
            )
            self._update_status(JobStatus.FAILED, error_message)

        error_message = (
            json.dumps(api_error_data)
            if isinstance(api_error_data, dict)
            else str(api_error_data)
        )
        raise RuntimeError(error_message) from exception

    def _extract_api_error_message(self, exception: Exception) -> Dict[str, Any]:
        """Extract the full API error object from various exception types.

        Args:
            exception: The exception to extract error from

        Returns:
            Dictionary containing error information
        """
        try:
            # Try to get error from response
            if hasattr(exception, "response") and exception.response is not None:
                try:
                    error_data = cast(dict[str, Any], exception.response.json())
                    if "error" in error_data:
                        return error_data
                except (json.JSONDecodeError, ValueError):
                    pass

            error_str = str(exception)

            # Try to parse JSON from error string
            if "API request failed with error message:" in error_str:
                import re

                json_match = re.search(r"\{(?:[^{}]|{[^{}]*})*\}", error_str)
                if json_match:
                    try:
                        error_json = cast(dict[str, Any], json.loads(json_match.group()))
                        if "error" in error_json:
                            return error_json
                    except json.JSONDecodeError:
                        pass

            # Check exception args
            if hasattr(exception, "args") and exception.args:
                for arg in exception.args:
                    if isinstance(arg, dict) and "error" in arg:
                        return cast(dict[str, Any], arg)
                    if isinstance(arg, str):
                        try:
                            parsed = cast(dict[str, Any], json.loads(arg))
                            if "error" in parsed:
                                return parsed
                        except json.JSONDecodeError:
                            pass

            # Fallback error object
            return {
                "error": {
                    "message": str(exception),
                    "type": type(exception).__name__,
                    "param": None,
                    "code": "unknown",
                }
            }

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Error while extracting API error message: %s", e)
            return {
                "error": {
                    "message": str(exception),
                    "type": "extraction_error",
                    "param": None,
                    "code": "unknown",
                }
            }
