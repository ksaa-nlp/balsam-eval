"""Result processing and export functionality."""

import json
import logging
import os
from pathlib import Path
from statistics import mean
from typing import Any, Dict

from src.db_operations import add_results_to_db
from src.core.helpers import normalize_string

logger = logging.getLogger(__name__)


class ResultProcessor:
    """Handles result processing and export operations."""

    def __init__(
        self,
        category_name: str,
        api_host: str | None = None,
        job_id: str | None = None,
        server_token: str | None = None,
        benchmark_id: str | None = None,
        task_id: str | None = None,
    ):
        """Initialize result processor.

        Args:
            category_name: Name of the category
            api_host: API host URL
            job_id: Job ID for database updates
            server_token: Server token for authentication
            benchmark_id: Benchmark ID
            task_id: Task ID (optional)
        """
        self.category_name = category_name
        self.api_host = api_host
        self.job_id = job_id
        self.server_token = server_token
        self.benchmark_id = benchmark_id
        self.task_id = task_id

    def calculate_average_scores(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate average scores across all tasks.

        Args:
            results: Evaluation results dictionary

        Returns:
            Dictionary of metric name to average score
        """
        average_scores: dict[str, float] = {}
        all_scores: dict[str, list[float]] = {}

        if "results" not in results:
            logger.warning("No 'results' key in results for average calculation")
            return average_scores

        logger.info(f"Calculating averages for {len(results['results'])} tasks")

        for task_name, task_result in results["results"].items():
            if not isinstance(task_result, dict):
                logger.warning(f"Task '{task_name}' has non-dict result, skipping")
                continue

            for key, value in task_result.items():
                # Only process metric keys ending with ",none"
                if not key.endswith(",none"):
                    continue

                metric_name = key.replace(",none", "")

                # Handle dictionary values (ROUGE and potentially others)
                if isinstance(value, dict):
                    # For ROUGE, only use rougeLsum as the representative score
                    if "rougeLsum" in value:
                        if metric_name not in all_scores:
                            all_scores[metric_name] = []
                        all_scores[metric_name].append(value["rougeLsum"])
                        logger.debug(
                            f"Task '{task_name}': Added {metric_name}.rougeLsum={value['rougeLsum']}"
                        )
                    else:
                        logger.warning(
                            f"Task '{task_name}': Dict value for '{key}' but rougeLsum not found. Keys: {list(value.keys())}"
                        )

                # Handle numeric values (BLEU, accuracy, LLM judge scores, etc.)
                elif isinstance(value, (int, float)):
                    if metric_name not in all_scores:
                        all_scores[metric_name] = []
                    all_scores[metric_name].append(float(value))
                    logger.debug(f"Task '{task_name}': Added {metric_name}={value}")

                else:
                    logger.warning(
                        f"Task '{task_name}': Unexpected type for '{key}': {type(value)}"
                    )

        # Calculate averages
        for metric_name, scores in all_scores.items():
            if scores:
                avg = mean(scores)
                average_scores[metric_name] = round(avg, 4)
                logger.info(
                    f"Average {metric_name}: {average_scores[metric_name]} (from {len(scores)} tasks, min={min(scores):.4f}, max={max(scores):.4f})"
                )
            else:
                logger.warning(f"No scores found for metric '{metric_name}'")

        logger.info(
            f"Calculated {len(average_scores)} average metrics: {list(average_scores.keys())}"
        )
        return average_scores

    def export_results(self, results: Dict[str, Any]) -> None:
        """Export results to file and optionally to database.

        Args:
            results: Evaluation results to export
        """
        average_scores = self.calculate_average_scores(results)
        results_with_averages = {**results, "average_scores": average_scores}

        # Export to file
        filename = (
            f"{normalize_string(self.task_id)}.json"
            if self.task_id
            else f"{normalize_string(self.category_name)}.json"
        )

        filepath = (
            os.path.join(".results", filename)
            if not self.task_id
            else filename
        )

        with open(filepath, "w", encoding="UTF-8") as fp:
            json.dump(results_with_averages, fp, ensure_ascii=False)

        logger.info(f"Results exported to {filename}")

        # Export to database if configured
        if (
            self.api_host
            and self.job_id
            and self.server_token
            and self.benchmark_id
            and "results" in results_with_averages
        ):
            for key, value in results_with_averages["results"].items():
                task_id = (
                    normalize_string(self.task_id)
                    if self.task_id
                    else value.get("task")
                )

                if not task_id:
                    logger.warning(f"No task_id found for task '{key}'")
                    continue

                add_results_to_db(
                    api_host=self.api_host,
                    job_id=self.job_id,
                    task_id=task_id,
                    server_token=self.server_token,
                    result=value,
                    category_name=self.category_name,
                    benchmark_id=self.benchmark_id,
                )
