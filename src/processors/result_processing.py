"""Result processing and export functionality."""

import json
import logging
import os
from statistics import mean
from typing import Any, Dict

import numpy as np

from src.core.helpers import normalize_string
from src.db_operations import add_results_to_db


logger = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return super().default(o)

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

        logger.info("Calculating averages for %s tasks", len(results["results"]))

        for task_name, task_result in results["results"].items():
            if not isinstance(task_result, dict):
                logger.warning("Task '%s' has non-dict result, skipping", task_name)
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
                            "Task '%s': Added %s.rougeLsum=%s",
                            task_name,
                            metric_name,
                            value["rougeLsum"],
                        )
                    else:
                        logger.warning(
                            "Task '%s': Dict value for '%s' but rougeLsum not found. Keys: %s",
                            task_name,
                            key,
                            list(value.keys()),
                        )

                # Handle numeric values (BLEU, accuracy, LLM judge scores, etc.)
                elif isinstance(value, (int, float)):
                    if metric_name not in all_scores:
                        all_scores[metric_name] = []
                    all_scores[metric_name].append(float(value))
                    logger.debug("Task '%s': Added %s=%s", task_name, metric_name, value)

                else:
                    logger.warning(
                        "Task '%s': Unexpected type for '%s': %s",
                        task_name,
                        key,
                        type(value),
                    )

        # Calculate averages
        for metric_name, scores in all_scores.items():
            if scores:
                avg = mean(scores)
                average_scores[metric_name] = round(avg, 4)
                logger.info(
                    "Average %s: %s (from %s tasks, min=%.4f, max=%.4f)",
                    metric_name,
                    average_scores[metric_name],
                    len(scores),
                    min(scores),
                    max(scores),
                )
            else:
                logger.warning("No scores found for metric '%s'", metric_name)

        logger.info(
            "Calculated %s average metrics: %s",
            len(average_scores),
            list(average_scores.keys()),
        )
        return average_scores

    @staticmethod
    def _strip_audio_data(results: Dict[str, Any]) -> None:
        """Remove large audio array data from results to reduce file size."""
        samples = results.get("samples", {})
        for task_samples in samples.values():
            if not isinstance(task_samples, list):
                continue
            for sample in task_samples:
                for arg_tuple in sample.get("arguments", []):
                    if isinstance(arg_tuple, (list, tuple)) and len(arg_tuple) >= 3:
                        aux = arg_tuple[2]
                        if isinstance(aux, dict) and "audio" in aux:
                            del aux["audio"]
                doc = sample.get("doc")
                if isinstance(doc, dict) and "audio" in doc:
                    del doc["audio"]

    @staticmethod
    def _strip_multimodal_data(results: Dict[str, Any]) -> Dict[str, Any]:
        """Remove audio/image binary data from samples before serialization."""
        if "samples" not in results:
            return results

        original_samples = results["samples"]
        results = {**results, "samples": {}}
        for task_name, task_samples in original_samples.items():
            cleaned = []
            for sample in task_samples:
                sample = {**sample}
                if "arguments" in sample and isinstance(sample["arguments"], list):
                    new_args = []
                    for arg_group in sample["arguments"]:
                        if isinstance(arg_group, (list, tuple)):
                            new_group = []
                            for item in arg_group:
                                if isinstance(item, dict):
                                    item = {
                                        k: v for k, v in item.items()
                                        if k not in ("audio", "images", "visuals")
                                    }
                                new_group.append(item)
                            new_args.append(new_group)
                        else:
                            new_args.append(arg_group)
                    sample["arguments"] = new_args
                cleaned.append(sample)
            results["samples"][task_name] = cleaned

        return results

    def export_results(self, results: Dict[str, Any]) -> None:
        """Export results to file and optionally to database.

        Args:
            results: Evaluation results to export
        """
        average_scores = self.calculate_average_scores(results)
        results_with_averages = {**results, "average_scores": average_scores}
        results_with_averages = self._strip_multimodal_data(results_with_averages)

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

        self._strip_audio_data(results_with_averages)

        with open(filepath, "w", encoding="UTF-8") as fp:
            json.dump(results_with_averages, fp, ensure_ascii=False, cls=_NumpyEncoder)

        logger.info("Results exported to %s", filename)

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
                    logger.warning("No task_id found for task '%s'", key)
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
