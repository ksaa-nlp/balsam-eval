"""Per-file result processing for the evaluation runner.

The runner produces exactly one result JSON per pool file; backend reads them
from GCS after the job finalises, so this module no longer pushes results to
the API directly.
"""

import json
import logging
import math
import os
from statistics import mean
from typing import Any, Dict

import numpy as np
from lm_eval.api.registry import get_aggregation, get_metric_aggregation

from src.metrics_registry import get_metrics_registry

logger = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that emits plain ints / floats / lists for numpy types."""

    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return super().default(o)


class ResultProcessor:
    """Writes a per-file result JSON to ``results_dir`` and returns the path."""

    def __init__(
        self,
        *,
        category: str,
        task_id: str,
        source_pool_path: str,
        results_dir: str,
    ):
        self.category = category
        self.task_id = task_id
        self.source_pool_path = source_pool_path
        self.results_dir = results_dir

    def export(self, results: Dict[str, Any], *, filename: str) -> str:
        """Persist ``results`` to ``<results_dir>/<filename>``."""
        results = self._strip_multimodal_data(results)
        average_scores = self._calculate_average_scores(results)
        if not average_scores:
            raise RuntimeError(
                "Evaluation produced no aggregate metrics; refusing to export result"
            )
        self._add_question_scores(results)
        enriched = {
            **results,
            "average_scores": average_scores,
            "category": self.category,
            "task": self.task_id,
            "pool_file": self.source_pool_path,
        }
        self._strip_audio_data(enriched)

        os.makedirs(self.results_dir, exist_ok=True)
        path = os.path.join(self.results_dir, filename)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(
                enriched,
                fp,
                ensure_ascii=False,
                cls=_NumpyEncoder,
                separators=(",", ":"),
            )
        logger.info("Wrote result file: %s", path)
        return path

    # -- score aggregation ----------------------------------------------------

    @staticmethod
    def _add_question_scores(results: Dict[str, Any]) -> None:
        """Calculate each sample's metrics instead of storing only raw inputs."""
        samples = results.get("samples")
        if not isinstance(samples, dict):
            return

        for task_samples in samples.values():
            if not isinstance(task_samples, list):
                continue
            for sample in task_samples:
                if not isinstance(sample, dict):
                    continue
                question_scores: dict[str, Any] = {}
                for metric_name in sample.get("metrics", []):
                    if metric_name not in sample:
                        raise RuntimeError(
                            f"Required metric result is missing: {metric_name}"
                        )
                    metric = get_metrics_registry().get(metric_name)
                    aggregation_name = (
                        metric.config.aggregation_name if metric is not None else None
                    )
                    aggregation = (
                        get_aggregation(aggregation_name)
                        if aggregation_name
                        else get_metric_aggregation(metric_name)
                    )
                    if aggregation is None:
                        raise RuntimeError(
                            f"Required metric aggregation is unavailable: {metric_name}"
                        )
                    score = aggregation([sample[metric_name]])
                    if isinstance(score, dict) and "rougeLsum" in score:
                        score = score["rougeLsum"]
                    if (
                        isinstance(score, (bool, np.bool_))
                        or not isinstance(score, (int, float, np.integer, np.floating))
                        or not math.isfinite(score)
                    ):
                        raise RuntimeError(
                            f"Invalid per-question metric value for {metric_name}"
                        )
                    question_scores[metric_name] = score
                sample["scores"] = question_scores

    def _calculate_average_scores(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Average ``key,none`` metrics across every per-task result block."""
        averaged: dict[str, float] = {}
        collected: dict[str, list[float]] = {}

        per_task = results.get("results")
        if not isinstance(per_task, dict):
            raise RuntimeError("Evaluation results must contain task metric blocks")

        for task_name, task_result in per_task.items():
            if not isinstance(task_result, dict):
                raise RuntimeError(f"Malformed aggregate block for task {task_name}")
            task_metric_count = 0
            for key, value in task_result.items():
                if not key.endswith(",none") or key.endswith("_stderr,none"):
                    continue
                task_metric_count += 1
                metric_name = key.replace(",none", "")
                if isinstance(value, dict):
                    if "rougeLsum" in value:
                        score = value["rougeLsum"]
                        if (
                            isinstance(score, (bool, np.bool_))
                            or not isinstance(
                                score, (int, float, np.integer, np.floating)
                            )
                            or not math.isfinite(score)
                        ):
                            raise RuntimeError(f"Invalid aggregate metric value for {key}")
                        collected.setdefault(metric_name, []).append(
                            float(score))
                        continue
                    raise RuntimeError(f"Unsupported aggregate metric value for {key}")
                if (
                    isinstance(value, (int, float, np.integer, np.floating))
                    and not isinstance(value, (bool, np.bool_))
                    and math.isfinite(value)
                ):
                    collected.setdefault(metric_name, []).append(float(value))
                    continue
                if isinstance(value, (int, float, np.integer, np.floating)):
                    raise RuntimeError(f"Invalid aggregate metric value for {key}")
                raise RuntimeError(
                    f"Unsupported aggregate metric value for {key}: {type(value).__name__}"
                )
            if task_metric_count == 0:
                raise RuntimeError(f"Task {task_name} has no aggregate metric values")

        for metric_name, scores in collected.items():
            if scores:
                averaged[metric_name] = round(mean(scores), 4)
        return averaged

    # -- payload cleanup ------------------------------------------------------

    @staticmethod
    def _strip_audio_data(results: Dict[str, Any]) -> None:
        """Strip raw audio arrays so we don't bloat the result JSON."""
        samples = results.get("samples") or {}
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
        """Drop audio / image / visuals payloads from sample arguments."""
        samples = results.get("samples")
        if not samples:
            return results

        cleaned_samples: dict[str, list[Any]] = {}
        for task_name, task_samples in samples.items():
            cleaned = []
            for sample in task_samples:
                sample = {**sample}
                args = sample.get("arguments")
                if isinstance(args, list):
                    new_args = []
                    for arg_group in args:
                        if isinstance(arg_group, (list, tuple)):
                            new_group = []
                            for item in arg_group:
                                if isinstance(item, dict):
                                    item = {
                                        k: v
                                        for k, v in item.items()
                                        if k not in ("audio", "images", "visuals")
                                    }
                                new_group.append(item)
                            new_args.append(new_group)
                        else:
                            new_args.append(arg_group)
                    sample["arguments"] = new_args
                cleaned.append(sample)
            cleaned_samples[task_name] = cleaned

        return {**results, "samples": cleaned_samples}
