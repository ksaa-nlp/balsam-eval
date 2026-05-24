"""Single-file evaluation job used by the unified runner.

One ``SingleFileEvaluationJob`` instance corresponds to one pool-dataset file
and produces exactly one result JSON.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, cast

import lm_eval.evaluator
import lm_eval.models  # noqa: F401  pylint: disable=unused-import  # registers models
import lm_eval.tasks
import requests

import src.adapters  # noqa: F401  pylint: disable=unused-import  # registers adapters
import src.metrics  # noqa: F401  pylint: disable=unused-import  # registers custom metrics
from src.adapters.utils import get_max_tokens_config
from src.processors.result_processing import ResultProcessor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# --- Compatibility patches ---------------------------------------------------

_original_relative_to = Path.relative_to


def _safe_relative_to(self, *args, **kwargs):
    """``Path.relative_to`` that returns the original path on ValueError."""
    try:
        return _original_relative_to(self, *args, **kwargs)
    except ValueError as e:
        if "not in the subpath of" in str(e):
            return self
        raise


Path.relative_to = _safe_relative_to  # type: ignore[method-assign]

# lm_eval clients occasionally call ``requests.post`` without a timeout — give
# them a generous default so we don't hang indefinitely on a stuck connection.
requests.post = lambda url, timeout=5000, **kwargs: requests.request(  # type: ignore[assignment]
    method="POST", url=url, timeout=timeout, **kwargs
)


_ASR_ADAPTERS = {"openai-asr", "google-stt", "azure-stt"}


def _set_api_key_env(adapter: str, api_key: Optional[str]) -> None:
    if not api_key:
        return
    if adapter in {"openai-chat-completions", "local-chat-completions", "openai", "openai-asr"}:
        os.environ["OPENAI_API_KEY"] = api_key
    elif adapter in {"anthropic-chat-completions", "anthropic"}:
        os.environ["ANTHROPIC_API_KEY"] = api_key
    elif adapter == "gemini":
        os.environ["GOOGLE_API_KEY"] = api_key
    elif adapter == "cohere":
        os.environ["CO_API_KEY"] = api_key
    elif adapter == "azure-stt":
        os.environ["AZURE_SPEECH_KEY"] = api_key


class SingleFileEvaluationJob:
    """Evaluate exactly one pool-dataset file and persist its result."""

    def __init__(
        self,
        *,
        task_name: str,
        category: str,
        task_id: str,
        source_pool_path: str,
        adapter: str,
        model_args: dict[str, Any],
        result_filename: str,
    ):
        """
        Args:
            task_name: Generated lm_eval task name (LMHDataset.name).
            category: Category identifier from the pool file (or fallback).
            task_id: Task identifier from the pool file (or fallback).
            source_pool_path: GCS path or local path the file came from
                (preserved in the result file as ``pool_file``).
            adapter: lm_eval adapter id (after any pre-processing).
            model_args: Arguments passed to ``simple_evaluate``.
            result_filename: Name of the result JSON in ``.results/``.
        """
        self.task_name = task_name
        self.category = category
        self.task_id = task_id
        self.source_pool_path = source_pool_path
        self.adapter = adapter
        self.model_args = dict(model_args)
        self.result_filename = result_filename

        if "eos_string" not in self.model_args:
            self.model_args["eos_string"] = "<|endoftext|>"

        _set_api_key_env(self.adapter, os.getenv("API_KEY"))

    def __call__(self) -> str:
        """Run the evaluation. Returns the local path of the result JSON."""
        logger.info(
            "Running lm_eval (task=%s, category=%s, task_id=%s, source=%s)",
            self.task_name,
            self.category,
            self.task_id,
            self.source_pool_path,
        )

        results = self._run_lm_eval()
        if not results:
            raise RuntimeError(f"lm_eval returned no results for {self.task_name}")

        self._sanitize_results(results)
        self._stamp_category_and_task(results)

        return ResultProcessor(
            category=self.category,
            task_id=self.task_id,
            source_pool_path=self.source_pool_path,
        ).export(results, filename=self.result_filename)

    # -- internal helpers ----------------------------------------------------

    def _run_lm_eval(self) -> Dict[str, Any]:
        temp_dir = Path(".temp").resolve()
        use_chat_template = self.adapter not in _ASR_ADAPTERS

        return cast(
            dict[str, Any],
            lm_eval.evaluator.simple_evaluate(
                model=self.adapter,
                model_args=self.model_args,
                tasks=[self.task_name],
                apply_chat_template=use_chat_template,
                task_manager=lm_eval.tasks.TaskManager(include_path=str(temp_dir)),
                batch_size=1,
                gen_kwargs=get_max_tokens_config(self.adapter, self.model_args["model"]),
            ),
        )

    @staticmethod
    def _sanitize_results(results: Dict[str, Any]) -> None:
        cfg = results.get("config")
        if isinstance(cfg, dict):
            model_args = cfg.get("model_args")
            if isinstance(model_args, dict):
                model_args.pop("api_key", None)

    def _stamp_category_and_task(self, results: Dict[str, Any]) -> None:
        per_task = results.get("results") or {}
        for task_block in per_task.values():
            if isinstance(task_block, dict):
                task_block.setdefault("task", self.task_id)
                task_block.setdefault("category", self.category)
        results["category"] = self.category
        results["task"] = self.task_id
        results["pool_file"] = self.source_pool_path
