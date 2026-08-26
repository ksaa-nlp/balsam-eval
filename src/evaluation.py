"""Single-file evaluation job used by the unified runner.

One ``SingleFileEvaluationJob`` instance corresponds to one pool-dataset file
and produces exactly one result JSON.
"""

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, cast

import lm_eval.evaluator
import lm_eval.models  # noqa: F401  pylint: disable=unused-import  # registers models
import lm_eval.tasks
import requests
from lm_eval.api.registry import get_model

import src.adapters  # noqa: F401  pylint: disable=unused-import  # registers adapters
import src.metrics  # noqa: F401  pylint: disable=unused-import  # registers custom metrics
from src.adapter_config import API_KEY_ENV_BY_ADAPTER, ASR_ADAPTERS
from src.adapters.utils import get_max_tokens_config
from src.processors.result_processing import ResultProcessor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_NON_BATCHING_ADAPTERS = {
    "anthropic-chat-completions",
    "cohere",
    "local-chat-completions",
    "openai",
    "openai-chat-completions",
}


def create_evaluation_model(
    adapter: str, model_args: dict[str, Any], batch_size: int
) -> Any:
    """Initialize one lm-eval model for reuse across all pool files."""
    args = dict(model_args)
    args.setdefault("eos_string", "<|endoftext|>")
    model_batch_size = 1 if adapter in _NON_BATCHING_ADAPTERS else batch_size
    return get_model(adapter).create_from_arg_obj(
        args,
        {"batch_size": model_batch_size, "max_batch_size": None, "device": None},
    )


# --- Compatibility patches ---------------------------------------------------

@contextmanager
def _lm_eval_compatibility_patches():
    """Apply legacy lm_eval workarounds only while it is executing."""
    original_relative_to = Path.relative_to
    original_post = requests.post

    def safe_relative_to(self, *args, **kwargs):
        try:
            return original_relative_to(self, *args, **kwargs)
        except ValueError as exc:
            if "not in the subpath of" in str(exc):
                return self
            raise

    def post_with_default_timeout(url, *args, **kwargs):
        kwargs.setdefault("timeout", 5000)
        return original_post(url, *args, **kwargs)

    Path.relative_to = safe_relative_to  # type: ignore[method-assign]
    requests.post = post_with_default_timeout  # type: ignore[assignment]
    try:
        yield
    finally:
        Path.relative_to = original_relative_to  # type: ignore[method-assign]
        requests.post = original_post  # type: ignore[assignment]


def _set_api_key_env(adapter: str, api_key: Optional[str]) -> None:
    if not api_key:
        return
    env_var = API_KEY_ENV_BY_ADAPTER.get(adapter)
    if env_var:
        os.environ[env_var] = api_key


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
        model: Any = None,
        batch_size: int = 8,
        bootstrap_iters: int = 100000,
        result_filename: str,
        results_dir: str,
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
            result_filename: Name of the result JSON written into ``results_dir``.
            results_dir: Local directory where result JSONs are written.

        Adapter-specific API-key env vars are populated by
        ``core.common.set_api_key_for_adapter`` in ``run.py`` before this job
        is constructed — we don't re-stamp them here.
        """
        self.task_name = task_name
        self.category = category
        self.task_id = task_id
        self.source_pool_path = source_pool_path
        self.adapter = adapter
        self.model_args = dict(model_args)
        self.model = model
        self.batch_size = batch_size
        self.bootstrap_iters = bootstrap_iters
        self.result_filename = result_filename
        self.results_dir = results_dir

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
            raise RuntimeError(
                f"lm_eval returned no results for {self.task_name}")

        self._sanitize_results(results)
        self._stamp_category_and_task(results)

        return ResultProcessor(
            category=self.category,
            task_id=self.task_id,
            source_pool_path=self.source_pool_path,
            results_dir=self.results_dir,
        ).export(results, filename=self.result_filename)

    # -- internal helpers ----------------------------------------------------

    def _run_lm_eval(self) -> Dict[str, Any]:
        temp_dir = Path(".temp").resolve()
        use_chat_template = self.adapter not in ASR_ADAPTERS
        model_batch_size = getattr(self.model, "batch_size", None)
        effective_batch_size = (
            model_batch_size
            if isinstance(model_batch_size, (int, str))
            else self.batch_size
        )

        with _lm_eval_compatibility_patches():
            results = cast(
                dict[str, Any],
                lm_eval.evaluator.simple_evaluate(
                    model=self.model if self.model is not None else self.adapter,
                    model_args=None if self.model is not None else self.model_args,
                    tasks=[self.task_name],
                    apply_chat_template=use_chat_template,
                    task_manager=lm_eval.tasks.TaskManager(
                        include_path=str(temp_dir), include_defaults=False),
                    batch_size=effective_batch_size,
                    bootstrap_iters=self.bootstrap_iters,
                    log_samples=True,
                    gen_kwargs=get_max_tokens_config(
                        self.adapter, self.model_args["model"]),
                ),
            )

        # Reusing a model makes lm-eval report its class name and no arguments.
        # Preserve result metadata produced before model reuse was introduced.
        config = results.get("config")
        if self.model is not None and isinstance(config, dict):
            config["model"] = self.adapter
            config["model_args"] = dict(self.model_args)
        return results

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
