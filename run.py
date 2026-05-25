"""Unified entry point for evaluation jobs.

The runner evaluates one pool-dataset file at a time. Each file produces a
single result JSON tagged with its category and task.

* **Remote mode** (Cloud Build): ``POOL_FILES`` lists GCS object paths inside
  ``GCLOUD_BUCKET``. Each file is downloaded, evaluated, and the result JSON
  is uploaded back to ``gs://${GCLOUD_BUCKET}/${RESULTS_PATH}/<file>.json``.
  The runner finalises the job by posting to
  ``POST {API_HOST}/evaluation-jobs/{JOB_ID}/finalize``.
* **Local mode**: ``POOL_FILES`` is unset, so JSON files in ``.tasks/`` are
  picked up instead. No status calls are made; results stay in ``.results/``.
"""

import hashlib
import json
import logging
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional, Tuple

from src.adapters.utils import process_adapter_and_url
from src.core.common import (
    copy_audio_to_temp,
    copy_images_to_temp,
    copy_metrics_combined_to_temp,
    copy_multimodal_utils_to_temp,
    set_api_key_for_adapter,
    setup_directories,
)
from src.core.config import EvalConfig
from src.core.helpers import (
    download_pool_file_from_gcs,
    upload_result_file_to_gcs,
)
from src.db_operations import JobOutcome, finalize_job
from src.evaluation import SingleFileEvaluationJob
from src.task import LMHDataset

TEMP_DIR = ".temp"
TASKS_DIR = ".tasks"
RESULTS_DIR = ".results"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)


def _log_job_start(config: EvalConfig, sources: list[str]) -> None:
    logger.info("=" * 80)
    logger.info("EVALUATION JOB STARTED")
    logger.info("Timestamp: %s", time.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Mode: %s", "remote" if config.is_remote_job() else "local")
    logger.info("Job ID: %s", config.job_id)
    logger.info("Category: %s", config.category_id)
    logger.info("Adapter: %s", config.adapter)
    logger.info("Model: %s", config.model_name)
    logger.info("Files (%d):", len(sources))
    for src in sources:
        logger.info("  - %s", src)
    logger.info("=" * 80)


def _log_job_end(succeeded: bool) -> None:
    logger.info("=" * 80)
    logger.info("EVALUATION JOB %s", "SUCCEEDED" if succeeded else "FAILED")
    logger.info("Timestamp: %s", time.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 80)


def resolve_pool_files(config: EvalConfig) -> list[str]:
    """Return the list of source identifiers (GCS paths or local file paths).

    Remote mode: returns POOL_FILES exactly as supplied by the backend.
    Local mode: discovers ``*.json`` task files under ``.tasks/`` (skipping
    already-exported ``_test.json`` / ``_dev.json`` split files).
    """
    if config.pool_files:
        return list(config.pool_files)

    if not os.path.isdir(TASKS_DIR):
        return []

    found: list[str] = []
    for name in sorted(os.listdir(TASKS_DIR)):
        if not name.endswith(".json"):
            continue
        if name.endswith("_test.json") or name.endswith("_dev.json"):
            continue
        found.append(os.path.join(TASKS_DIR, name))
    return found


def _slugify_source(source: str) -> str:
    """Derive a unique filename stem from a pool-file source path.

    The runner can be handed multiple pool files whose basenames collide
    (e.g. ``cat-A/dataset-7.json`` and ``cat-B/dataset-7.json`` after GCS
    prefixing). We hash the full path so each gets a distinct local
    materialisation under ``.temp/`` and a distinct result filename in GCS,
    while still keeping a human-readable suffix.
    """
    stem = Path(source).stem or "pool-file"
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
    return f"{stem}-{digest}"


def _materialise_pool_file(source: str, is_remote: bool, bucket: Optional[str]) -> str:
    """Place a pool file into TEMP_DIR in the flat format LMHDataset expects.

    Returns the basename (without extension) of the file inside TEMP_DIR.
    """
    file_stem = _slugify_source(source)
    dest_path = os.path.join(TEMP_DIR, f"{file_stem}.json")

    if is_remote:
        if not bucket:
            raise ValueError("GCLOUD_BUCKET is required for remote pool files")
        download_pool_file_from_gcs(
            bucket=bucket,
            object_path=source,
            dest_path=dest_path,
        )
    else:
        os.makedirs(TEMP_DIR, exist_ok=True)
        shutil.copyfile(source, dest_path)

    with open(dest_path, "r", encoding="utf-8") as fp:
        raw = json.load(fp)

    # Legacy local format: payload nested under ``json`` with outer category/task.
    if isinstance(raw, dict) and isinstance(raw.get("json"), dict):
        inner = raw["json"]
        if "task" in raw:
            inner["task"] = raw["task"]
        if "category" in raw:
            inner["category"] = raw["category"]
        normalised: dict[str, Any] = inner
    else:
        normalised = raw

    with open(dest_path, "w", encoding="utf-8") as fp:
        json.dump(normalised, fp, ensure_ascii=False)

    return Path(dest_path).stem


def _evaluate_one_file(
    *,
    source: str,
    is_remote: bool,
    config: EvalConfig,
    processed_adapter: str,
    model_args: dict[str, Any],
) -> Tuple[str, str]:
    """Evaluate a single pool file. Returns (local_result_path, result_filename)."""
    file_stem = _materialise_pool_file(source, is_remote, config.bucket)

    dataset = LMHDataset(file_stem, directory=TEMP_DIR)
    dataset.export()

    # Materialise any referenced media into TEMP_DIR for lm_eval's loader.
    # Remote mode passes the GCS bucket so references stored as object paths
    # (e.g. ``cat-1/dataset-7/img.png``) can be pulled down on demand.
    media_bucket = config.bucket if is_remote else None
    for split in ("test", "dev"):
        split_file = os.path.join(TEMP_DIR, f"{dataset.file_name}_{split}.json")
        if os.path.exists(split_file):
            copy_images_to_temp(split_file, TEMP_DIR, bucket=media_bucket)
            copy_audio_to_temp(split_file, TEMP_DIR, bucket=media_bucket)

    category = str(dataset.category_id or config.category_id or "")
    task_id = str(dataset.task_id or "")
    result_filename = f"{file_stem}.json"

    job = SingleFileEvaluationJob(
        task_name=dataset.name,
        category=category,
        task_id=task_id,
        source_pool_path=source,
        adapter=processed_adapter,
        model_args=model_args,
        result_filename=result_filename,
        results_dir=RESULTS_DIR,
    )
    return job(), result_filename


def _try_finalize(config: EvalConfig, outcome: JobOutcome, error: Optional[str] = None) -> None:
    """Best-effort POST to ``/evaluation-jobs/:id/finalize``.

    Swallows network failures so the caller can continue with ``sys.exit`` —
    a finalize failure is logged but shouldn't replace the original error.
    """
    if not config.is_remote_job():
        return
    try:
        finalize_job(
            api_host=config.api_host or "",
            finalize_token=config.finalize_token or "",
            job_id=config.job_id or "",
            outcome=outcome,
            error=(error or None) and (error or "")[:4000],
        )
    except Exception as finalize_exc:  # pylint: disable=broad-exception-caught
        logger.error("Finalize call failed (%s): %s", outcome.value, finalize_exc)


def _run() -> int:
    """Execute the job. Returns a process exit code.

    All preparation (config load, validation, dir setup) happens here so
    failures are caught by the top-level handler in ``main`` and reported
    back to the backend via ``finalize``.
    """
    _setup_logging()
    setup_directories(TEMP_DIR, RESULTS_DIR)
    copy_multimodal_utils_to_temp(TEMP_DIR)
    copy_metrics_combined_to_temp(TEMP_DIR)

    config = EvalConfig.from_env()
    set_api_key_for_adapter(config.adapter, config.api_key)

    is_remote = config.is_remote_job()
    if is_remote:
        config.validate_remote()
    else:
        config.validate_local()

    processed_adapter, processed_base_url = process_adapter_and_url(
        config.adapter, config.base_url
    )
    model_args = config.get_model_args(processed_base_url)

    pool_files = resolve_pool_files(config)
    _log_job_start(config, pool_files)

    if not pool_files:
        message = "No pool files to evaluate."
        logger.error(message)
        _try_finalize(config, JobOutcome.FAILED, message)
        return 1

    failures: list[str] = []

    for source in pool_files:
        logger.info("--- Evaluating: %s ---", source)
        try:
            local_result, result_filename = _evaluate_one_file(
                source=source,
                is_remote=is_remote,
                config=config,
                processed_adapter=processed_adapter,
                model_args=model_args,
            )
            if is_remote and config.bucket and config.results_path:
                upload_result_file_to_gcs(
                    bucket=config.bucket,
                    local_path=local_result,
                    object_path=f"{config.results_path.rstrip('/')}/{result_filename}",
                )
            logger.info("Completed: %s", source)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Failed to evaluate %s: %s", source, exc)
            logger.error("Traceback:\n%s", traceback.format_exc())
            failures.append(f"{source}: {exc}")

    succeeded = not failures
    _log_job_end(succeeded)

    if succeeded:
        _try_finalize(config, JobOutcome.SUCCEEDED)
        return 0

    _try_finalize(config, JobOutcome.FAILED, "; ".join(failures))
    return 1


def main() -> None:
    """Top-level entry point.

    A bare ``try/except`` here is what stops the backend from sitting on a
    stuck job: any unexpected exception (config validation failure, GCS auth
    error, import-time crash, etc.) is reported via ``finalize`` instead of
    letting the runner exit silently.
    """
    try:
        exit_code = _run()
    except SystemExit:
        raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Fatal error in runner: %s", exc)
        logger.error("Traceback:\n%s", traceback.format_exc())
        # Best-effort: rebuild minimum config from env to report the failure.
        try:
            config = EvalConfig.from_env()
            _try_finalize(config, JobOutcome.FAILED, f"runner crashed: {exc}")
        except Exception as cfg_exc:  # pylint: disable=broad-exception-caught
            logger.error("Could not load config to report crash: %s", cfg_exc)
        sys.exit(1)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
