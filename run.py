"""Unified entry point for evaluation jobs (both local and remote)."""

import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.adapter_utils import process_adapter_and_url
from src.core.common import (
    copy_images_to_temp,
    copy_multimodal_utils_to_temp,
    set_api_key_for_adapter,
    setup_directories,
)
from src.core.config import EvalConfig
from src.db_operations import get_tasks_from_category, submit_model_evaluation
from src.evaluation import EvaluationJob
from src.core.helpers import download_dataset_from_gcs
from src.task import LMHDataset

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Directories
TEMP_DIR = ".temp"
TASKS_DIR = ".tasks"
RESULTS_DIR = ".results"


def setup_logging(local: bool = False) -> None:
    """Setup logging configuration.

    Args:
        local: If True, log to stdout; otherwise use default logging
    """
    if local:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def log_job_start(config: EvalConfig) -> None:
    """Log job start information.

    Args:
        config: Evaluation configuration
    """
    logger.info("="*80)
    logger.info("EVALUATION JOB STARTED")
    logger.info("Timestamp: %s", time.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Job ID: %s", config.job_id)
    logger.info("Category: %s", config.category_id)
    logger.info("Adapter: %s", config.adapter)
    logger.info("Model: %s", config.model_name)
    logger.info("\n%s", "=" * 80)


def log_job_end() -> None:
    """Log job completion information."""
    logger.info("\n%s", "=" * 80)
    logger.info("EVALUATION JOB COMPLETED")
    logger.info("Timestamp: %s", time.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("\n%s", "=" * 80)


def load_local_tasks() -> tuple[dict[str, list[str]], dict[str, str]]:
    """Load tasks from local .tasks directory.

    Returns:
        Tuple of (tasks_by_category, task_mapper)
        - tasks_by_category: {category: [dataset_names]}
        - task_mapper: {dataset_name: task_id}
    """
    tasks_temp: dict[str, list[str]] = {}
    task_mapper: dict[str, str] = {}

    logger.info("Reading tasks from directory '%s'", TASKS_DIR)

    for file in os.listdir(f"./{TASKS_DIR}"):
        if not file.endswith("json"):
            continue

        with open(f"./{TASKS_DIR}/{file}", "r", encoding="utf-8") as f:
            content = f.read()
            d = json.loads(content)

            # Skip datasets with no metric
            if "json" in d:
                if d["json"]["metric_list"][0]["metric"] == "":
                    continue

            task_mapper[d["name"]] = d["task"]

            with open(f"./{TEMP_DIR}/{file}", "w", encoding="utf-8") as f_out:
                if "json" in d:
                    d["json"]["category"] = d["category"]
                    d["json"]["task"] = d["task"]
                    json.dump(d["json"], f_out, ensure_ascii=False)
                else:
                    json.dump(d, f_out, ensure_ascii=False)

            dataset = LMHDataset(str(file.rsplit(".", 1)[0]), TEMP_DIR)
            dataset.export()

            # Copy images to .temp directory if dataset contains images
            for split in ["test", "dev"]:
                json_file = os.path.join(TEMP_DIR, f"{dataset.file_name}_{split}.json")
                if os.path.exists(json_file):
                    copy_images_to_temp(json_file, TEMP_DIR)

            if tasks_temp.get(d["category"]) is None:
                tasks_temp[d["category"]] = []
            tasks_temp[d["category"]].append(dataset.name)

    return tasks_temp, task_mapper


def load_remote_datasets(config: EvalConfig) -> list[LMHDataset]:
    """Load datasets from remote GCS storage.

    Args:
        config: Evaluation configuration

    Returns:
        List of datasets
    """
    if not config.category_id:
        raise ValueError("category_id is required for remote evaluation")
    if not config.api_host:
        raise ValueError("api_host is required for remote evaluation")
    if not config.server_token:
        raise ValueError("server_token is required for remote evaluation")

    datasets_ids = get_tasks_from_category(
        category=config.category_id,
        api_host=config.api_host,
        server_token=config.server_token,
        evaluation_types=config.evaluation_types,
    )

    datasets: list[LMHDataset] = []
    for dataset_id in datasets_ids:
        # Download and export each dataset
        returned_data = download_dataset_from_gcs(
            dataset_id=dataset_id, directory=TEMP_DIR
        )
        dataset = LMHDataset(dataset_id, directory=TEMP_DIR)
        dataset.export()

        # Copy images to .temp directory if dataset contains images
        for split in ["test", "dev"]:
            json_file = f"{TEMP_DIR}/{dataset.file_name}_{split}.json"
            if os.path.exists(json_file):
                copy_images_to_temp(json_file, TEMP_DIR)

        datasets.append(dataset)

    return datasets


def organize_remote_datasets(
    datasets: list[LMHDataset],
) -> dict[str, dict[str, list[LMHDataset]]]:
    """Organize remote datasets by category and task.

    Args:
        datasets: List of datasets

    Returns:
        Nested dict: {category: {task: [datasets]}}
    """
    categories: dict[str, dict[str, list[LMHDataset]]] = {}

    for dataset in datasets:
        if dataset.category_id:
            if categories.get(dataset.category_id) is None:
                categories[dataset.category_id] = {}
            if categories[dataset.category_id].get(str(dataset.task_id)) is None:
                categories[dataset.category_id][str(dataset.task_id)] = []
            categories[dataset.category_id][dataset.task_id].append(dataset)

    # Clean up empty categories/tasks
    for category in list(categories.keys()):
        if len(categories[category]) == 0:
            del categories[category]
        for task in list(categories[category].keys()):
            if len(categories[category][task]) == 0:
                del categories[category][task]

    return categories


def run_local_evaluation(config: EvalConfig) -> None:
    """Run local evaluation job.

    Args:
        config: Evaluation configuration
    """
    # Set environment
    os.environ["ENV"] = "local"
    setup_logging(local=True)

    # Process adapter and base_url
    processed_adapter, processed_base_url = process_adapter_and_url(
        config.adapter, config.base_url
    )

    # Load tasks from local directory
    tasks_temp, task_mapper = load_local_tasks()

    # Build model arguments
    model_args = config.get_model_args(processed_base_url)

    # Submit evaluation if configured
    submit_results: dict[str, Any] = {"jobs_ids": {}}
    if all(
        [
            config.adapter,
            config.server_token,
            config.api_host,
            config.user_id,
            config.benchmark_id,
            tasks_temp,
        ]
    ):
        if not processed_base_url:
            raise ValueError("model_url is required for submitting evaluation")
        if not config.api_key:
            raise ValueError("api_key is required for submitting evaluation")

        submit_results = submit_model_evaluation(
            model_name=config.model_name,
            model_url=processed_base_url,
            adapter=processed_adapter,
            api_key=config.api_key,
            categories=list(tasks_temp.keys()),
            server_token=config.server_token or "",  # type: ignore[arg-type]
            api_host=config.api_host or "",  # type: ignore[arg-type]
            user_id=config.user_id or "",  # type: ignore[arg-type]
            benchmark_id=config.benchmark_id or "",  # type: ignore[arg-type]
            evaluation_types=config.get_evaluation_types_list(),
        )
        logger.info(submit_results)

        if submit_results["status_code"] != 200:
            raise RuntimeError(f"[ERROR] Failed to submit evaluation: {submit_results}")

    def run_category(category: str, datasets: list[str]) -> tuple[str, bool, str | None]:
        """Run evaluation for a single category.

        Returns:
            Tuple of (category, success, error_message)
        """
        logger.info("Running evaluation for category: %s", category)

        try:
            job = EvaluationJob(
                tasks=datasets,
                adapter=processed_adapter,  # type: ignore[arg-type]
                model_args=model_args,
                tasks_mapper_dict=task_mapper,
                category_name=category,
                job_id=submit_results["jobs_ids"].get(category, None),
                llm_judge_api_key=config.llm_judge_api_key,
                llm_judge_model=config.llm_judge,
                llm_judge_provider=config.llm_judge_provider,
                server_token=config.server_token,
                api_host=config.api_host,
                benchmark_id=config.benchmark_id,
            )
            job()
            return (category, True, None)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(
                "An error occurred while running the job for task %s: %s",
                category, e
            )
            return (category, False, str(e))

    # Run categories sequentially or in parallel
    if config.parallel_categories and len(tasks_temp) > 1:
        logger.info("Running %d categories in parallel...", len(tasks_temp))
        with ThreadPoolExecutor(
            max_workers=min(len(tasks_temp), os.cpu_count() or 4)
        ) as executor:
            # Submit all category jobs
            future_to_category = {
                executor.submit(run_category, category, datasets): category
                for category, datasets in tasks_temp.items()
            }

            # Process completed jobs
            for future in as_completed(future_to_category):
                category, success, error = future.result()
                if success:
                    logger.info("Category '%s' completed successfully", category)
                else:
                    logger.error("Category '%s' failed with error: %s", category, error)
    else:
        # Sequential execution (default behavior)
        for category, datasets in tasks_temp.items():
            run_category(category, datasets)


def run_remote_evaluation(config: EvalConfig) -> None:
    """Run remote evaluation job.

    Args:
        config: Evaluation configuration
    """
    setup_logging(local=True)
    log_job_start(config)

    # Process adapter and base_url
    processed_adapter, processed_base_url = process_adapter_and_url(
        config.adapter, config.base_url
    )

    # Load datasets from remote storage
    datasets = load_remote_datasets(config)

    # Organize datasets by category and task
    categories = organize_remote_datasets(datasets)

    logger.info("Total categories: %d", len(categories))
    logger.info(str(categories))

    # Build model arguments
    model_args = config.get_model_args(processed_base_url)

    # Run evaluation job per task
    for category, tasks in categories.items():
        logger.info("Running evaluation for category: %s", category)
        logger.info("Total tasks: %d", len(tasks))

        for task, task_datasets in tasks.items():
            if len(task_datasets) == 0:
                logger.warning("Skipping task '%s' - no datasets found", task)
                continue

            logger.info("\n%s", "=" * 80)
            logger.info("Starting evaluation for task: %s", task)
            logger.info("Category: %s", category)
            logger.info("Number of datasets: %d", len(task_datasets))
            logger.info("Dataset names: %s", [dataset.name for dataset in task_datasets])
            logger.info("Adapter: %s", processed_adapter)
            logger.info("Model: %s", config.model_name)
            logger.info("\n%s", "=" * 80)

            try:
                job = EvaluationJob(
                    tasks=[dataset.name for dataset in task_datasets],
                    adapter=processed_adapter,  # type: ignore[arg-type]
                    model_args=model_args,
                    task_id=task,
                    job_id=config.job_id,
                    api_host=config.api_host,
                    server_token=config.server_token,
                    category_name=category,
                    benchmark_id=config.benchmark_id,
                    llm_judge_api_key=config.llm_judge_api_key,
                    llm_judge_model=config.llm_judge,
                    llm_judge_provider=config.llm_judge_provider,
                )
                job()
                logger.info("✅ Task '%s' completed successfully", task)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("❌ Task '%s' failed with error: %s", task, e)
                import traceback

                logger.error("Error traceback:\n%s", traceback.format_exc())
                sys.exit(1)

    log_job_end()

    # Log result files
    result_files = list(Path(RESULTS_DIR).glob("*.json")) + list(Path(".").glob("*.json"))
    logger.info("Generated result files: %s", result_files)


def main() -> None:
    """Main entry point."""
    # Setup directories
    setup_directories(TEMP_DIR, RESULTS_DIR)

    # Copy multimodal utils
    copy_multimodal_utils_to_temp(TEMP_DIR)

    # Load configuration
    config = EvalConfig.from_env()

    # Set API key environment variable
    set_api_key_for_adapter(config.adapter, config.api_key)

    # Determine execution mode and run
    if config.is_remote_job():
        config.validate_remote()
        run_remote_evaluation(config)
    else:
        config.validate_local()
        run_local_evaluation(config)


if __name__ == "__main__":
    main()
