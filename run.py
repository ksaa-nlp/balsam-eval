"""Entry point for the evaluation job."""

import logging
import os

import requests

from src.evaluation import EvaluatationJob
from src.task import LMHDataset, download_dataset_from_gcs


CATEGORY = os.getenv("CATEGORY", "")
API_HOST = os.getenv("API_HOST", "")
SERVER_TOKEN = os.getenv("SERVER_TOKEN", "")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_tasks_from_category() -> list[str]:
    if not CATEGORY:
        raise ValueError("Category is required, terminating the process.")

    logger.info("Getting tasks for category %s", CATEGORY)

    webhook_url = f"https://{API_HOST}/api/tasks/{CATEGORY}"
    logger.info("Sending request to %s", webhook_url)
    response = requests.get(
        webhook_url,
        headers={
            "Content-Type": "application/json",
            "x-server-token": SERVER_TOKEN,
        },
        timeout=20,
    )
    response.raise_for_status()

    if response.status_code != 200:
        logger.error("Failed to retrieve tasks for category %s", CATEGORY)
        raise ValueError("Failed to retrieve tasks for category")

    # Make sure we have the datasets
    if "datasets" not in response.json():
        logger.error("No datasets found for category %s", CATEGORY)
        raise ValueError("No datasets found for category")

    return response.json()["datasets"]


if __name__ == "__main__":

    # Get the tasks for the category
    datasets_ids = get_tasks_from_category()

    datasets: list[LMHDataset] = []
    for dataset_id in datasets_ids:
        # Write the dataset files to disk
        # TODO download a list of datasets instead of a looping through
        download_dataset_from_gcs(dataset_id=dataset_id, directory=".temp")
        # Initialize the LM Hanress task files and export it
        # to a temporary directory
        dataset = LMHDataset(dataset_id, directory=".temp")
        dataset.export()

        datasets.append(dataset)

    # Map categories to datasets
    categories: dict[str, dict[str, list[LMHDataset]]] = {}
    for dataset in datasets:
        if dataset.category_id:
            if categories.get(dataset.category_id) is None:
                categories[dataset.category_id] = {}
            if categories[dataset.category_id].get(str(dataset.task_id)) is None:
                categories[dataset.category_id][str(dataset.task_id)] = []
            categories[dataset.category_id][dataset.task_id].append(dataset)

    print(f"Total categories: {len(categories)}")
    print(categories)

    # We're sending all tasks in a category at once to LM Harness
    for category, tasks in categories.items():
        print(f"Running evaluation for category: {category}")
        print(f"Total tasks: {len(datasets)}")

        for task, _datasets in tasks.items():
            job = EvaluatationJob(
                base_url=os.environ["BASE_URL"],
                tasks=[dataset.name for dataset in _datasets],
                api_key=os.getenv("API_KEY", "openai-api-key"),
                adapter=os.environ["ADAPTER"],
                model=os.environ["MODEL"],
                task_id=task,
                output_path=f"{task}",
            )
            job.run()
