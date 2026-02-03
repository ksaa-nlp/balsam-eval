"""Entry point for the evaluation job."""

import logging
import os

from dotenv import load_dotenv
from src.db_operations import get_tasks_from_category
from src.evaluation import EvaluationJob
from src.helpers import download_dataset_from_gcs
from src.task import LMHDataset
from src.adapter_utils import process_adapter_and_url

# Load environment variables
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Constants from environment
API_HOST = os.getenv("API_HOST")
SERVER_TOKEN = os.getenv("SERVER_TOKEN")
CATEGORY_ID = os.getenv("CATEGORY")
BENCHMARK_ID = os.getenv("BENCHMARK_ID")
ADAPTER = os.getenv("ADAPTER")
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MAX_TOKENS = os.getenv("MAX_TOKENS")
TEMPERATURE = os.getenv("TEMPERATURE")
MODEL_NAME = os.getenv("MODEL")
JOB_ID = os.getenv("JOB_ID")
EVALUATION_TYPES = os.getenv("EVALUATION_TYPES")
LLM_JUDGE = os.getenv("JUDGE_MODEL")
LLM_JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER")
LLM_JUDGE_API_KEY = os.getenv("JUDGE_API_KEY")

# Validation
if not all([API_HOST, SERVER_TOKEN, CATEGORY_ID, ADAPTER, BENCHMARK_ID]):
    raise ValueError(
        "API_HOST, SERVER_TOKEN, CATEGORY, BENCHMARK_ID, and ADAPTER environment variables are required"
    )
if not MODEL_NAME:
    raise ValueError("MODEL name is required")


# Collect datasets
if __name__ == "__main__":
    if not CATEGORY_ID or not API_HOST or not SERVER_TOKEN:
        raise ValueError("CATEGORY, API_HOST, and SERVER_TOKEN must be set.")
    
    # Process adapter and base_url using shared utility
    processed_adapter, processed_base_url = process_adapter_and_url(ADAPTER, BASE_URL)
    
    datasets_ids = get_tasks_from_category(
        category=CATEGORY_ID,
        api_host=API_HOST,
        server_token=SERVER_TOKEN,
        evaluation_types=EVALUATION_TYPES,
    )

    datasets: list[LMHDataset | LMHDatasetV2] = []
    for dataset_id in datasets_ids:
        # Download and export each dataset
        returned_data = download_dataset_from_gcs(
            dataset_id=dataset_id, directory=".temp"
        )
        dataset = (
            LMHDataset(dataset_id, directory=".temp")
            if "json" in returned_data
            else LMHDatasetV2(dataset_id, directory=".temp")
        )
        dataset.export()
        datasets.append(dataset)

    # Organize datasets by category and task
    categories: dict[str, dict[str, list[LMHDataset | LMHDatasetV2]]] = {}
    for dataset in datasets:
        if dataset.category_id:
            if categories.get(dataset.category_id) is None:
                categories[dataset.category_id] = {}
            if categories[dataset.category_id].get(str(dataset.task_id)) is None:
                categories[dataset.category_id][str(dataset.task_id)] = []
            categories[dataset.category_id][dataset.task_id].append(dataset)

    # check if any categories are empty
    for category in categories:
        if len(categories[category]) == 0:
            del categories[category]

    # check if any tasks are empty
    for category in categories:
        for task in categories[category]:
            if len(categories[category][task]) == 0:
                del categories[category][task]

    print(f"Total categories: {len(categories)}")
    print(categories)

    # Build model args with processed base_url
    model_args = {"model": MODEL_NAME}
    if processed_base_url:
        model_args["base_url"] = processed_base_url
    if API_KEY:
        model_args["api_key"] = API_KEY

    # Run evaluation job per task
    for category, tasks in categories.items():
        print(f"Running evaluation for category: {category}")
        print(f"Total tasks: {len(datasets)}")

        for task, _datasets in tasks.items():
            if len(_datasets) == 0:
                continue
            job = EvaluationJob(
                tasks=[dataset.name for dataset in _datasets],
                adapter=processed_adapter,  # Use processed adapter
                model_args=model_args,
                task_id=task,
                job_id=JOB_ID,
                api_host=API_HOST,
                server_token=SERVER_TOKEN,
                category_name=category,
                benchmark_id=BENCHMARK_ID,
                llm_judge_api_key=LLM_JUDGE_API_KEY,
                llm_judge_model=LLM_JUDGE,
                llm_judge_provider=LLM_JUDGE_PROVIDER,
            )
            job()
