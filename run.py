"""Entry point for the evaluation job."""

import logging
import os

from dotenv import load_dotenv
from src.db_operations import get_tasks_from_category
from src.evaluation import EvaluatationJob
from src.task import LMHDataset, download_dataset_from_gcs

# Load environment variables
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Constants from environment
API_HOST = os.getenv("API_HOST")
SERVER_TOKEN = os.getenv("SERVER_TOKEN")
CATEGORY = os.getenv("CATEGORY")
BENCHMARK_ID = os.getenv("BENCHMARK_ID")
ADAPTER = os.getenv("ADAPTER")
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MAX_TOKENS = os.getenv("MAX_TOKENS")
TEMPERATURE = os.getenv("TEMPERATURE")
MODEL_NAME = os.getenv("MODEL")
JOB_ID = os.getenv("JOB_ID")
LLM_JUDGE = os.getenv("JUDGE_MODEL")
LLM_JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER")
LLM_JUDGE_API_KEY = os.getenv("JUDGE_API_KEY")


# Validation
if not all([API_HOST, SERVER_TOKEN, CATEGORY, ADAPTER, BENCHMARK_ID]):
    raise ValueError(
        "API_HOST, SERVER_TOKEN, CATEGORY, BENCHMARK_ID, and ADAPTER environment variables are required")
if not MODEL_NAME:
    raise ValueError("MODEL name is required")

if API_KEY:
    os.environ["OPENAI_API_KEY"] = API_KEY

# Collect datasets
if __name__ == "__main__":
    if not CATEGORY or not API_HOST or not SERVER_TOKEN:
        raise ValueError("CATEGORY, API_HOST, and SERVER_TOKEN must be set.")
    datasets_ids = get_tasks_from_category(
        category=CATEGORY, api_host=API_HOST, server_token=SERVER_TOKEN
    )

    datasets: list[LMHDataset] = []
    for dataset_id in datasets_ids:
        # Download and export each dataset
        download_dataset_from_gcs(dataset_id=dataset_id, directory=".temp")
        dataset = LMHDataset(dataset_id, directory=".temp")
        dataset.export()
        datasets.append(dataset)

    # Organize datasets by category and task
    categories: dict[str, dict[str, list[LMHDataset]]] = {}
    for dataset in datasets:
        if dataset.category_id:
            categories.setdefault(dataset.category_id, {})
            categories[dataset.category_id].setdefault(
                str(dataset.task_id), [])
            categories[dataset.category_id][str(
                dataset.task_id)].append(dataset)

    print(f"Total categories: {len(categories)}")
    print(categories)

    # Build model args
    model_args = {"model": MODEL_NAME}
    if BASE_URL:
        model_args["base_url"] = BASE_URL
    if API_KEY:
        model_args["api_key"] = API_KEY

    # Run evaluation job per task
    for category, tasks in categories.items():
        print(f"Running evaluation for category: {category}")
        print(f"Total tasks: {len(datasets)}")

        for task, _datasets in tasks.items():
            job = EvaluatationJob(
                tasks=[dataset.name for dataset in _datasets],
                adapter=ADAPTER,
                model_args=model_args,
                task_id=task,
                output_path=str(task),
                job_id=JOB_ID,
                api_host=API_HOST,
                server_token=SERVER_TOKEN,
                category_name=CATEGORY,
                benchmark_id=BENCHMARK_ID,
                llm_judge_api_key=LLM_JUDGE_API_KEY,
                llm_judge_model=LLM_JUDGE,
                llm_judge_provider=LLM_JUDGE_PROVIDER,
            )
            job.run()
