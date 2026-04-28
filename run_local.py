"""Entry point for the evaluation job."""

import os
import json
import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from src.db_operations import submit_model_evaluation
from src.evaluation import EvaluationJob
from src.task_v2 import LMHDataset as LMHDatasetV2
from src.task import LMHDataset
from src.adapter_utils import process_adapter_and_url

# Load environment variables
load_dotenv()

# Constants from .env
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL")
ADAPTER = os.getenv("ADAPTER")
SERVER_TOKEN = os.getenv("SERVER_TOKEN")
API_HOST = os.getenv("API_HOST")
USER_ID = os.getenv("USER_ID")
BENCHMARK_ID = os.getenv("BENCHMARK_ID")
EVALUATION_TYPES = os.getenv("EVALUATION_TYPES")
LLM_JUDGE = os.getenv("JUDGE_MODEL")
LLM_JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER")
LLM_JUDGE_API_KEY = os.getenv("JUDGE_API_KEY")
PARALLEL_CATEGORIES = os.getenv("PARALLEL_CATEGORIES", "false").lower() in ("true", "1", "yes", "on")


# Validation
if not MODEL_NAME:
    raise ValueError("Model name is required")
if not ADAPTER:
    raise ValueError("Adapter is required")

# Set API key environment variables based on adapter type
if API_KEY:
    if ADAPTER == "openai-chat-completions" or ADAPTER == "local-chat-completions":
        os.environ["OPENAI_API_KEY"] = API_KEY
    elif ADAPTER == "anthropic-chat-completions":
        os.environ["ANTHROPIC_API_KEY"] = API_KEY
    elif ADAPTER == "gemini":
        os.environ["GOOGLE_API_KEY"] = API_KEY
    elif ADAPTER == "groq":
        os.environ["GROQ_API_KEY"] = API_KEY
    elif ADAPTER == "humain":
        os.environ["HUMAIN_API_KEY"] = API_KEY

# Directories
TEMP_DIR = ".temp"
TASKS_DIR = ".tasks"
RESULTS_DIR = ".results"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Copy multimodal_utils.py to .temp directory for lm_eval to find it
multimodal_utils_src = "src/multimodal_utils.py"
multimodal_utils_dst = os.path.join(TEMP_DIR, "multimodal_utils.py")
if os.path.exists(multimodal_utils_src):
    shutil.copy2(multimodal_utils_src, multimodal_utils_dst)
    print(f"Copied multimodal_utils.py to {multimodal_utils_dst}")
else:
    print(f"Warning: {multimodal_utils_src} not found")

# Helper function to copy images if needed
def copy_images_to_temp(json_file_path: str, temp_dir: str):
    """Copy images referenced in JSON file to temp directory."""
    from pathlib import Path

    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}

    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    images_dir = os.path.join(temp_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    for item in data:
        if 'images' in item and isinstance(item['images'], list):
            for img_path in item['images']:
                if os.path.exists(img_path):
                    img_name = os.path.basename(img_path)
                    dst_path = os.path.join(images_dir, img_name)
                    if not os.path.exists(dst_path):
                        shutil.copy2(img_path, dst_path)
                        print(f"Copied image: {img_name} -> {dst_path}")
                    # Update path in item to relative path
                    item['images'] = [dst_path if p == img_path else p for p in item['images']]

    # Write back updated JSON
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    os.environ["ENV"] = "local"  # setup local env for logging to stdout

    # Process adapter and base_url using shared utility
    processed_adapter, processed_base_url = process_adapter_and_url(ADAPTER, BASE_URL)

    # Read the tasks from directory
    tasks_temp: dict = dict()
    task_mapper: dict = dict()
    print(f"Reading tasks from directory '{TASKS_DIR}'")

    for file in os.listdir(f"./{TASKS_DIR}"):
        if not file.endswith("json"):
            continue
        # Open the file and extract content
        with open(f"./{TASKS_DIR}/{file}", "r", encoding="utf-8") as f:
            content = f.read()
            d = json.loads(content)
            
            # # Skip datasets with no metric
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

            # Initialize LMHDataset
            dataset = LMHDatasetV2(str(file.rsplit(".", 1)[0]), TEMP_DIR) if "json" not in d else LMHDataset(str(file.rsplit(".", 1)[0]), TEMP_DIR)
            dataset.export()

            # Copy images to .temp directory if dataset contains images
            for split in ["test", "dev"]:
                json_file = os.path.join(TEMP_DIR, f"{dataset.file_name}_{split}.json")
                if os.path.exists(json_file):
                    copy_images_to_temp(json_file, TEMP_DIR)

            if tasks_temp.get(d["category"]) is None:
                tasks_temp[d["category"]] = []
            tasks_temp[d["category"]].append(dataset.name)

    # Model arguments with processed base_url
    model_args = {"model": MODEL_NAME}
    if processed_base_url:
        model_args["base_url"] = processed_base_url
    if API_KEY:
        model_args["api_key"] = API_KEY


    # Initialize a model evaluation
    if (
        ADAPTER
        and SERVER_TOKEN
        and API_HOST
        and USER_ID
        and BENCHMARK_ID
        and tasks_temp
    ):
        submit_results = submit_model_evaluation(
            model_name=MODEL_NAME,
            model_url=processed_base_url,  # Use processed base_url
            adapter=processed_adapter,  # Use processed adapter
            api_key=API_KEY,
            categories=list(tasks_temp.keys()),
            server_token=SERVER_TOKEN,
            api_host=API_HOST,
            user_id=USER_ID,
            benchmark_id=BENCHMARK_ID,
            evaluation_types=EVALUATION_TYPES.split(",") if EVALUATION_TYPES else [],
        )
        print(submit_results)
        if submit_results["status_code"] != 200:
            raise Exception(f"[ERROR] Failed to submit evaluation: {submit_results}")

    else:
        submit_results = {"jobs_ids": {}}

    def run_category(category, datasets):
        """Run evaluation for a single category."""
        print(f"Running evaluation for category: {category}")

        try:
            job = EvaluationJob(
                tasks=datasets,
                adapter=processed_adapter,  # Use processed adapter
                model_args=model_args,
                tasks_mapper_dict=task_mapper,
                category_name=category,
                job_id=submit_results["jobs_ids"].get(category, None),
                llm_judge_api_key=LLM_JUDGE_API_KEY,
                llm_judge_model=LLM_JUDGE,
                llm_judge_provider=LLM_JUDGE_PROVIDER,
                server_token=SERVER_TOKEN,
                api_host=API_HOST,
                benchmark_id=BENCHMARK_ID,
            )
            job()
            return (category, True, None)
        except Exception as e:
            print(f"An error occurred while running the job for task {category}: {e}")
            return (category, False, str(e))

    # Run categories sequentially or in parallel
    if PARALLEL_CATEGORIES and len(tasks_temp) > 1:
        print(f"Running {len(tasks_temp)} categories in parallel...")
        with ThreadPoolExecutor(max_workers=min(len(tasks_temp), os.cpu_count() or 4)) as executor:
            # Submit all category jobs
            future_to_category = {
                executor.submit(run_category, category, datasets): category
                for category, datasets in tasks_temp.items()
            }

            # Process completed jobs
            for future in as_completed(future_to_category):
                category, success, error = future.result()
                if success:
                    print(f"Category '{category}' completed successfully")
                else:
                    print(f"Category '{category}' failed with error: {error}")
    else:
        # Sequential execution (default behavior)
        for category, datasets in tasks_temp.items():
            run_category(category, datasets)
