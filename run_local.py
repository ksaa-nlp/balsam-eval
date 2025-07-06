"""Entry point for the evaluation job."""

import os
import json
from dotenv import load_dotenv

from src.db_operations import submit_model_evaluation
from src.evaluation import EvaluatationJob
from src.task import LMHDataset

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
LLM_JUDGE = os.getenv("JUDGE_MODEL")
LLM_JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER")
LLM_JUDGE_API_KEY = os.getenv("JUDGE_API_KEY")


# Validation
if not MODEL_NAME:
    raise ValueError("Model name is required")
if not ADAPTER:
    raise ValueError("Adapter is required")

# Derived constants
if API_KEY:
    os.environ["OPENAI_API_KEY"] = API_KEY

# Directories
TEMP_DIR = ".temp"
TASKS_DIR = ".tasks"
os.makedirs(TEMP_DIR, exist_ok=True)

# Control flag
NEED_CREATE_EVALUATION_JOB = False


if __name__ == "__main__":
    # Read the tasks from directory
    tasks_temp: dict = dict()
    print(f"Reading tasks from directory '{TASKS_DIR}'")

    for file in os.listdir(f"./{TASKS_DIR}"):
        if not file.endswith("json"):
            continue
        # Open the file and extract content
        with open(f"./{TASKS_DIR}/{file}", "r", encoding="utf-8") as f:
            content = f.read()
            d = json.loads(content)

            # Skip datasets with no metric
            if d["json"]["metric_list"][0]["metric"] == "":
                continue

            with open(
                f"./{TEMP_DIR}/{file.split('.')[0]}.json", "w", encoding="utf-8"
            ) as f_out:
                d["json"]["category"] = d["category"]
                d["json"]["task"] = d["task"]
                json.dump(d["json"], f_out, ensure_ascii=False)

            # Initialize LMHDataset
            dataset = LMHDataset(str(file.split(".")[0]), TEMP_DIR)
            dataset.export()

            if tasks_temp.get(d["category"]) is None:
                tasks_temp[d["category"]] = {}
            if tasks_temp[d["category"]].get(d["task"]) is None:
                tasks_temp[d["category"]][d["task"]] = []
            tasks_temp[d["category"]][d["task"]].append(dataset.name)

    # Model arguments
    model_args = {"model": MODEL_NAME}
    if BASE_URL:
        model_args["base_url"] = BASE_URL
    if API_KEY:
        model_args["api_key"] = API_KEY

    # Initialize evaluation job
    for category, tasks in tasks_temp.items():
        print("Running evaluation for category:", category)
        print("Total tasks:", len(tasks))

        if NEED_CREATE_EVALUATION_JOB and BASE_URL and API_KEY and ADAPTER and SERVER_TOKEN and API_HOST and USER_ID:
            submit_results = submit_model_evaluation(
                model_name=MODEL_NAME,
                model_url=BASE_URL,
                adapter=ADAPTER,
                api_key=API_KEY,
                category=category,
                server_token=SERVER_TOKEN,
                api_host=API_HOST,
                user_id=USER_ID,
            )
            if not submit_results["success"]:
                print(
                    f"Failed to submit evaluation: {submit_results['error']}")
                continue
        else:
            submit_results = {"job_id": None}

        for task, _datasets in tasks.items():
            try:
                job = EvaluatationJob(
                    tasks=_datasets,
                    adapter=ADAPTER,
                    task_id=task,
                    output_path=str(task),
                    model_args=model_args,
                    job_id=submit_results["job_id"],
                    llm_judge_api_key=LLM_JUDGE_API_KEY,
                    llm_judge_model=LLM_JUDGE,
                    llm_judge_provider=LLM_JUDGE_PROVIDER,
                )
                job.run()
            except Exception as e:
                print(
                    f"An error occurred while running the job for task {task}: {e}")
                continue
