"""Entry point for the evaluation job."""

from typing import List, Dict, Any
import requests
import os
import json
from dotenv import load_dotenv

from src.evaluation import EvaluatationJob
from src.task import LMHDataset


load_dotenv()

TEMP_DIR = ".temp"
TASKS_DIR = ".tasks"
os.makedirs(TEMP_DIR, exist_ok=True)


def submit_model_evaluation(
    model_name: str,
    model_url: str,
    adapter: str,
    api_key: str,
    category: str,
    tasks: List[str],
) -> Dict[str, Any]:
    """
    Submit model for evaluation

    Args:
        model_name: Name of the model
        model_url: URL endpoint for the model
        adapter: Type of adapter to use
        api_key: API key for the model
        category: Evaluation category
        tasks: List of tasks for evaluation

    Returns:
        Dictionary containing response data and status
    """

    server_token = os.environ.get("SERVER_TOKEN")
    if not server_token:
        raise ValueError("SERVER_TOKEN environment variable is not set")

    api_host = os.environ.get("API_HOST")
    if not api_host:
        raise ValueError("API_HOST environment variable is not set")

    user_id = os.environ.get("USER_ID")
    if not user_id:
        raise ValueError("USER_ID environment variable is not set")

    # Headers
    headers = {
        "Content-Type": "application/json",
        "x-server-token": server_token,
    }

    # Request payload
    data = {
        "modelName": model_name,
        "modelUnique": model_name,
        "modelUrl": model_url,
        "adapter": adapter,
        "apiKey": api_key,
        "category": category,
        "tasks": tasks,
        "userId": user_id,
    }

    try:
        response = requests.post(
            f"https://{api_host}/api/local/evulation/submit",
            headers=headers,
            json=data
        )

        result = {
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "raw_response": response.text
        }

        # Try to parse JSON response
        try:
            json_data = response.json()
            result["data"] = json_data

            # Extract specific fields from successful response
            if result["success"] and json_data.get("code") == "OK":
                result["model_id"] = json_data.get("data", {}).get("modelId")
                result["evaluation_id"] = json_data.get(
                    "data", {}).get("evaluationId")
                result["job_id"] = json_data.get("data", {}).get("jobId")
                result["message"] = json_data.get("data", {}).get("message")

        except json.JSONDecodeError:
            result["data"] = None

        return result

    except requests.exceptions.RequestException as e:
        return {
            "status_code": None,
            "success": False,
            "error": str(e),
            "raw_response": None,
            "data": None
        }


if __name__ == "__main__":
    # Read the tasks from directory "tasks"
    tasks_temp: dict[str, list[str]] = dict()
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
            ) as f:
                d["json"]["category"] = d["category"]
                d["json"]["task"] = d["task"]
                json.dump(d["json"], f, ensure_ascii=False)

            # Initialize LMHDataset
            dataset = LMHDataset(str(file.split(".")[0]), TEMP_DIR)
            dataset.export()

            if tasks_temp.get(d["category"]) is None:
                tasks_temp[d["category"]] = {}
            if tasks_temp[d["category"]].get(d["task"]) is None:
                tasks_temp[d["category"]][d["task"]] = []
            tasks_temp[d["category"]][d["task"]].append(dataset.name)

    # Get model arguments from environment variables
    model_args = {}
    base_url = os.getenv("BASE_URL")
    if base_url:
        model_args["base_url"] = base_url
    api_key = os.getenv("API_KEY")
    if api_key:
        model_args["api_key"] = api_key
        os.environ["OPENAI_API_KEY"] = api_key
    max_tokens = os.getenv("MAX_TOKENS")
    if max_tokens:
        model_args["max_tokens"] = int(max_tokens)
    temperature = os.getenv("TEMPERATURE")
    if temperature:
        model_args["temperature"] = float(temperature)
    model_name = os.getenv("MODEL")
    if model_name:
        model_args["model"] = model_name
    else:
        raise ValueError("Model name is required")

    adapter = os.environ["ADAPTER"]

    # Initialize the evaluation job
    for category, tasks in tasks_temp.items():
        print("Running evaluation for category:", category)
        print("Total tasks:", len(tasks))

        for task, _datasets in tasks.items():
            try:
                submit_results = submit_model_evaluation(
                    model_name=model_name,
                    model_url=base_url,
                    adapter=adapter,
                    api_key=api_key,
                    category=category,
                    tasks=[task],
                )
                if not submit_results["success"]:
                    print(
                        f"Failed to submit evaluation job for task {task}: {submit_results['error']}")
                    continue

                job = EvaluatationJob(
                    tasks=_datasets,
                    adapter=adapter,
                    task_id=task,
                    output_path=str(task),
                    model_args=model_args,
                    job_id=submit_results["job_id"],
                )
                job.run()
            except Exception as e:
                print(
                    f"An error occurred while running the job for task {task}: {e}")
                continue
