from enum import Enum
import json
from typing import Any, Dict, List, Optional
import requests


class JobStatus(Enum):
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def sanitize_text(text: str) -> str:
    return text.strip().lower().replace(" ", "_").replace("-", "_")


def map_alias_to_task(task_alias: str, is_sanitized: bool = False) -> str:
    try:
        with open("tasks_mappping.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return sanitize_text(data[task_alias]) if is_sanitized else data[task_alias]
    except Exception as e:
        print(f"❌ Error mapping alias '{task_alias}': {e}")
        raise


def submit_model_evaluation(
    model_name: str,
    model_url: str,
    adapter: str,
    api_key: str,
    categories: List[str],
    server_token: str,
    api_host: str,
    user_id: str,
    benchmark_id: str
) -> Dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "x-server-token": server_token,
    }

    data = {
        "modelName": model_name,
        "modelUnique": model_name,
        "modelUrl": model_url,
        "adapter": adapter,
        "apiKey": api_key,
        "categories": categories,
        "userId": user_id,
        "benchmarkId": benchmark_id
    }

    try:
        response = requests.post(
            f"{api_host}/api/local/evulation/submit",
            headers=headers,
            json=data
        )

        result = {
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "raw_response": response.text
        }

        try:
            json_data = response.json()
            result["data"] = json_data

            if result["success"] and json_data.get("code") == "OK":
                result["model_id"] = json_data.get("data", {}).get("modelId")
                result["evaluation_id"] = json_data.get(
                    "data", {}).get("evaluationId")
                result["jobs_ids"] = json_data.get("data", {}).get("jobsIds")
                result["message"] = json_data.get("data", {}).get("message")

        except json.JSONDecodeError:
            print(f"❌ JSON decode error from response for model {model_name}")
            result["data"] = None

        return result

    except requests.exceptions.RequestException as e:
        print(f"❌ RequestException submitting model '{model_name}': {e}")
        return {
            "status_code": None,
            "success": False,
            "error": str(e),
            "raw_response": None,
            "data": None
        }


def get_avrage_scores(result: dict[str, Any]) -> dict[str, Any]:
    final_results = {}
    if "rouge,none" in result:
        final_results["nGramScore"] = result["rouge,none"].get("rougeLsum", 0)
    elif "bleu,none" in result:
        final_results["nGramScore"] = result["bleu,none"]

    if "accuracy,none" in result:
        final_results["mcqScore"] = result["accuracy,none"]

    if "llm_as_judge" in result:
        final_results["llmAsJudgeScore"] = result["llm_as_judge"].get(
            "average_score", 0)

    return final_results


def add_results_to_db(
    api_host: str,
    job_id: str,
    task_id: str,
    server_token: str,
    result: dict[str, Any],
    category_name: str,
    benchmark_id: str,
) -> None:
    if not api_host:
        return

    final_scores = get_avrage_scores(result)

    payload = {
        "id": job_id,
        "taskId": task_id,
        "results": json.dumps(result, ensure_ascii=False),
        "status": JobStatus.COMPLETED.value,
        "scores": json.dumps(final_scores, ensure_ascii=False),
        "categoryName": category_name,
        "benchmarkId": benchmark_id
    }

    webhook_url = f"{api_host}/api/webhook/job"
    try:
        response = requests.post(
            webhook_url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "x-server-token": server_token,
            },
            timeout=20,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        print(
            f"❌ Failed to post job results (job: {job_id}, task: {task_id}): {e}")
        raise RuntimeError(f"Failed to post job results: {e}") from e


def update_status(
    api_host: str,
    job_id: str,
    server_token: str,
    status: JobStatus,
    error_message: Optional[str] = None
) -> None:
    if not api_host:
        return

    payload = {
        "id": job_id,
        "status": status.value,
        "error": error_message or "",
    }

    webhook_url = f"{api_host}/api/webhook/job"
    try:
        response = requests.post(
            webhook_url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "x-server-token": server_token,
            },
            timeout=20,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"❌ Failed to update job status for job {job_id}: {e}")
        raise RuntimeError(f"Failed to update job status: {e}") from e


def get_tasks_from_category(category: str, api_host: str, server_token: str,metric_type:Optional[str]=None) -> list[str]:
    if not category:
        raise ValueError("Category is required, terminating the process.")

    webhook_url = f"{api_host}/api/tasks/{category}"
    if metric_type:
        webhook_url += f"?type={metric_type}"
    response = requests.get(
        webhook_url,
        headers={
            "Content-Type": "application/json",
            "x-server-token": server_token,
        },
        timeout=20,
    )
    response.raise_for_status()

    if response.status_code != 200:
        print(f"Failed to retrieve tasks for category {category}")
        raise ValueError("Failed to retrieve tasks for category")

    # Make sure we have the datasets
    if "datasets" not in response.json():
        print(f"No datasets found for category {category}")
        raise ValueError("No datasets found for category")

    return response.json()["datasets"]
