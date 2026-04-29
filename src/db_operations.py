"""Database operations and API interactions."""

import json
import time
from enum import Enum
from typing import Any, Dict, List, Optional, cast

import requests


class JobStatus(Enum):
    """Job status enumeration."""

    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def sanitize_text(text: str) -> str:
    """Sanitize text for use as an identifier.

    Args:
        text: Text to sanitize

    Returns:
        Sanitized text
    """
    return text.strip().lower().replace(" ", "_").replace("-", "_")


def _make_request_with_retry(
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    json_data: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
    initial_timeout: float = 30.0,
    max_timeout: float = 120.0,
) -> requests.Response:
    """Make HTTP request with retry logic.

    Args:
        method: HTTP method
        url: Request URL
        headers: Optional headers
        json_data: Optional JSON data
        params: Optional query parameters
        max_retries: Maximum number of retries
        initial_timeout: Initial timeout in seconds
        max_timeout: Maximum timeout in seconds

    Returns:
        Response object

    Raises:
        requests.RequestException: If all retries fail
    """
    last_exception = None
    timeout: float = initial_timeout

    for attempt in range(max_retries):
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=json_data,
                params=params,
                timeout=timeout,
            )
            return response

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = min(2**attempt, 30)
                timeout = min(timeout * 1.5, max_timeout)
                print(
                    f"⚠️  Connection failed (attempt {attempt + 1}/{max_retries}). "
                    f"Retrying in {wait_time}s with timeout={timeout:.1f}s..."
                )
                time.sleep(wait_time)
            # On last attempt, loop ends and falls through to raise below

        except requests.RequestException:  # pylint: disable=try-except-raise
            raise  # Non-retryable errors bubble up immediately

    print(f"❌ All {max_retries} retry attempts failed.")
    raise requests.RequestException(
        f"Failed to connect to {url} after {max_retries} attempts. "
        f"Last error: {last_exception}"
    ) from last_exception


def map_alias_to_task(task_alias: str, is_sanitized: bool = False) -> str:
    """Map task alias to task name.

    Args:
        task_alias: Task alias to map
        is_sanitized: Whether to sanitize the output

    Returns:
        Mapped task name

    Raises:
        Exception: If mapping fails
    """
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
    *,
    server_token: str,
    api_host: str,
    user_id: str,
    benchmark_id: str,
    evaluation_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Submit model evaluation to server.

    Args:
        model_name: Name of the model
        model_url: URL of the model
        adapter: Adapter type
        api_key: API key for the model
        categories: List of categories to evaluate
        server_token: Server authentication token
        api_host: API host URL
        user_id: User ID
        benchmark_id: Benchmark ID
        evaluation_types: Optional list of evaluation types

    Returns:
        Response data from server
    """
    if evaluation_types is None:
        evaluation_types = []
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
        "benchmarkId": benchmark_id,
        # "evaluationTypeValues": evaluation_types,
    }
    if evaluation_types:
        data["evaluationTypeValues"] = evaluation_types

    try:
        response = _make_request_with_retry(
            method="POST",
            url=f"{api_host}/api/local/evulation/submit",
            headers=headers,
            json_data=data,
            max_retries=3,
            initial_timeout=30,
            max_timeout=120,
        )

        result = {
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "raw_response": response.text,
        }

        try:
            json_data = response.json()
            result["data"] = json_data

            if result["success"] and json_data.get("code") == "OK":
                result["model_id"] = json_data.get("data", {}).get("modelId")
                result["evaluation_id"] = json_data.get("data", {}).get("evaluationId")
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
            "data": None,
        }


def get_avrage_scores(result: dict[str, Any]) -> dict[str, Any]:
    """
    Extract scores from result dict for database storage.

    This function handles the score keys as they are stored in the results,
    without changing their names since the system depends on these names.
    """
    final_results = {}

    print(f"[DEBUG] get_avrage_scores called with result keys: {result.keys()}")

    # Handle ROUGE metric (returns a dictionary with multiple ROUGE scores)
    if "rouge,none" in result:
        rouge_result = result["rouge,none"]
        print(
            f"[DEBUG] Found rouge,none: type={type(rouge_result)}, value={rouge_result}"
        )
        # If rouge_result is a dict, extract rougeLsum; otherwise use the value directly
        if isinstance(rouge_result, dict):
            # Only use rougeLsum as the representative score
            extracted_value = rouge_result.get("rougeLsum", 0)
            print(f"[DEBUG] Extracted rougeLsum: {extracted_value}")
            final_results["nGramScore"] = extracted_value
        else:
            final_results["nGramScore"] = rouge_result
            print(f"[DEBUG] Using rouge_result directly: {final_results["nGramScore"]}")
    # Handle BLEU metric (returns a float)
    elif "bleu,none" in result:
        final_results["nGramScore"] = result["bleu,none"]
        print(f"[DEBUG] Found bleu,none: {final_results["nGramScore"]}")

    # Handle accuracy metric
    if "accuracy,none" in result:
        final_results["mcqScore"] = result["accuracy,none"]
        print(f"[DEBUG] Found accuracy,none: {final_results["mcqScore"]}")

    # Handle LLM as judge metrics (dict format with average_score)
    # These are set by process_results_with_llm_judge for task-level summaries
    if "llm_as_judge" in result:
        final_results["llmAsJudgeScore"] = result["llm_as_judge"].get(
            "average_score", 0
        )
        print(f"[DEBUG] Found llm_as_judge: {final_results["llmAsJudgeScore"]}")

    if "mcq_llm_as_judge" in result:
        final_results["MCQllmAsJudgeScore"] = result["mcq_llm_as_judge"].get(
            "average_score", 0
        )
        print(f"[DEBUG] Found mcq_llm_as_judge: {final_results["MCQllmAsJudgeScore"]}")

    # Handle LLM judge scores in numeric format (set by process_results_with_llm_judge)
    # These are the actual score values (not dict summaries)
    if "llm_judge_score,none" in result:
        value = result["llm_judge_score,none"]
        if isinstance(value, (int, float)):
            # Only set if not already set by dict format (prefer dict format if available)
            if "llmAsJudgeScore" not in final_results:
                final_results["llmAsJudgeScore"] = float(value)
                print(
                    f"[DEBUG] Found llm_judge_score,none: {final_results["llmAsJudgeScore"]}"
                )

    if "mcq_llm_judge_score,none" in result:
        value = result["mcq_llm_judge_score,none"]
        if isinstance(value, (int, float)):
            # Only set if not already set by dict format (prefer dict format if available)
            if "MCQllmAsJudgeScore" not in final_results:
                final_results["MCQllmAsJudgeScore"] = float(value)
                print(
                    f"[DEBUG] Found mcq_llm_judge_score,none: {final_results["MCQllmAsJudgeScore"]}"
                )

    print(f"[DEBUG] Final results: {final_results}")
    return final_results


def add_results_to_db(
    api_host: str,
    job_id: str,
    task_id: str,
    server_token: str,
    result: dict[str, Any],
    *,
    category_name: str,
    benchmark_id: str,
) -> None:
    """Add evaluation results to database via webhook.

    Args:
        api_host: API host URL
        job_id: Job ID
        task_id: Task ID
        server_token: Server authentication token
        result: Evaluation results dictionary
        category_name: Category name
        benchmark_id: Benchmark ID

    Raises:
        RuntimeError: If posting results fails
    """
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
        "benchmarkId": benchmark_id,
    }

    webhook_url = f"{api_host}/api/webhook/job"
    try:
        response = _make_request_with_retry(
            method="POST",
            url=webhook_url,
            headers={
                "Content-Type": "application/json",
                "x-server-token": server_token,
            },
            json_data=payload,
            max_retries=3,
            initial_timeout=30,
            max_timeout=120,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"❌ Failed to post job results (job: {job_id}, task: {task_id}): {e}")
        raise RuntimeError(f"Failed to post job results: {e}") from e


def update_status(
    api_host: str,
    job_id: str,
    server_token: str,
    status: JobStatus,
    error_message: Optional[str] = None,
) -> None:
    """Update job status via webhook.

    Args:
        api_host: API host URL
        job_id: Job ID
        server_token: Server authentication token
        status: New job status
        error_message: Optional error message

    Raises:
        RuntimeError: If status update fails
    """
    if not api_host:
        return

    payload = {
        "id": job_id,
        "status": status.value,
        "error": error_message or "",
    }

    webhook_url = f"{api_host}/api/webhook/job"
    try:
        response = _make_request_with_retry(
            method="POST",
            url=webhook_url,
            headers={
                "Content-Type": "application/json",
                "x-server-token": server_token,
            },
            json_data=payload,
            max_retries=3,
            initial_timeout=30,
            max_timeout=120,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"❌ Failed to update job status for job {job_id}: {e}")
        raise RuntimeError(f"Failed to update job status: {e}") from e


def get_tasks_from_category(
    category: str,
    api_host: str,
    server_token: str,
    evaluation_types: Optional[str] = None,
) -> list[str]:
    if not category:
        raise ValueError("Category is required, terminating the process.")

    webhook_url = f"{api_host}/api/tasks/{category}"
    if evaluation_types:
        types_param = "&types=".join(evaluation_types.split(","))
        webhook_url += f"?types={types_param}"
    else:
        webhook_url += "?types=generation"

    response = _make_request_with_retry(
        method="GET",
        url=webhook_url,
        headers={
            "Content-Type": "application/json",
            "x-server-token": server_token,
        },
        max_retries=3,
        initial_timeout=30,
        max_timeout=120,
    )
    response.raise_for_status()

    if response.status_code != 200:
        print(f"Failed to retrieve tasks for category {category}")
        raise ValueError("Failed to retrieve tasks for category")

    # Make sure we have the datasets
    response_data = response.json()
    if "datasets" not in response_data:
        print(f"No datasets found for category {category}")
        raise ValueError("No datasets found for category")

    return cast(list[str], response_data["datasets"])
