import os
import json
import traceback
import uuid
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

# model name and user id mapping
model_access_map = {
    "C4AI Aya Expanse 32B": "0m4lv2jy91yzon7",
    "Claude 3.5 Sonnet": "djsek471pqdxitt",
    "Databricks DBRX": "djsek471pqdxitt",
    "AceGPT-v2 8B Chat": "0m4lv2jy91yzon7",
    "Yehia 7B Preview": "0m4lv2jy91yzon7",
    "ALLaM-7B Instruct Preview": "0m4lv2jy91yzon7",
    "GPT-4o": "djsek471pqdxitt",
    "Amazon Titan Nova Pro": "djsek471pqdxitt",
    "Command R+": "0m4lv2jy91yzon7",
    "Command R 7B (2024-12)": "0m4lv2jy91yzon7",
    "JAIS 13B Chat": "djsek471pqdxitt",
    "Iron Horse Gamma Velorum V5a": "djsek471pqdxitt",
    "DeepSeek-V3": "0m4lv2jy91yzon7",
    "Gemma 2 9B": "0m4lv2jy91yzon7",
    "Qwen 2.5 32B": "0m4lv2jy91yzon7",
    "Gemini 2.0 Flash": "djsek471pqdxitt",
    "Mistral Large (72B)": "djsek471pqdxitt",
    "AraGPT2-Mega": "0m4lv2jy91yzon7",
    "SILMA 9B Instruct v1.0": "0m4lv2jy91yzon7",
    "Mistral SABA": "djsek471pqdxitt",
    "Grok-2": "djsek471pqdxitt"
}


# Load environment variables from .env file
load_dotenv()


def create_model_if_not_exists(conn, model, model_name, model_url, adapter, user_id):
    """Create a model if it doesn't exist and return its ID."""
    try:
        with conn.cursor() as cursor:
            # Check if model exists
            cursor.execute(
                "SELECT id FROM public.model WHERE name = %s", (model_name,))
            result = cursor.fetchone()

            if result:
                return result[0]

            # Create model if it doesn't exist
            model_id = model_name.lower().replace(" ", "_").replace(".", "_")
            cursor.execute(
                """
                INSERT INTO public.model
                (id, model, user_id, name, model_url, adapter, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
                RETURNING id
                """,
                (model_id, model, user_id, model_name, model_url, adapter)
            )
            conn.commit()
            return model_id
    except Exception as e:
        print(f"Error creating model '{model_name}': {e}")
        conn.rollback()
        return None


def create_category_if_not_exists(conn, category_name, category_name_ar):
    """Create a category if it doesn't exist and return its ID."""
    try:
        with conn.cursor() as cursor:
            # Convert category name to snake_case for ID
            category_id = category_name.lower().replace(" ", "_").replace("-", "_")

            # Check if category exists by ID
            cursor.execute(
                "SELECT id FROM public.category WHERE id = %s", (category_id,))
            result = cursor.fetchone()

            if result:
                return result[0]

            # Create category if it doesn't exist
            cursor.execute(
                """
                INSERT INTO public.category
                (id, name, name_ar)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (category_id, category_name, category_name_ar)
            )
            conn.commit()
            return category_id
    except Exception as e:
        print(f"Error creating category '{category_name}': {e}")
        conn.rollback()
        return None


def create_task_if_not_exists(conn, task_name, task_name_ar, category_id, user_id):
    """Create a task if it doesn't exist and return its ID."""
    try:
        with conn.cursor() as cursor:
            # Convert task name to snake_case for ID
            task_id = task_name.lower().replace(" ", "_").replace("-", "_")

            # Check if task exists by ID
            cursor.execute(
                "SELECT id FROM public.task WHERE id = %s", (task_id,))
            result = cursor.fetchone()

            if result:
                return result[0]

            # Create task if it doesn't exist
            cursor.execute(
                """
                INSERT INTO public.task
                (id, category_id, user_id, name, name_ar, status, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
                RETURNING id
                """,
                (task_id, category_id, user_id, task_name, task_name_ar, "approved")
            )
            conn.commit()
            return task_id
    except Exception as e:
        print(f"Error creating task '{task_name}': {e}")
        conn.rollback()
        return None


def extract_scores_from_data(data):
    """Extract scores from the complex JSON data structure."""
    try:
        # First check for rouge metrics
        if "rouge,none" in data:
            return {
                # "rouge1": data["rouge,none"].get("rouge1", 0),
                # "rouge2": data["rouge,none"].get("rouge2", 0),
                # "rougeL": data["rouge,none"].get("rougeL", 0),
                "rougeLsum": data["rouge,none"].get("rougeLsum", 0)
            }
        # Check for BLEU metrics
        elif "bleu,none" in data:
            return {
                "bleu,none": data.get("bleu,none", 0),
            }
        # Check in results section
        elif "results" in data:
            for key, result in data["results"].items():
                if hasattr(result, "rouge_none") or "rouge_none" in result:
                    rouge_data = result.get("rouge_none", {})
                    return {
                        # "rouge1": rouge_data.get("rouge1", 0),
                        # "rouge2": rouge_data.get("rouge2", 0),
                        # "rougeL": rouge_data.get("rougeL", 0),
                        "rougeLsum": rouge_data.get("rougeLsum", 0)
                    }

        raise Exception("No results were found in data: " + str(data))
    except Exception as e:
        print(data)
        traceback.print_exc()
        print(f"Error extracting scores: {e}")
        return {"rouge1": 0, "rouge2": 0, "rougeL": 0, "rougeLsum": 0}


def calculate_average_score(scores):
    """Calculate average score from metrics."""
    # TODO: corect avrage calculation
    try:
        # For ROUGE metrics
        if "rougeLsum" in scores:
            # Prioritize rougeLsum if available
            return float(scores.get("rougeLsum", 0))

        # For BLEU metrics
        elif "bleu,none" in scores:
            return float(scores.get("bleu,none", 0))

        raise Exception("No valid scores found")
    except Exception as e:
        print(f"Error calculating average score: {e}")
        return 0


def extract_model_info(data):
    """Extract model information from the data."""
    try:
        if "config" in data and "model_args" in data["config"]:
            return {
                "model": data["config"]["model_args"]["model"],
                "base_url": data["config"]["model_args"]["base_url"],
                "adapter": data["config"]["model"],
            }
        return None
    except Exception as e:
        print(f"Error extracting model info: {e}")
        return None


def extract_task_results_from_data(task_key, data):
    """Extract task results from the JSON data."""
    try:
        results = data.get("results", {})
        if task_key in results:
            return results[task_key]
        return None
    except Exception as e:
        print(f"Error extracting task results: {e}")
        return None


def extract_tasks_from_data(data):
    """Extract task names from the JSON data."""
    tasks = []

    try:
        # Try to extract from results section
        if "results" in data:
            for key, result in data["results"].items():
                task_name = result.get("task", "")
                # Use same name for English and Arabic
                tasks.append(
                    {"name": task_name, "name_ar": task_name, "key": key})

    except Exception as e:
        print(f"Error extracting tasks: {e}")

    return tasks


def process_results_folder(base_path):
    """
    Process all result files in the given folder structure.

    Args:
        base_path: Base directory containing model folders
    """

    # Database configuration from environment variables
    db_config = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", 5432)),
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD")
    }

    # Connect to PostgreSQL
    conn = psycopg2.connect(**db_config)

    try:
        # Walk through the directory structure
        for model_folder in os.listdir(base_path):
            model_path = os.path.join(base_path, model_folder)
            user_id = model_access_map[model_folder] if model_folder in model_access_map else "djsek471pqdxitt"

            if os.path.isdir(model_path):
                model_info = None  # Will extract from first JSON file

                # Process each category file in the model folder
                for category_file in os.listdir(model_path):
                    if category_file.endswith('.json'):
                        # Read the result file first to extract model info if needed
                        file_path = os.path.join(model_path, category_file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                results_data = json.load(f)

                                # Extract model info if not yet available
                                if model_info is None:
                                    model_info = extract_model_info(
                                        results_data)

                                if model_info is None:
                                    # If still can't get model info, use defaults
                                    model_info = {
                                        "model": model_folder,
                                        "base_url": "",
                                        "adapter": "openai"
                                    }

                                # Create or get model ID
                                model_id = create_model_if_not_exists(
                                    conn,
                                    model_info["model"],
                                    model_folder,  # Use folder name as display name
                                    model_info["base_url"],
                                    model_info["adapter"],
                                    user_id
                                )

                                if not model_id:
                                    print(
                                        f"Skipping model folder '{model_folder}' - could not create/retrieve model")
                                    continue

                                # Get category name from file name
                                category_name = os.path.splitext(
                                    category_file)[0]

                                # Create or get category ID
                                category_id = create_category_if_not_exists(
                                    conn, category_name, category_name  # Same for English and Arabic
                                )

                                if not category_id:
                                    print(
                                        f"Skipping category file '{category_file}' - could not create/retrieve category")
                                    continue

                                # Create or get evaluation ID for this model
                                evaluation_id = create_evaluation_if_not_exists(
                                    conn, model_id, user_id)
                                if not evaluation_id:
                                    print(
                                        f"Skipping model folder '{model_folder}' - could not create/retrieve evaluation")
                                    continue

                                # Create job for this evaluation
                                job_id = create_job_if_not_exists(
                                    conn, model_id, category_id, user_id, evaluation_id)
                                if not job_id:
                                    print(
                                        f"Skipping model folder '{model_folder}' - could not create job")
                                    continue

                                # Extract tasks from the file
                                tasks = extract_tasks_from_data(results_data)

                                if tasks:
                                    # Process each task
                                    for task_data in tasks:
                                        task_id = create_task_if_not_exists(
                                            conn,
                                            task_data["name"],
                                            task_data["name_ar"],
                                            category_id,
                                            user_id
                                        )

                                        if not task_id:
                                            print(
                                                f"Skipping task '{task_data['name']}' - could not create/retrieve task " +
                                                category_file + ", " + model_info["model"])
                                            continue

                                        # Create evaluation-task link
                                        link_id = create_evaluation_task_link(
                                            conn, evaluation_id, task_id)

                                        task_result_data = extract_task_results_from_data(
                                            task_data["key"], results_data)

                                        # Insert the result
                                        result_id = insert_result(
                                            conn, job_id, user_id, evaluation_id,
                                            category_id, task_id, task_result_data
                                        )

                                        if not result_id:
                                            print(
                                                f"Skipping task '{task_data['name']}' - could not insert result " +
                                                category_file + ", " + model_info["model"])
                                            continue
                                else:
                                    print("Error no tasks found: " +
                                          category_file + ", " + model_info["model"])

                                # # TODO: what is this?
                                # # No tasks found, use category directly
                                # final_average_scores = results_data["average_scores"]
                                # final_score_data = {
                                #     "rouge,none" if "rougeLsum" in final_average_scores else "bleu,none": final_average_scores
                                # }

                                # result_id = insert_result(
                                #     conn, job_id, user_id, evaluation_id,
                                #     category_id, None, final_score_data
                                # )
                        except Exception as e:
                            traceback.print_exc()
                            print(f"Error processing file {file_path}: {e}")

                update_evaluation_score(conn, evaluation_id)

        print("All results processed successfully")

    except Exception as e:
        print(f"Error processing results: {e}")
        conn.rollback()
    finally:
        conn.close()


def create_evaluation_if_not_exists(conn, model_id, user_id):
    """Create an evaluation if needed and return its ID."""
    try:
        with conn.cursor() as cursor:
            # Check if evaluation exists for this model
            cursor.execute(
                "SELECT id FROM public.evaluation WHERE model_id = %s AND user_id = %s",
                (model_id, user_id)
            )
            result = cursor.fetchone()

            if result:
                return result[0]

            # Create a new evaluation
            evaluation_id = f"eval_{uuid.uuid4()}"
            cursor.execute(
                """
                INSERT INTO public.evaluation
                (id, user_id, model_id, status, created_at, updated_at)
                VALUES (%s, %s, %s, %s, NOW(), NOW())
                RETURNING id
                """,
                (str(evaluation_id), str(user_id), str(model_id), "running")
            )
            conn.commit()
            return evaluation_id
    except Exception as e:
        traceback.print_exc()
        print(f"Error creating evaluation for model {model_id}: {e}")
        conn.rollback()
        return None


def update_evaluation_score(conn, evaluation_id):
    """Create an evaluation if needed and return its ID."""
    try:
        with conn.cursor() as cursor:
            # Check if evaluation exists for this model
            cursor.execute(
                "SELECT * from public.result WHERE evaluation_id = %s",
                (evaluation_id,)
            )
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

            results = [dict(zip(columns, row)) for row in rows]

            total = 0

            for row in results:
                total += row["average_score"]

            average = total / len(results) if results else 0

            cursor.execute(
                "UPDATE public.evaluation SET average_score = %s, status = 'completed' WHERE id = %s",
                (average, evaluation_id)
            )
            conn.commit()

            if cursor.rowcount > 0:
                return True

            print(
                f"Failed to update evaluation {evaluation_id} final average score")
            return False
    except Exception as e:
        traceback.print_exc()
        print(f"Error updating evaluation for evaluation {evaluation_id}: {e}")
        conn.rollback()
        return False


def create_job_if_not_exists(conn, model_id, category_id, user_id, evaluation_id):
    """Create a job if needed and return its ID."""
    try:
        with conn.cursor() as cursor:
            # Create a new job
            job_id = f"job_{uuid.uuid4()}"
            cursor.execute(
                """
                INSERT INTO public.job
                (id, user_id, category_id, evaluation_id, status, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
                RETURNING id
                """,
                (job_id, user_id, category_id, evaluation_id, "completed")
            )
            conn.commit()
            return job_id
    except Exception as e:
        print(f"Error creating job: {e}")
        conn.rollback()
        return None


def create_evaluation_task_link(conn, evaluation_id, task_id):
    """Create a link between evaluation and task in the evaluation_task table."""
    try:
        with conn.cursor() as cursor:
            # Check if link already exists
            cursor.execute(
                "SELECT id FROM public.evaluation_task WHERE evaluation_id = %s AND task_id = %s",
                (evaluation_id, task_id)
            )
            result = cursor.fetchone()

            if result:
                return result[0]  # Link already exists

            # Create new link
            link_id = f"ev_task_{uuid.uuid4()}"
            cursor.execute(
                """
                INSERT INTO public.evaluation_task
                (id, evaluation_id, task_id)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (link_id, evaluation_id, task_id)
            )
            conn.commit()
            return link_id
    except Exception as e:
        print(f"Error creating evaluation-task link: {e}")
        conn.rollback()
        return None


def insert_result(conn, job_id, user_id, evaluation_id, category_id, task_id, results_json):
    """Insert a result into the database."""
    try:
        with conn.cursor() as cursor:
            # Generate a unique ID for the result
            result_id = f"res_{uuid.uuid4()}"

            # Extract scores and calculate average
            scores = extract_scores_from_data(results_json)
            average_score = calculate_average_score(scores)

            # SQL for inserting result
            sql = """
            INSERT INTO public.result
            (id, job_id, user_id, evaluation_id, category_id, task_id, average_score, results, updated_at, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
            ON CONFLICT DO NOTHING
            RETURNING id
            """

            # Execute query
            cursor.execute(
                sql,
                (result_id, job_id, user_id, evaluation_id, category_id,
                 task_id, average_score, Json(results_json))
            )
            conn.commit()

            return result_id
    except Exception as e:
        print(f"Error inserting result: {e}")
        conn.rollback()
        return None


if __name__ == "__main__":
    # Get configuration from environment variables
    base_results_path = os.getenv("RESULTS_PATH")

    # Validate required environment variables
    required_vars = ["DB_NAME", "DB_USER", "DB_PASSWORD", "RESULTS_PATH"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(
            f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file")
        exit(1)

    # Process all result files
    process_results_folder(base_results_path)
