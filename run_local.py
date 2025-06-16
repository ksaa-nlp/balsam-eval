"""Entry point for the evaluation job."""

import os
import json
from dotenv import load_dotenv

from src.evaluation import EvaluatationJob
from src.task import LMHDataset


load_dotenv()

TEMP_DIR = ".temp"
TASKS_DIR = ".tasks"
os.makedirs(TEMP_DIR, exist_ok=True)

if __name__ == "__main__":
    # Read the tasks from directory "tasks"
    tasks: dict[str, list[str]] = dict()
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
                f"./{TEMP_DIR}/{file.split(".")[0]}.json", "w", encoding="utf-8"
            ) as f:
                d["json"]["category"] = d["category"]
                d["json"]["task"] = d["task"]
                json.dump(d["json"], f, ensure_ascii=False)

            # Initialize LMHDataset
            dataset = LMHDataset(f"{file.split(".")[0]}", TEMP_DIR)
            dataset.export()

            if tasks.get(d["category"]) is None:
                tasks[d["category"]] = []
            tasks[d["category"]].append(dataset.name)

    # Initialize the evaluation job
    for category, t in tasks.items():
        print(f"Running evaluation for category: {category}")
        print(f"Total tasks: {len(t)}")
        try:
            model_args = {}

            base_rul = os.getenv("BASE_URL")
            if base_rul:
                model_args["base_url"] = base_rul
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

            job = EvaluatationJob(
                tasks=t,
                adapter=os.getenv("ADAPTER", "local-chat-completions"),
                task_id=category,
                output_path=category,
                model_args=model_args,
            )
            job.run()
        except Exception as e:
            print(f"An error occurred: {e}")
            continue
