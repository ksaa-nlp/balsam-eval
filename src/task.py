"""A module for working with tasks and datasets."""

import json
import logging
import os
import yaml
from typing import Any
from pathlib import Path


from google.cloud import storage
from . import bleu_score
from . import accuracy_score
from . import utils

ValidationError = {
    "missing_split": "Expected 'test' or 'dev' keys in the data.",
    "missing_key": "Expected 'instruction', 'input', and 'output' keys in the split.",
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


def load_dataset_from_local(
    dataset_id: str, directory: str = "tasks"
) -> dict[str, Any]:
    """
    Load a dataset from a local directory. The dataset is identified by a dataset ID
    which corresponds to the file name without the `.json` extension.
    """
    # Create the path for the file
    file_path = f"{directory}/{dataset_id}.json"

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")

    # Load the dataset
    with open(file_path, "r", encoding="utf8") as fp:
        dataset = json.load(fp)
        return dataset


def download_dataset_from_gcs(dataset_id: str, directory: str) -> dict[str, Any]:
    """
    Download a dataset from GCS. The dataset is identified by a dataset ID
    which corresponds to its primary key in the database.
    """
    storage_client = storage.Client()
    bucket_name = os.getenv("GCLOUD_BUCKET")
    bucket = storage_client.bucket(bucket_name)
    # Create a temporary directory to store the dataset
    os.makedirs(directory, exist_ok=True)
    blob = bucket.blob(f"datasets/{dataset_id}.json")
    blob.download_to_filename(f".temp/{dataset_id}.json")

    print(f"Downloaded {dataset_id}.json from GCS bucket.")

    # Read the dataset from the file
    with open(f".temp/{dataset_id}.json", "r", encoding="utf8") as fp:
        dataset = json.load(fp)
        dd = dataset["json"]
        dd["task"] = dataset["task"]
        dd["category"] = dataset["category"]
        # Overwrite the dataset with the new data
        with open(f".temp/{dataset_id}.json", "w", encoding="utf8") as fp:
            json.dump(dd, fp, ensure_ascii=False)
        return dd


class LMHDataset:
    """
    Represents an LM Harness task. The task is loaded from a JSON file
    and can be exported to a YAML file to be compatible with LM Harness.
    """

    def __init__(self, file_name: str = "dataset", directory=None) -> None:
        task: dict[str, Any] = {}
        # If a directory is provided, prepend it to the path
        if directory:
            self.directory = directory
            os.makedirs(directory, exist_ok=True)
            self.path = f"{directory}/{file_name}.json"
        with open(self.path, "r", encoding="utf8") as fp:
            task.update(json.load(fp))
        task = self._escape_newline(task)

        # Required fields
        self.name = task.pop("name")
        self.data = task.pop("data")
        self.category_id = task.pop("category")
        self.metric = task["metric_list"][0]
        if not self.metric:
            print(f"Skipping task {self.name} as it has no metric.")
        self.task_id = task.pop("task")

        # Remaining fields
        self.task = task

        self.file_name = file_name

        logger.info("Loaded task: %s", self.name)

    def _escape_newline(self, task: dict[str, Any]) -> dict[str, Any]:
        """Make sure newlines are escaped in the task."""
        for key, value in task.items():
            if isinstance(value, str) and "\n" in value:
                task[key] = value.replace("\n", "\\n")
            if isinstance(value, dict):
                task[key] = self._escape_newline(value)
            if isinstance(value, list):
                for idx, item in enumerate(value):
                    if isinstance(item, str) and "\n" in item:
                        task[key][idx] = item.replace("\n", "\\n")
                    if isinstance(item, dict):
                        task[key][idx] = self._escape_newline(item)
        return task

    def _export_data(self, split: str) -> None:
        """Export the data for a given split to a JSON file."""
        # A static check to ensure `output` is a string
        for example in self.data.get(split, []):
            if isinstance(example.get("output"), list):
                example["output"] = example["output"][0]

        if self.data.get(split):
            with open(
                f"{self.directory}/{self.file_name}_{split}.json", "w", encoding="utf8"
            ) as fp:
                json.dump(self.data[split], fp, ensure_ascii=False)
        logger.info("Exported %s split to %s.json", split, split)

    def validate(self) -> None:
        """Validate the task."""
        # Check that the data contains either a test or dev split.
        # One of these is required for the task to be valid.
        if not self.data.get("test") and not self.data.get("dev"):
            raise ValueError("Expected 'test' or 'dev' keys in the data.")
        # Check the splits for the required keys
        for split in ["train", "dev", "test"]:
            if self.data.get(split):
                for example in self.data[split]:
                    if any(key not in example for key in ["input", "output"]):
                        raise ValueError(ValidationError["missing_key"])

    def export(self) -> None:
        """Export task to YAML format compatible with LM Harness."""
        self.validate()
        self._export_splits()
        yaml_data = self._build_base_yaml()

        if "morphological" in self.name.lower():
            self._export_morphological(yaml_data)
        elif "bleu" in self.metric["metric"].lower() if isinstance(self.metric, dict) else "bleu" in self.metric.lower():
            self._export_bleu(yaml_data)
        elif "accuracy" in self.metric["metric"].lower() if isinstance(self.metric, dict) else "accuracy" in self.metric.lower():
            self._export_accuracy(yaml_data)
        else:
            self._write_yaml(yaml_data, suffix="None")

    def _export_splits(self) -> None:
        for split in ["train", "dev", "test"]:
            self._export_data(split)

    def _build_base_yaml(self) -> dict[str, Any]:
        doc_to_text = "{{instruction}}\n{{input}}" if self.data["test"][0].get(
            "instruction") else "{{input}}"
        doc_to_target = "{{output[0]}}" if isinstance(
            self.data["test"][0].get("output"), list) else "output"
        data_files = {
            k: f"{self.directory}/{self.file_name}_{k}.json"
            for k in ["dev", "test"] if self.data.get(k)
        }

        return {
            "task": self.name,
            "dataset_path": "json",
            "dataset_name": None,
            "test_split": "test" if "test" in data_files else None,
            "validation_split": "dev" if "dev" in data_files else None,
            "doc_to_text": doc_to_text,
            "doc_to_target": doc_to_target,
            "output_type": "generate_until",
            "generation_kwargs": {"do_sample": False, "until": "<|endoftext|>"},
            "dataset_kwargs": {"data_files": data_files},
            **self.task,
        }

    def _export_morphological(self, yaml_data: dict[str, Any]) -> None:
        yaml_data.update({
            "process_results": utils.process_results,
            "metric_list": [{
                "metric": self.metric["metric"] if isinstance(self.metric, dict) else self.metric,
                "aggregation": utils.custom_rouge_agg,
                "higher_is_better": True,
            }],
        })
        self._write_yaml(yaml_data, suffix="Morphological")
        self._copy_dependency("utils.py")

    def _export_bleu(self, yaml_data: dict[str, Any]) -> None:
        yaml_data.update({
            "process_results": bleu_score.process_results,
            "metric_list": [{
                "metric": self.metric['metric'] if isinstance(self.metric, dict) else self.metric,
                "aggregation": bleu_score.custem_bleu_aggregation,
                "higher_is_better": True,
            }],
        })
        self._write_yaml(yaml_data, suffix="BLEU")
        self._copy_dependency("bleu_score.py")

    def _export_accuracy(self, yaml_data: dict[str, Any]) -> None:

        # Add instruction to just output the letter of the answer
        if not yaml_data.get("fewshot_config"):
            yaml_data["fewshot_config"] = {}

        # Add instructions to the doc_to_text field to ensure model outputs just the letter
        original_doc_to_text = yaml_data.get(
            "doc_to_text", "{{instruction}}\n{{input}}")

        yaml_data["doc_to_text"] = (
            f"{original_doc_to_text}\n\n"
            # Arabic instructions
            "يرجى اتباع الإرشادات التالية عند الإجابة:\n"
            "- إذا كان السؤال من نوع اختيار من متعدد، فأجب فقط بالحرف المقابل للإجابة الصحيحة (A أو B أو C أو D) بدون أي شرح أو تفسير.\n"
            "- إذا كان السؤال من نوع صح أو خطأ (نعم أو لا)، فأجب فقط بكلمة 'صح' أو 'خطأ' بدون أي شرح إضافي.\n"
            "- لا تُضِف أي تعليقات أو تفسيرات خارج المطلوب.\n\n"
            # English instructions (optional)
            "Please follow these instructions when answering:\n"
            "- If the question is multiple choice, respond **only** with the letter of the correct answer (A, B, C, or D) — no explanation.\n"
            "- If the question is true/false (yes or no), respond **only** with 'True' or 'False' (or 'Yes' / 'No') — no explanation.\n"
            "- Do not include any extra comments, reasoning, or justifications in your response."
        )

        # Keep using generate_until output type for accuracy
        yaml_data.update({
            # Removed doc_to_choice as we are using generate_until instead of multiple_choice
            "output_type": "generate_until",  # Keep using generate_until
            "process_results": accuracy_score.process_results,
            "metric_list": [{
                "metric": self.metric['metric'] if isinstance(self.metric, dict) else self.metric,
                "aggregation": accuracy_score.custom_accuracy_aggregation,
                "higher_is_better": True,
            }],
            # Keep the generation_kwargs that would be removed in the original implementation
            "generation_kwargs": {"do_sample": False, "until": "<|endoftext|>"},
        })

        self._write_yaml(yaml_data, suffix="Accuracy")
        self._copy_dependency("accuracy_score.py")

    def _write_yaml(self, data: dict[str, Any], suffix: str) -> None:
        out_path = Path(self.directory) / f"{self.file_name}.yaml"
        with open(out_path, "w", encoding="utf8") as f:
            yaml.dump(data, f)
        logger.info("%s Exported task to %s.yaml", suffix, self.name)

    def _copy_dependency(self, filename: str) -> None:
        src = Path(__file__).parent / filename
        dst = Path(self.directory) / filename
        dst.write_text(src.read_text(encoding="utf8"), encoding="utf8")
