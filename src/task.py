"""A module for working with tasks and datasets."""

import json
import logging
import os
import yaml
from typing import Any
from pathlib import Path


from google.cloud import storage
from . import bleu_score
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


storage_client = storage.Client()


def download_dataset_from_gcs(dataset_id: str, directory: str) -> dict[str, Any]:
    """
    Download a dataset from GCS. The dataset is identified by a dataset ID
    which corresponds to its primary key in the database.
    """
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
        """
        Construct a YAML file from the dataset matching the
        required format for LM Harness and write it to current directory.
        """
        self.validate()
        # Write a JSON file for each split
        for split in ["train", "dev", "test"]:
            self._export_data(split)
        # We know at this point that the data is valid
        # and each split has the required keys so we can proceed.
        doc_to_text = "{{input}}"
        if self.data["test"][0].get("instruction"):
            doc_to_text = "{{instruction}}\n{{input}}"

        doc_to_target = "output"
        if isinstance(self.data["test"][0].get("output"), list):
            doc_to_target = " " + "{{output[0]}}"

        data_files = {}
        # if self.data.get("train"):
        # data_files["train"] = f"{self.directory}/{self.file_name}_train.json"
        if self.data.get("dev"):
            data_files["dev"] = f"{self.directory}/{self.file_name}_dev.json"
        if self.data.get("test"):
            data_files["test"] = f"{self.directory}/{self.file_name}_test.json"

        yaml_data = {
            "task": self.name,
            "dataset_path": "json",
            "dataset_name": None,
            "test_split": "test" if self.data.get("test") else None,
            "validation_split": "dev" if self.data.get("dev") else None,
            "doc_to_text": doc_to_text,
            "doc_to_target": doc_to_target,
            "output_type": "generate_until",
            "generation_kwargs":{"do_sample": False, "until":"<|endoftext|>"},
            "dataset_kwargs": {
                "data_files": data_files,
            },
            # Dump the rest of the task
            **self.task,
        }

        if "morphological" in self.name.lower():
            yaml_data = {
                "task": self.name,
                "dataset_path": "json",
                "dataset_name": None,
                "test_split": "test" if self.data.get("test") else None,
                "validation_split": "dev" if self.data.get("dev") else None,
                "doc_to_text": doc_to_text,
                "doc_to_target": doc_to_target,
                "process_results": utils.process_results,
                "output_type": "generate_until",
                "generation_kwargs":{"do_sample": False, "until":"<|endoftext|>"},
                "metric_list": [
                    {
                        "metric": self.metric["metric"],
                        "aggregation": utils.custom_rouge_agg,
                        "higher_is_better": True,
                    }
                ],
                "dataset_kwargs": {"data_files": data_files},
            }

            with open(
                f"{self.directory}/{self.file_name}.yaml", "w", encoding="utf8"
            ) as fp:
                fp.write(yaml.dump(yaml_data))
            with open(f"{self.directory}/utils.py", "w", encoding="utf8") as f:
                # Copy the contents of the utils.py file
                with open(
                    f"{Path(__file__).parent.absolute()}/utils.py", "r", encoding="utf8"
                ) as u:
                    f.write(u.read())
            logger.info("Exported task to %s.yaml", self.name)
            
        elif "bleu" in self.metric.lower():
            yaml_data = {
                "task": self.name,
                "dataset_path": "json",
                "dataset_name": None,
                "test_split": "test" if self.data.get("test") else None,
                # "validation_split": "dev" if self.data.get("dev") else None,
                "doc_to_text": doc_to_text,
                "doc_to_target": doc_to_target,
                "process_results": bleu_score.process_results,
                "output_type": "generate_until",
                "generation_kwargs": {"do_sample": False, "until": "<|endoftext|>"},
                "metric_list": [
                    {
                        "metric": self.metric, 
                        "aggregation": bleu_score.custem_bleu_aggregation,
                        "higher_is_better": True,
                    }
                ],
                "dataset_kwargs": {"data_files": data_files},
            }

            with open(f"{self.directory}/{self.file_name}.yaml", "w", encoding="utf8") as fp:
                fp.write(yaml.dump(yaml_data))

            with open(f"{self.directory}/bleu_score.py", "w", encoding="utf8") as f:
                # Copy the contents of the utils.py file
                with open(f"{Path(__file__).parent.absolute()}/bleu_score.py", "r", encoding="utf8") as u:
                    f.write(u.read())

            logger.info("Exported task to %s.yaml", self.name)        
        else:
            with open(
                f"{self.directory}/{self.file_name}.yaml", "w", encoding="utf8"
            ) as fp:
                fp.write(yaml.dump(yaml_data))
            logger.info("Exported task to %s.yaml", self.name)
