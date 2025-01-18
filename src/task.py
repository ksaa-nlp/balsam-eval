"""A module for working with tasks and datasets."""

import json
import logging
import os
from typing import Any

import yaml
from google.cloud import storage

ValidationError = {
    "missing_split": "Expected 'test' or 'validation' keys in the data.",
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
        self.metric = task.pop("metric")
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
        if self.data.get(split):
            with open(
                f"{self.directory}/{self.file_name}_{split}.json", "w", encoding="utf8"
            ) as fp:
                json.dump(self.data[split], fp, ensure_ascii=False)
        logger.info("Exported %s split to %s.json", split, split)

    def validate(self) -> None:
        """Validate the task."""
        # Check that the data contains either a test or validation split.
        # One of these is required for the task to be valid.
        if not self.data.get("test") and not self.data.get("validation"):
            raise ValueError("Expected 'test' or 'validation' keys in the data.")
        # Check the splits for the required keys
        for split in ["train", "validation", "test"]:
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
        for split in ["train", "validation", "test"]:
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
        if self.data.get("validation"):
            data_files["validation"] = (
                f"{self.directory}/{self.file_name}_validation.json"
            )
        if self.data.get("test"):
            data_files["test"] = f"{self.directory}/{self.file_name}_test.json"

        yaml_data = {
            "task": self.name,
            "dataset_path": "json",
            "dataset_name": None,
            "test_split": "test" if self.data.get("test") else None,
            "validation_split": "validation" if self.data.get("validation") else None,
            "doc_to_text": doc_to_text,
            "doc_to_target": doc_to_target,
            "output_type": "generate_until",
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
                "dataset_name": "null",
                "test_split": {"test" if self.data.get("test") else "null"},
                "doc_to_text": doc_to_text,
                "doc_to_target": doc_to_target,
                "process_results": "!function utils_morph.process_results",
                "output_type": "generate_until",
                "metric_list": [
                    {
                        "metric": self.metric,
                        "aggregation": "!function utils_morph.custom_rouge_agg",
                        "higher_is_better": True,
                    }
                ],
                "dataset_kwargs": {"data_files": data_files},
            }

            process_results = """
import re
import nltk
import evaluate
import pyarabic.araby as araby
import unicodedata
import numpy as np
import sys
from scipy.optimize import linear_sum_assignment
from lm_eval.api.registry import register_aggregation, register_metric

# Download necessary resources
nltk.download('punkt_tab')

# Load Rouge evaluator from `evaluate` library
rouge = evaluate.load('rouge')

# Punctuation table
PUNCT_TABLE = dict.fromkeys(i for i in range(sys.maxunicode)
                            if unicodedata.category(chr(i)).startswith('P'))
ALL_PUNCTUATIONS = "".join(chr(p) for p in PUNCT_TABLE)
others = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
ALL_PUNCTUATIONS += ''.join([o for o in others if o not in ALL_PUNCTUATIONS])
 
def prepare_texts(text, change_curly_braces):

    # 1. put spaces before and after each punctuation
    text = re.sub('([' + ALL_PUNCTUATIONS + '])', ' \\1 ', text)

    # 2. change all {} to []
    if change_curly_braces:
        text = text.replace('{', '[').replace('}', ']')

    return text
def rouge1_scores(predictions, references):  # This is a passthrough function

    return (predictions[0], references[0])


def custom_rouge_agg(items):
    tokenizer = lambda x: x.split()
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    # Preprocess the texts
    refs = [prepare_texts(ref, change_curly_braces=False).strip() for ref in refs]
    preds = [prepare_texts(pred, change_curly_braces=True).strip() for pred in preds]
    # Initialize sums for each ROUGE score
    rouge1= 0.0
    rouge2 = 0.0
    rougeL= 0.0
    rougeLsum = 0.0
    
    for i in range(len(refs)):
        # Compute ROUGE scores
        score = rouge.compute(references=[refs[i]], predictions=[preds[i]], tokenizer=tokenizer)
        
        # Aggregate each type of ROUGE score
        rouge1 += score["rouge1"]
        rouge2 += score["rouge2"]
        rougeL += score["rougeL"]
        rougeLsum += score["rougeLsum"]

    count = len(refs)
    avg_rouge1 = rouge1 / count
    avg_rouge2 = rouge2 / count
    avg_rougeL = rougeL / count
    avg_rougeLsum = rougeLsum / count

    return {
        "rouge1": avg_rouge1,
        "rouge2": avg_rouge2,
        "rougeL": avg_rougeL,
        "rougeLsum": avg_rougeLsum,
    }
    
def process_results(doc, results):
    preds, golds = results[0], doc["output"]
    # rouge_scores = evaluate_rouge_metric(preds, golds)
    # print(f"ROUGE Scores: {rouge_scores}")
    return {'rouge':[golds, preds]}
    	"""

            with open(
                f"{self.directory}/{self.file_name}.yaml", "w", encoding="utf8"
            ) as fp:
                fp.write(yaml.dump(yaml_data))
            with open(f"{self.directory}/utils_morph.py", "w", encoding="utf8") as f:
                f.write(process_results)
            logger.info("Exported task to %s.yaml", self.name)
        else:
            with open(
                f"{self.directory}/{self.file_name}.yaml", "w", encoding="utf8"
            ) as fp:
                fp.write(yaml.dump(yaml_data))
            logger.info("Exported task to %s.yaml", self.name)
