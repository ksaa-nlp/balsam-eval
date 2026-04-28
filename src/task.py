"""Task management module for LM Harness (simple JSON format)."""

import json
import logging
import os
import yaml
from pathlib import Path
from typing import Any

from src.core.helpers import sanitize_config_name
from src.metrics_registry import get_metrics_registry

logger = logging.getLogger(__name__)


class LMHDataset:
    """Represents an LM Harness task in simple JSON format."""

    def __init__(self, file_name: str = "dataset", directory: str | None = None) -> None:
        """Initialize LM Harness dataset.

        Args:
            file_name: Name of the dataset file (without extension)
            directory: Directory containing the dataset file
        """
        self.directory = directory or "."
        os.makedirs(self.directory, exist_ok=True)
        self.path = f"{self.directory}/{file_name}.json"
        self.file_name = file_name

        # Load task data
        with open(self.path, "r", encoding="utf8") as fp:
            task = json.load(fp)

        task = self._escape_newline(task)

        # Extract task properties
        self.name = task.pop("name")
        self.data = task.pop("data")
        self.category_id = task.pop("category")
        self.metric = task["metric_list"][0]
        self.task_id = task.pop("task")
        self.task = task

    def _escape_newline(self, task: dict[str, Any]) -> dict[str, Any]:
        """Escape newlines in task data.

        Args:
            task: Task dictionary

        Returns:
            Task with escaped newlines
        """
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
        """Export data for a split.

        Args:
            split: Split name (train, dev, test)
        """
        for example in self.data.get(split, []):
            if isinstance(example.get("output"), list):
                example["output"] = example["output"][0]

        if self.data.get(split):
            with open(
                f"{self.directory}/{self.file_name}_{split}.json",
                "w",
                encoding="utf8",
            ) as fp:
                json.dump(self.data[split], fp, ensure_ascii=False)

    def validate(self) -> None:
        """Validate the task data."""
        if not self.data.get("test") and not self.data.get("dev"):
            raise ValueError("Expected 'test' or 'dev' keys in data.")

        for split in ["train", "dev", "test"]:
            if self.data.get(split):
                for example in self.data[split]:
                    if any(key not in example for key in ["input", "output"]):
                        raise ValueError("Expected 'input' and 'output' keys.")

    def export(self) -> None:
        """Export task to YAML format."""
        self.validate()
        self._export_splits()
        base_yaml = self._build_base_yaml()

        # Apply metric configuration
        registry = get_metrics_registry()
        metric_name = (
            self.metric["metric"] if isinstance(self.metric, dict) else self.metric
        )
        metric_type = registry.detect_metric_type(metric_name)

        if metric_type:
            metric_obj = registry.get(metric_type)
            if metric_obj:
                yaml_data = metric_obj.get_yaml_config(base_yaml)
                self._write_yaml(yaml_data)
                return

        self._write_yaml(base_yaml)

    def _export_splits(self) -> None:
        """Export all data splits."""
        for split in ["train", "dev", "test"]:
            self._export_data(split)

    def _build_base_yaml(self) -> dict[str, Any]:
        """Build base YAML configuration.

        Returns:
            Base YAML configuration dictionary
        """
        doc_to_text = (
            "{{instruction}}\n{{input}}"
            if self.data["test"][0].get("instruction")
            else "{{input}}"
        )
        doc_to_target = (
            "{{output[0]}}"
            if isinstance(self.data["test"][0].get("output"), list)
            else "output"
        )
        data_files = {
            k: f"{self.directory}/{self.file_name}_{k}.json"
            for k in ["dev", "test"]
            if self.data.get(k)
        }

        return {
            "task": self.name,
            "dataset_path": "json",
            "dataset_name": sanitize_config_name(self.name),
            "test_split": "test" if "test" in data_files else None,
            "validation_split": "dev" if "dev" in data_files else None,
            "doc_to_text": doc_to_text,
            "doc_to_target": doc_to_target,
            "output_type": "generate_until",
            "generation_kwargs": {"do_sample": False, "until": [""]},
            "dataset_kwargs": {"data_files": data_files},
            **self.task,
        }

    def _write_yaml(self, data: dict[str, Any]) -> None:
        """Write YAML configuration to file.

        Args:
            data: YAML configuration dictionary
        """
        out_path = Path(self.directory) / f"{self.file_name}.yaml"
        with open(out_path, "w", encoding="utf8") as f:
            yaml.dump(data, f)
