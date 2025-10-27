"""Task management module."""

import json
import logging
import os
import yaml
from typing import Any
from pathlib import Path
from .metrics_registry import get_metrics_registry

logger = logging.getLogger(__name__)


class LMHDataset:
    """Represents an LM Harness task."""
    
    def __init__(self, file_name: str = "dataset", directory=None) -> None:
        self.directory = directory or "."
        os.makedirs(self.directory, exist_ok=True)
        self.path = f"{self.directory}/{file_name}.json"
        
        with open(self.path, "r", encoding="utf8") as fp:
            task = json.load(fp)
        
        task = self._escape_newline(task)
        
        self.name = task.pop("name")
        self.data = task.pop("data")
        self.category_id = task.pop("category")
        self.metric = task["metric_list"][0]
        self.task_id = task.pop("task")
        self.task = task
        self.file_name = file_name
    
    def _escape_newline(self, task: dict[str, Any]) -> dict[str, Any]:
        """Escape newlines in task."""
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
        """Export data for a split."""
        for example in self.data.get(split, []):
            if isinstance(example.get("output"), list):
                example["output"] = example["output"][0]
        
        if self.data.get(split):
            with open(f"{self.directory}/{self.file_name}_{split}.json", "w", encoding="utf8") as fp:
                json.dump(self.data[split], fp, ensure_ascii=False)
    
    def validate(self) -> None:
        """Validate the task."""
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
        
        registry = get_metrics_registry()
        metric_name = self.metric["metric"] if isinstance(self.metric, dict) else self.metric
        metric_type = registry.detect_metric_type(metric_name)
        
        if metric_type:
            metric_obj = registry.get(metric_type)
            if metric_obj:
                yaml_data = metric_obj.get_yaml_config(base_yaml)
                self._write_yaml(yaml_data)
            else:
                self._write_yaml(base_yaml)
        else:
            self._write_yaml(base_yaml)
    
    def _export_splits(self) -> None:
        for split in ["train", "dev", "test"]:
            self._export_data(split)
    
    def _build_base_yaml(self) -> dict[str, Any]:
        doc_to_text = "{{instruction}}\n{{input}}" if self.data["test"][0].get("instruction") else "{{input}}"
        doc_to_target = "{{output[0]}}" if isinstance(self.data["test"][0].get("output"), list) else "output"
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
            "generation_kwargs": {"do_sample": False, "until": ["<|endoftext|>"]},
            "dataset_kwargs": {"data_files": data_files},
            **self.task,
        }
    
    def _write_yaml(self, data: dict[str, Any]) -> None:
        out_path = Path(self.directory) / f"{self.file_name}.yaml"
        with open(out_path, "w", encoding="utf8") as f:
            yaml.dump(data, f)