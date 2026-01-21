"""
Task management module for LM Evaluation Harness.
Supports JSON, CSV, and XML templates and exports clean YAML/JSON configs.
"""

import json
import csv
import logging
import os
import yaml
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import random
from .metrics_registry import get_metrics_registry

logger = logging.getLogger(__name__)

# TODO: need to not remove elements from orginal dict
class LMHDataset:
    """
    Represents an LM Harness task.
    Accepts multiple source formats and produces standardized JSON and YAML.
    """

    # Metadata keys from new templates to exclude from final YAML
    IGNORED_METADATA = {
        "author",
        "organization",
        "category",
        "version",
        "Guidelines_creating_data",
        "List_possible_outputs",
        "Type of Data",
        "Type of result",
        "image_note",
        "Type of input",
        "Type of output",
        "source",
        "type_of_input",
        "type_of_output",
        "type_of_result",
        "guidelines_creating_data",
        "guidelines_creating_dataset",
    }

    def __init__(
        self, file_name: str = "dataset-n", 
        directory: str = None, 
    ) -> None:
        """
        Args:
            file_name: Name of the file (extension is optional).
            directory: Path to the directory containing the file.
            dev_size: Number of items from the end of the data to use for the 'dev' split.
        """
        self.directory = directory or "."
        os.makedirs(self.directory, exist_ok=True)

        # 1. Resolve file path and extension
        self.path, self.extension = self._resolve_file_path(self.directory, file_name)
        self.file_name = Path(self.path).stem

        # 2. Load and normalize based on format
        if self.extension == ".json":
            task_dict = self._load_json(self.path)
        elif self.extension == ".csv":
            task_dict = self._load_csv(self.path)
        elif self.extension == ".xml":
            task_dict = self._load_xml(self.path)
        else:
            raise ValueError(f"Unsupported file format: {self.extension}")

        # 3. Escape newlines to prevent YAML formatting issues
        task_dict = self._escape_newline(task_dict)
        self.metadata = {}

        for key in ("version", "author", "organization", "category", "task", "source"):
            if key in task_dict:
                self.metadata[key] = task_dict[key]

        # 4. Extract core harness fields
        self.name = task_dict.pop("name", "Unknown Task")
        self.task_id = task_dict.pop("task", "unknown_task_id")

        # Handle metric (could be a string in new template or dict in old)
        self.metric = task_dict.pop("metric", None)
        self.category_id = task_dict.pop("category", None)

        # 5. Handle Data Splitting (Taking last X elements for dev)
        raw_data = task_dict.pop("data", [])
        self.data = self._create_splits(raw_data)

        # 6. Filter remaining metadata to keep YAML clean
        self.task_kwargs = {
            k: v for k, v in task_dict.items() if k not in self.IGNORED_METADATA
        }

    def _resolve_file_path(self, directory: str, file_name: str) -> Tuple[str, str]:
        """Checks if file exists with or without supported extensions."""
        base_path = os.path.join(directory, file_name)
        if os.path.isfile(base_path):
            return base_path, os.path.splitext(base_path)[1].lower()

        for ext in [".json", ".csv", ".xml"]:
            path = f"{base_path}{ext}"
            if os.path.isfile(path):
                return path, ext
        raise FileNotFoundError(f"File {file_name} not found in {directory}")

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf8") as fp:
            return json.load(fp)

    def _load_csv(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
        if len(reader) < 2:
            raise ValueError("CSV structure invalid: metadata or data missing.")

        # Parse Meta (Rows 0 & 1)
        task = dict(zip(reader[0], reader[1]))

        # Find start of data section (where 'id' is the first column)
        data_header_idx = -1
        for i, row in enumerate(reader):
            if row and row[0] == "id":
                data_header_idx = i
                break

        if data_header_idx == -1:
            raise ValueError("CSV data section missing 'id' column header.")

        keys = reader[data_header_idx]
        data_rows = []
        for row in reader[data_header_idx + 1 :]:
            if not row or not any(row):
                continue
            row_dict = {keys[i]: row[i] for i in range(len(keys)) if i < len(row)}

            item = {
                "id": row_dict.get("id"),
                "instruction": row_dict.get("instruction"),
                "output": row_dict.get("output"),
                "source": row_dict.get("source"),
                "input": [
                    v for k, v in row_dict.items() if k.startswith("input_") and v
                ],
                "Experimental prompts": [
                    v
                    for k, v in row_dict.items()
                    if k.startswith("experimental_") and v
                ],
            }
            mcq = [v for k, v in row_dict.items() if k.startswith("mcq_") and v]
            if mcq:
                item["mcq"] = mcq
            data_rows.append(item)

        task["data"] = data_rows
        return task

    def _load_xml(self, path: str) -> Dict[str, Any]:
        tree = ET.parse(path)
        root = tree.getroot()
        task = {
            child.tag: child.text
            for child in root
            if child.tag not in ["data", "guidelines_creating_data"]
        }

        data_rows = []
        data_node = root.find("data")
        if data_node is not None:
            for item_node in data_node.findall("item"):
                item = {}
                for child in item_node:
                    if child.tag == "experimental_prompts":
                        item["Experimental prompts"] = [
                            p.text for p in child.findall("prompt") if p.text
                        ]
                    elif child.tag == "input":
                        item["input"] = [
                            v.text for v in child.findall("value") if v.text
                        ]
                    elif child.tag == "mcq":
                        item["mcq"] = [
                            o.text for o in child.findall("option") if o.text
                        ]
                    else:
                        item[child.tag] = child.text
                data_rows.append(item)
        task["data"] = data_rows
        return task

    def _create_splits(self, raw_data: Any) -> Dict[str, list]:
            """
           extract the dev/test data from the template and return error
            """
            if isinstance(raw_data, dict):
                if "test" not in raw_data and "dev" not in raw_data:
                    raise ValueError("Invalid Template: 'data' object must contain 'test' or 'dev' keys.")
                return {
                    "test": raw_data.get("test", []),
                    "dev": raw_data.get("dev", [])
                }

            raise ValueError("ERROR: Unsupported Dataset Template.")



    def _escape_newline(self, task: Any) -> Any:
        if isinstance(task, dict):
            return {k: self._escape_newline(v) for k, v in task.items()}
        elif isinstance(task, list):
            return [self._escape_newline(i) for i in task]
        elif isinstance(task, str):
            return task.replace("\n", "\\n")
        return task

    def _export_data(self, split: str) -> None:
        """Saves a split list to a JSON file (required by LM Harness)."""
        items = self.data.get(split, [])
        if not items:
            return

        processed = []
        for item in items:
            it = item.copy()
            # Join input list to single string for Jinja templates
            if isinstance(it.get("input"), list):
                it["input"] = "\n".join(str(x) for x in it["input"])
            # Ensure output is a single string
            if isinstance(it.get("output"), list) and it["output"]:
                it["output"] = it["output"][0]
            processed.append(it)

        out_path = f"{self.directory}/{self.file_name}_{split}.json"
        with open(out_path, "w", encoding="utf8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)

    def validate(self) -> None:
        if not any(self.data.values()):
            raise ValueError("Dataset data is empty.")

    def export(self) -> None:
        """Main method to create the JSON split files and the YAML config."""
        self.validate()

        # 1. Export the split JSON files
        for split in ["test", "dev"]:
            self._export_data(split)

        # 2. Build the configuration dictionary
        base_yaml = self._build_base_yaml()

        # 3. Apply metric logic from registry
        registry = get_metrics_registry()
        m_name = self.metric["metric"] if isinstance(self.metric, dict) else self.metric

        if not m_name:
            # No metric specified, use base yaml
            final_yaml = base_yaml
            logger.warning("No metric specified in dataset. Using base configuration.")
        else:
            metric_info = registry.get_metric_info(m_name)
            
            if not metric_info["found"]:
                # Metric not found
                logger.warning(f"Metric '{m_name}' not found in custom or LM Harness registries. " 
                             f"Including it in YAML but it may not work during evaluation.")
                final_yaml = base_yaml.copy()
                final_yaml["metric_list"] = [{
                    "metric": m_name,
                    "aggregation": m_name,
                    "higher_is_better": True,
                }]
            elif metric_info["source"] == "custom":
                # Custom metric with YAML configuration
                metric_obj = metric_info["custom_metric"]
                final_yaml = metric_obj.get_yaml_config(base_yaml)
                logger.info(f"Using custom metric: {metric_info['name']}")
            else:
                # LM Harness metric - use metric configuration
                final_yaml = base_yaml.copy()
                final_yaml["output_type"] = metric_info["output_type"]
                final_yaml["metric_list"] = [{
                    "metric": metric_info["name"],
                    "aggregation": metric_info["aggregation"],
                    "higher_is_better": metric_info["higher_is_better"],
                }]
                
                # Adjust generation_kwargs based on output_type
                if metric_info["output_type"] == "loglikelihood":
                    # For loglikelihood tasks, we don't need generation_kwargs
                    if "generation_kwargs" in final_yaml:
                        del final_yaml["generation_kwargs"]
                
                logger.info(f"Using LM Harness metric: {metric_info['name']} "
                          f"(output_type: {metric_info['output_type']})")

        # 4. Write YAML
        self._write_yaml(final_yaml)

    def _build_base_yaml(self) -> Dict[str, Any]:
        """Constructs the YAML content using filtered metadata."""
        # Check if test split exists, otherwise use dev for reference
        ref_items = self.data.get("test") or self.data.get("dev")
        ref_item = ref_items[0] if ref_items else {}

        # Clean doc_to_text: check for instruction key
        has_ins = ref_item.get("instruction") and "not_ins" not in str(
            ref_item.get("instruction")
        )
        doc_to_text = "{{instruction}}\n{{input}}" if has_ins else "{{input}}"

        # Construct data file paths
        data_files = {
            k: f"{self.directory}/{self.file_name}_{k}.json"
            for k in ["dev", "test"]
            if self.data.get(k)
        }

        # Final dictionary for YAML
        return {
            "task": self.name,
            "dataset_path": "json",
            "dataset_name": self.name,
            "test_split": "test" if "test" in data_files else None,
            "validation_split": "dev" if "dev" in data_files else None,
            "doc_to_text": doc_to_text,
            "doc_to_target": "output",
            "output_type": "generate_until",
            "generation_kwargs": {"do_sample": False, "until": ["<|endoftext|>"]},
            "dataset_kwargs": {"data_files": data_files},
            "metadata": self.metadata,
            **self.task_kwargs,
        }

    def _write_yaml(self, data: Dict[str, Any]) -> None:
            """Writes dictionary to YAML file with proper string formatting."""
            out_path = Path(self.directory) / f"{self.file_name}.yaml"
            
            # Custom representer for multiline strings to use literal style (|)
            def str_representer(dumper, data):
                if '\n' in data:
                    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
                return dumper.represent_scalar('tag:yaml.org,2002:str', data)
            
            yaml.add_representer(str, str_representer)
            
            with open(out_path, "w", encoding="utf8") as f:
                yaml.dump(data, f, sort_keys=False, allow_unicode=True, default_flow_style=False, width=1000)