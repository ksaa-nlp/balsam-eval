"""Task management module for LM Harness (multi-format support).

Supports JSON, CSV, and XML templates and exports clean YAML/JSON configs.
"""

import csv
import json
import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from src.core.helpers import sanitize_config_name
from src.metrics_registry import get_metrics_registry

logger = logging.getLogger(__name__)


def _is_image_file(filename: str) -> bool:
    """Check if a filename has an image extension.

    Args:
        filename: File name to check

    Returns:
        True if file has an image extension
    """
    return Path(filename).suffix.lower() in {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".webp",
    }


class LMHDataset:
    """Represents an LM Harness task with multi-format support.

    Accepts multiple source formats (JSON, CSV, XML) and produces
    standardized JSON and YAML configurations.
    """

    # Metadata keys from templates to exclude from final YAML
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
        self, file_name: str = "dataset-n", directory: str | None = None
    ) -> None:
        """Initialize LM Harness dataset.

        Args:
            file_name: Name of the file (extension is optional)
            directory: Path to the directory containing the file
            dev_size: Number of items from the end for 'dev' split (unused)
        """
        self.directory = directory or "."
        os.makedirs(self.directory, exist_ok=True)

        # Resolve file path and extension
        self.path, self.extension = self._resolve_file_path(self.directory, file_name)
        self.file_name = Path(self.path).stem

        # Store source directory for image path resolution
        self.source_directory = str(Path(self.path).parent)

        # Load and normalize based on format
        if self.extension == ".json":
            task_dict = self._load_json(self.path)
        elif self.extension == ".csv":
            task_dict = self._load_csv(self.path)
        elif self.extension == ".xml":
            task_dict = self._load_xml(self.path)
        else:
            raise ValueError(f"Unsupported file format: {self.extension}")

        # Escape newlines to prevent YAML formatting issues
        task_dict = self._escape_newline(task_dict)
        self.metadata = {}

        # Generate unique task name
        task_part = task_dict.get("task", "Unknown_Task")
        category_part = task_dict.get("category", "Unknown_Category")
        type_part = task_dict.get("Type of result", "")
        metric_part = task_dict.get("metric", "")

        unique_suffix = os.urandom(5).hex()
        self.name = (
            f"{sanitize_config_name(task_part)}_"
            f"{sanitize_config_name(category_part)}_"
            f"{sanitize_config_name(type_part)}_"
            f"{sanitize_config_name(metric_part)}_"
            f"{unique_suffix}"
        )

        # Store metadata
        for key in ("version", "author", "organization", "category", "task", "source"):
            if key in task_dict:
                self.metadata[key] = task_dict[key]

        # Extract core harness fields
        task_dict.pop("name", "Unknown Task")
        self.task_id = task_dict.pop("task", "unknown_task_id")
        self.metric = task_dict.pop("metric", None)
        self.category_id = task_dict.pop("category", None)

        # Handle data splitting (Taking last X elements for dev)
        raw_data = task_dict.pop("data", [])
        self.data = raw_data

        # Filter remaining metadata to keep YAML clean
        self.task_kwargs = {
            k: v for k, v in task_dict.items() if k not in self.IGNORED_METADATA
        }

    def _resolve_file_path(self, directory: str, file_name: str) -> Tuple[str, str]:
        """Resolve file path and extension.

        Args:
            directory: Directory to search
            file_name: File name (with or without extension)

        Returns:
            Tuple of (full_path, extension)

        Raises:
            FileNotFoundError: If file not found
        """
        base_path = os.path.join(directory, file_name)
        if os.path.isfile(base_path):
            return base_path, os.path.splitext(base_path)[1].lower()

        for ext in [".json", ".csv", ".xml"]:
            path = f"{base_path}{ext}"
            if os.path.isfile(path):
                return path, ext
        raise FileNotFoundError(f"File {file_name} not found in {directory}")

    def _load_json(self, path: str) -> Dict[str, Any]:
        """Load JSON file.

        Args:
            path: Path to JSON file

        Returns:
            Task dictionary
        """
        with open(path, "r", encoding="utf8") as fp:
            task: dict[str, Any] = json.load(fp)
            return task

    def _load_csv(self, path: str) -> Dict[str, Any]:
        """Load CSV file.

        Args:
            path: Path to CSV file

        Returns:
            Task dictionary
        """
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

        task["data"] = data_rows  # type: ignore[assignment]
        return task

    def _load_xml(self, path: str) -> Dict[str, Any]:
        """Load XML file.

        Args:
            path: Path to XML file

        Returns:
            Task dictionary
        """
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
                item: dict[str, Any] = {}
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
        task["data"] = data_rows  # type: ignore[assignment]
        return task

    def _escape_newline(self, task: Any) -> Any:
        """Escape newlines in task data.

        Args:
            task: Task data (can be dict, list, or string)

        Returns:
            Task with escaped newlines
        """
        if isinstance(task, dict):
            return {k: self._escape_newline(v) for k, v in task.items()}
        if isinstance(task, list):
            return [self._escape_newline(i) for i in task]
        if isinstance(task, str):
            return task.replace("\n", "\\n")
        return task

    def _normalize_schema(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize all items to have the same schema.

        Args:
            items: List of data items

        Returns:
            Normalized items with consistent schema
        """
        if not items:
            return items

        # Collect all unique keys
        all_keys: set[str] = set()
        for item in items:
            all_keys.update(item.keys())

        logger.info(
            "Found %d unique keys across %d items: %s",
            len(all_keys),
            len(items),
            sorted(all_keys),
        )

        # Define standard schema
        required_fields = {
            "id": None,
            "input": "",
            "output": "",
        }

        optional_fields = {
            "mcq": None,
            "instruction": "",
            "Experimental prompts": None,
            "source_link": "",
            "difficulty": "",
            "source": "",
        }

        standard_schema = {**required_fields, **optional_fields}

        # Add unexpected keys
        for key in all_keys:
            if key not in standard_schema:
                logger.warning(
                    "Found unexpected field '%s' in data, adding to schema", key
                )
                standard_schema[key] = ""

        # Normalize each item
        normalized = []
        for _, item in enumerate(items):
            normalized_item = {}

            for field, default_value in standard_schema.items():
                if field in item:
                    normalized_item[field] = item[field]
                else:
                    # Don't add empty list fields
                    if (
                        field in ["mcq", "Experimental prompts"]
                        and default_value is None
                    ):
                        continue
                    normalized_item[field] = default_value

            normalized.append(normalized_item)

        logger.info("Normalized %s items to consistent schema", len(normalized))
        return normalized

    def _export_data(self, split: str) -> None:
        """Save split data to JSON file.

        Args:
            split: Split name (test or dev)
        """
        items = self.data.get(split, [])
        if not items:
            return

        source_dir = self.source_directory
        processed = []

        for item in items:
            it = item.copy()

            # Handle input list - check for images
            if isinstance(it.get("input"), list):
                text_parts = []
                image_paths = []

                for input_item in it["input"]:
                    if _is_image_file(str(input_item)):
                        # Try multiple locations for the image
                        possible_paths = [
                            os.path.abspath(os.path.join(source_dir, str(input_item))),
                            os.path.abspath(os.path.join(".tasks", str(input_item))),
                            os.path.abspath(str(input_item)),
                        ]

                        abs_path = None
                        for path in possible_paths:
                            if os.path.exists(path):
                                abs_path = path
                                break

                        if abs_path:
                            image_paths.append(abs_path)
                        else:
                            image_paths.append(
                                os.path.abspath(
                                    os.path.join(source_dir, str(input_item))
                                )
                            )

                        text_parts.append("<image>")
                    else:
                        text_parts.append(str(input_item))

                # Join text with newlines
                it["input"] = "\n".join(text_parts)

                # Store image paths for doc_to_visual
                if image_paths:
                    it["images"] = image_paths
                    logger.info(
                        "Item %s contains %s image(s)", it.get("id"), len(image_paths)
                    )

            # Ensure output is a single string
            if isinstance(it.get("output"), list) and it["output"]:
                it["output"] = it["output"][0]
            processed.append(it)

        # Normalize schema
        processed = self._normalize_schema(processed)

        out_path = f"{self.directory}/{self.file_name}_{split}.json"
        with open(out_path, "w", encoding="utf8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)

        logger.info("Exported %s items to %s", len(processed), out_path)

    def validate(self) -> None:
        """Validate dataset has data."""
        if not any(self.data.values()):
            raise ValueError("Dataset data is empty.")

    def export(self) -> None:
        """Export dataset to JSON split files and YAML config."""
        self.validate()

        # Export split JSON files
        for split in ["test", "dev"]:
            self._export_data(split)

        # Build configuration
        base_yaml = self._build_base_yaml()

        # Apply metric logic from registry
        registry = get_metrics_registry()
        m_name = self.metric["metric"] if isinstance(self.metric, dict) else self.metric

        if not m_name:
            final_yaml = base_yaml
            logger.warning("No metric specified in dataset. Using base configuration.")
        else:
            detected_metric = registry.detect_metric_type(m_name)

            if detected_metric:
                metric_obj = registry.get(detected_metric)
                if metric_obj:
                    final_yaml = metric_obj.get_yaml_config(base_yaml)
                    logger.info("Using custom metric: %s", detected_metric)
                else:
                    logger.warning(
                        "Metric '%s' detected but couldn't retrieve object.",
                        detected_metric,
                    )
                    final_yaml = base_yaml.copy()
                    final_yaml["metric_list"] = [
                        {
                            "metric": m_name,
                            "aggregation": m_name,
                            "higher_is_better": True,
                        }
                    ]
            else:
                logger.info(
                    "Metric '%s' not found in custom registry. "
                    "Assuming it's a built-in LM Harness metric.",
                    m_name,
                )
                final_yaml = base_yaml.copy()
                final_yaml["metric_list"] = [
                    {
                        "metric": m_name,
                        "aggregation": m_name,
                        "higher_is_better": True,
                    }
                ]

        # Write YAML
        self._write_yaml(final_yaml)

    def _build_base_yaml(self) -> Dict[str, Any]:
        """Build base YAML configuration.

        Returns:
            Base YAML configuration
        """
        # Check if test split exists
        ref_items = self.data.get("test") or self.data.get("dev")
        ref_item = ref_items[0] if ref_items else {}

        # Clean doc_to_text
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

        # Check if dataset contains images
        has_images = any(
            _is_image_file(str(input_item))
            for items in self.data.values()
            for item in items
            if isinstance(item.get("input"), list)
            for input_item in item["input"]
        )

        # Build base YAML
        yaml_config = {
            "task": self.name,
            "dataset_path": "json",
            "dataset_name": sanitize_config_name(self.name),
            "test_split": "test" if "test" in data_files else None,
            "validation_split": "dev" if "dev" in data_files else None,
            "doc_to_text": doc_to_text,
            "doc_to_target": "output",
            "output_type": "generate_until",
            "generation_kwargs": {"do_sample": False, "until": [""]},
            "dataset_kwargs": {"data_files": data_files},
            "metadata": self.metadata,
            **self.task_kwargs,
        }

        # Add doc_to_visual if dataset contains images
        if has_images:
            yaml_config["doc_to_visual"] = "!function multimodal_utils.doc_to_visual"
            logger.info("Dataset contains images - adding doc_to_visual function")

        return yaml_config

    def _write_yaml(self, data: Dict[str, Any]) -> None:
        """Write YAML file with proper formatting.

        Args:
            data: YAML configuration dictionary
        """
        out_path = Path(self.directory) / f"{self.file_name}.yaml"

        # Extract doc_to_visual if it's a function tag
        doc_to_visual_value = data.pop("doc_to_visual", None)
        is_function_tag = (
            doc_to_visual_value
            and isinstance(doc_to_visual_value, str)
            and doc_to_visual_value.startswith("!function")
        )

        # Custom representer for multiline strings
        def str_representer(dumper, data):
            if "\n" in data:
                return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
            return dumper.represent_scalar("tag:yaml.org,2002:str", data)

        yaml.add_representer(str, str_representer)

        with open(out_path, "w", encoding="utf8") as f:
            yaml.dump(
                data,
                f,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
                width=1000,
            )

            # Add doc_to_visual without quotes if it's a function tag
            if is_function_tag:
                f.write(f"doc_to_visual: {doc_to_visual_value}\n")
