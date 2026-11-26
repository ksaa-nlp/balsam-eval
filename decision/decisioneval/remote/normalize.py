"""Platform (BALSAM) pool file -> decisioneval MCQ dataset named after its object path."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

TYPE_HEADER = "type_of_result"
MCQ = "mcq"
TEST_SPLIT = "test"
ITEM_FIELDS = ("id", "instruction", "input", "mcq", "output")
# luna.py's fallback, so no protocol ever sends a model an empty instruction.
DEFAULT_INSTRUCTION = "اختر الإجابة الصحيحة."


class NotMcqPool(ValueError):
    """The pool file is not something a decision model can be scored on."""


@dataclass(frozen=True)
class PoolDocument:
    source: str
    document: dict[str, Any]
    category: str
    task: str
    dataset_item_ids: Mapping[str, Any] = field(default_factory=dict)

    def dataset_item_id(self, item_id: Any) -> Any:
        return self.dataset_item_ids.get(str(item_id))


def normalize_pool(raw: Any, source: str) -> PoolDocument:
    """Normalise one parsed pool file. Raises `NotMcqPool` with the reason."""
    task_file = _unwrap(raw)
    kind = _type_of_result(task_file)
    if kind != MCQ:
        raise NotMcqPool(f"type of result is {kind or 'missing'}, not MCQ")

    category = _text(task_file.get("category"))
    task = _text(task_file.get("task"))
    items = [_item(row) for row in _test_rows(task_file.get("data")) if isinstance(row, Mapping)]
    if not items:
        raise NotMcqPool("the test split is missing or empty")

    return PoolDocument(
        source=source,
        document={
            "Type of result": "MCQ",
            "name": source,
            "category": category or "uncategorized",
            "task": task or source,
            "data": {TEST_SPLIT: [item for item, _ in items]},
        },
        category=category,
        task=task,
        dataset_item_ids={
            str(item["id"]): dataset_item_id
            for item, dataset_item_id in items
            if dataset_item_id is not None and "id" in item
        },
    )


def _unwrap(raw: Any) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise NotMcqPool("the pool file is not a JSON object")
    inner = raw.get("json")
    if not isinstance(inner, Mapping):
        return raw
    merged = dict(inner)
    for key in ("category", "task"):
        if key in raw:
            merged[key] = raw[key]
    return merged


def _type_of_result(task_file: Mapping[str, Any]) -> str:
    for key, value in task_file.items():
        if isinstance(key, str) and key.strip().lower().replace(" ", "_") == TYPE_HEADER:
            return str(value or "").strip().lower()
    return ""


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _test_rows(data: Any) -> list[Any]:
    if isinstance(data, Mapping):
        rows = data.get(TEST_SPLIT)
        return list(rows) if isinstance(rows, list) else []
    if isinstance(data, list):
        return [
            row
            for row in data
            if not isinstance(row, Mapping)
            or _text(row.get("split_type")).lower() in ("", TEST_SPLIT)
        ]
    return []


def _item(row: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
    item = {key: row[key] for key in ITEM_FIELDS if key in row}
    if isinstance(item.get("input"), str):
        item["input"] = [item["input"]]
    if not _text(item.get("instruction")):
        item["instruction"] = DEFAULT_INSTRUCTION
    return item, row.get("dataset_item_id")
