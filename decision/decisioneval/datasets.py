"""Read MCQ JSON datasets without modifying them.

Format: `{"Type of result": "MCQ", "name", "category", "task",
"data": {"test": [{"id", "instruction", "input", "mcq": [...], "output"}]}}`.
Only the test split is scored. An item that cannot be scored is excluded with
a reason rather than silently dropped.
"""

import json
from pathlib import Path

from decisioneval.models import (
    Dataset,
    ExcludedItem,
    LoadReport,
    Question,
    SkippedFile,
    option_id_for_index,
)

_MEDIA_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".webp",
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
}


class DatasetError(Exception):
    """Base error for dataset loading."""


class NotMCQDataset(DatasetError):
    """The file parsed, and it is not an MCQ dataset."""


class DatasetValidationError(DatasetError):
    """The file is MCQ, and its contents are not usable."""


class ItemValidationError(DatasetValidationError):
    """One test question cannot be scored. The rest of the dataset can continue."""

    def __init__(self, item_id: str | int | None, reason: str, detail: str):
        self.item_id = item_id
        self.reason = reason
        super().__init__(detail)


def load_paths(paths: list[str | Path] | tuple[str | Path, ...]) -> LoadReport:
    """Load several files or directories into one pilot selection."""
    if not paths:
        raise DatasetValidationError("no dataset path was given")
    datasets: list[Dataset] = []
    skipped: list[SkippedFile] = []
    for path in paths:
        report = _load_one(path)
        datasets.extend(report.datasets)
        skipped.extend(report.skipped)
    _reject_duplicate_names(datasets)
    return LoadReport(datasets=tuple(datasets), skipped=tuple(skipped))


def _load_one(path: str | Path) -> LoadReport:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise DatasetValidationError(f"path does not exist: {dataset_path}")
    if dataset_path.is_dir():
        return _load_directory(dataset_path)
    if dataset_path.is_file():
        return LoadReport(datasets=(load_dataset(dataset_path),), skipped=())
    raise DatasetValidationError(f"path is neither a file nor a directory: {dataset_path}")


def load_dataset(path: str | Path) -> Dataset:
    """Load one dataset file. Dev items are ignored. Test items are validated."""
    dataset_path = Path(path)
    payload = _read_json(dataset_path)
    if not isinstance(payload, dict):
        raise NotMCQDataset(f"{dataset_path}: the root is not a dataset object")
    _require_mcq(payload, dataset_path)

    name = _require_text(payload.get("name"), dataset_path, "name")
    category = _require_text(payload.get("category"), dataset_path, "category")
    task = _require_text(payload.get("task"), dataset_path, "task")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise DatasetValidationError(f"invalid MCQ data in {dataset_path}: the data field is missing")

    test_items = data.get("test")
    if not isinstance(test_items, list) or len(test_items) == 0:
        raise DatasetValidationError(
            f"invalid MCQ data in {dataset_path}: the test split is missing or empty"
        )

    questions: list[Question] = []
    excluded: list[ExcludedItem] = []
    for index, item in enumerate(test_items):
        try:
            questions.append(_parse_item(item, index, dataset_path, name, category, task))
        except ItemValidationError as exc:
            excluded.append(
                ExcludedItem(item_id=exc.item_id, reason=exc.reason, detail=str(exc))
            )
        except DatasetValidationError as exc:
            excluded.append(
                ExcludedItem(
                    item_id=_raw_item_id(item),
                    reason="invalid_item",
                    detail=str(exc),
                )
            )
    return Dataset(
        name=name,
        category=category,
        task=task,
        source_path=str(dataset_path),
        questions=tuple(questions),
        excluded=tuple(excluded),
    )


def _load_directory(directory: Path) -> LoadReport:
    files = sorted(path for path in directory.glob("*.json") if path.is_file())
    if not files:
        raise DatasetValidationError(f"no JSON files in {directory}")

    datasets: list[Dataset] = []
    skipped: list[SkippedFile] = []
    for file_path in files:
        try:
            datasets.append(load_dataset(file_path))
        except NotMCQDataset as exc:
            skipped.append(SkippedFile(path=str(file_path), reason=str(exc)))

    if not datasets:
        details = "\n".join(f"- {item.path}: {item.reason}" for item in skipped)
        raise DatasetValidationError(
            f"no usable MCQ files in {directory}. skipped: {len(skipped)}\n{details}"
        )

    _reject_duplicate_names(datasets)
    return LoadReport(datasets=tuple(datasets), skipped=tuple(skipped))


def _reject_duplicate_names(datasets: list[Dataset]) -> None:
    seen: dict[str, str] = {}
    for dataset in datasets:
        previous = seen.get(dataset.name)
        if previous is not None:
            raise DatasetValidationError(
                f"duplicate dataset name '{dataset.name}' in:\n- {previous}\n- {dataset.source_path}"
            )
        seen[dataset.name] = dataset.source_path


def _read_json(path: Path) -> object:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(f"could not parse JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise DatasetValidationError(f"could not open {path}: {exc}") from exc


def _require_mcq(payload: dict, path: Path) -> None:
    if "Type of result" not in payload:
        raise NotMCQDataset(f"{path}: the 'Type of result' field is missing")
    value = payload["Type of result"]
    if not isinstance(value, str) or value.strip().lower() != "mcq":
        raise NotMCQDataset(f"{path}: Type of result = {value!r}, not MCQ")


def _require_text(value: object, path: Path, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(
            f"invalid MCQ data in {path}: the {field} field is missing or empty"
        )
    return value.strip()


def _parse_item(
    item: object,
    index: int,
    path: Path,
    dataset_name: str,
    category: str,
    task: str,
) -> Question:
    label = f"question number {index + 1}"
    if not isinstance(item, dict):
        raise DatasetValidationError(
            f"invalid MCQ data in {path}: {label} is not an object"
        )
    if "id" not in item or item["id"] is None or isinstance(item["id"], bool):
        raise DatasetValidationError(
            f"invalid MCQ data in {path}: {label} has no usable id"
        )
    item_id = item["id"]
    if not isinstance(item_id, (str, int)):
        raise DatasetValidationError(
            f"invalid MCQ data in {path}: question {item_id!r} has an unsupported id type"
        )
    label = f"question {item_id}"

    instruction = item.get("instruction", "")
    if not isinstance(instruction, str):
        raise DatasetValidationError(
            f"invalid MCQ data in {path}: {label}: instruction must be a string"
        )

    text_input = _parse_input(item.get("input"), path, label)
    options = _parse_options(item.get("mcq"), path, label)
    gold_answer, gold_option_id = _resolve_gold(item.get("output"), options, item_id)
    option_ids = tuple(option_id_for_index(option_index) for option_index in range(len(options)))

    return Question(
        id=item_id,
        dataset_name=dataset_name,
        category=category,
        task=task,
        instruction=instruction.strip(),
        input=text_input,
        options=tuple(options),
        option_ids=option_ids,
        gold_answer=gold_answer,
        gold_option_id=gold_option_id,
    )


def _parse_input(value: object, path: Path, label: str) -> str:
    if not isinstance(value, list) or not value:
        raise DatasetValidationError(
            f"invalid MCQ data in {path}: {label}: input must be a non-empty list of strings"
        )
    parts: list[str] = []
    for part in value:
        if not isinstance(part, str):
            raise DatasetValidationError(
                f"invalid MCQ data in {path}: {label}: an input element is not a string"
            )
        suffix = Path(part.strip()).suffix.lower()
        if suffix in _MEDIA_SUFFIXES:
            raise DatasetValidationError(
                f"invalid MCQ data in {path}: {label}: input holds an unsupported media file ({part})"
            )
        text = part.strip()
        if text:
            parts.append(text)
    if not parts:
        raise DatasetValidationError(
            f"invalid MCQ data in {path}: {label}: input is empty"
        )
    return "\n".join(parts)


def _parse_options(value: object, path: Path, label: str) -> list[str]:
    if not isinstance(value, list):
        raise DatasetValidationError(
            f"invalid MCQ data in {path}: {label}: mcq must be a list of options"
        )
    options: list[str] = []
    for option in value:
        if not isinstance(option, str) or not option.strip():
            raise DatasetValidationError(
                f"invalid MCQ data in {path}: {label}: an option is empty or not a string"
            )
        options.append(option.strip())
    if len(options) < 2:
        raise DatasetValidationError(
            f"invalid MCQ data in {path}: {label}: an MCQ needs at least two options"
        )
    return options


def _resolve_gold(
    value: object, options: list[str], item_id: str | int
) -> tuple[str, str]:
    """Return the gold text and its single option id.

    Duplicate distractors are allowed. The question is excluded when the
    output matches nothing, or when the same gold text occupies more than
    one position.
    """
    if not isinstance(value, str) or not value.strip():
        raise ItemValidationError(
            item_id,
            "invalid_output",
            f"question {item_id}: output must be the text of the correct option",
        )
    gold = value.strip()
    matches = [index for index, option in enumerate(options) if option == gold]
    if not matches:
        rendered = ", ".join(options)
        raise ItemValidationError(
            item_id,
            "gold_not_in_options",
            f"question {item_id}: gold answer '{gold}' is not among the options ({rendered})",
        )
    if len(matches) > 1:
        positions = ", ".join(option_id_for_index(index) for index in matches)
        raise ItemValidationError(
            item_id,
            "ambiguous_gold",
            f"question {item_id}: gold answer '{gold}' matches {len(matches)} options ({positions})",
        )
    match = matches[0]
    return options[match], option_id_for_index(match)


def _raw_item_id(item: object) -> str | int | None:
    if not isinstance(item, dict) or "id" not in item or isinstance(item["id"], bool):
        return None
    if isinstance(item["id"], (str, int)):
        return item["id"]
    return None
