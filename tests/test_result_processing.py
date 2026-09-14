import json
from unittest.mock import Mock

import numpy as np
import pytest

from src.processors.result_processing import ResultProcessor, _NumpyEncoder


def make_processor(tmp_path):
    return ResultProcessor(
        category="category",
        task_id="task-id",
        source_pool_path="pools/source.json",
        results_dir=str(tmp_path),
    )


def test_numpy_encoder_serializes_arrays_and_scalars():
    encoded = json.loads(
        json.dumps(
            {"array": np.array([1, 2]), "int": np.int64(3), "float": np.float32(1.5)},
            cls=_NumpyEncoder,
        )
    )

    assert encoded == {"array": [1, 2], "int": 3, "float": 1.5}


def test_calculate_average_scores_collects_supported_metric_shapes(tmp_path):
    processor = make_processor(tmp_path)
    results = {
        "results": {
            "first": {
                "accuracy,none": 0.12344,
                "rouge,none": {"rougeLsum": 0.8},
                "wer,none": 2.5,
                "ignored,stderr": 10,
            },
            "second": {"accuracy,none": 0.12346},
        }
    }

    assert processor._calculate_average_scores(results) == {
        "accuracy": 0.1235,
        "rouge": 0.8,
        "wer": 2.5,
    }


def test_strip_multimodal_data_copies_samples_without_mutating_input(tmp_path):
    processor = make_processor(tmp_path)
    original = {
        "samples": {
            "task": [
                {
                    "arguments": [
                        ["request", {}, {"audio": [1], "images": [2], "keep": 3}],
                        "unchanged",
                    ]
                }
            ]
        }
    }

    cleaned = processor._strip_multimodal_data(original)

    assert cleaned["samples"]["task"][0]["arguments"] == [
        ["request", {}, {"keep": 3}],
        "unchanged",
    ]
    assert original["samples"]["task"][0]["arguments"][0][2]["audio"] == [1]


def test_add_question_scores_uses_registered_aggregation(monkeypatch, tmp_path):
    processor = make_processor(tmp_path)
    sample = {
        "metrics": ["accuracy"],
        "accuracy": ["a", "a"],
        "scores": {"old": 1},
    }

    def get_aggregation(name):
        if name == "accuracy":
            return lambda items: 1.0 if items == [["a", "a"]] else 0.0
        return None

    monkeypatch.setattr(
        "src.processors.result_processing.get_metric_aggregation", get_aggregation
    )

    processor._add_question_scores({"samples": {"task": [sample]}})

    assert sample["scores"] == {"accuracy": 1.0}


@pytest.mark.parametrize("failure", ["missing", "broken"])
def test_add_question_scores_propagates_required_aggregation_failure(
    monkeypatch, tmp_path, failure
):
    processor = make_processor(tmp_path)
    sample = {"metrics": [failure], failure: [1]}
    aggregation = None if failure == "missing" else lambda _items: 1 / 0
    monkeypatch.setattr(
        "src.processors.result_processing.get_metric_aggregation",
        lambda _name: aggregation,
    )

    if failure == "missing":
        with pytest.raises(RuntimeError, match="Required metric aggregation"):
            processor._add_question_scores({"samples": {"task": [sample]}})
    else:
        with pytest.raises(ZeroDivisionError):
            processor._add_question_scores({"samples": {"task": [sample]}})


def test_add_question_scores_rejects_missing_required_metric_result(tmp_path):
    sample = {"metrics": ["accuracy"]}

    with pytest.raises(RuntimeError, match="Required metric result is missing"):
        make_processor(tmp_path)._add_question_scores(
            {"samples": {"task": [sample]}}
        )


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "bad"])
def test_add_question_scores_rejects_invalid_aggregation_values(
    monkeypatch, tmp_path, value
):
    monkeypatch.setattr(
        "src.processors.result_processing.get_metric_aggregation",
        lambda _name: lambda _items: value,
    )

    with pytest.raises(RuntimeError, match="Invalid per-question metric value"):
        make_processor(tmp_path)._add_question_scores(
            {"samples": {"task": [{"metrics": ["accuracy"], "accuracy": [1]}]}}
        )


@pytest.mark.parametrize("value", [True, float("nan"), float("inf")])
def test_calculate_average_scores_rejects_invalid_numeric_values(tmp_path, value):
    with pytest.raises(RuntimeError, match="Invalid aggregate metric value"):
        make_processor(tmp_path)._calculate_average_scores(
            {"results": {"task": {"accuracy,none": value}}}
        )


def test_export_enriches_result_and_removes_raw_media(monkeypatch, tmp_path):
    processor = make_processor(tmp_path)
    monkeypatch.setattr(
        "src.processors.result_processing.get_metric_aggregation",
        lambda _name: lambda _items: np.float32(1.0),
    )
    results = {
        "results": {"task": {"accuracy,none": 1.0}},
        "samples": {
            "task": [
                {
                    "metrics": ["accuracy"],
                    "accuracy": ["yes", "yes"],
                    "arguments": [["request", {}, {"audio": [1], "keep": True}]],
                    "doc": {"audio": [1], "text": "مرحبا"},
                }
            ]
        },
    }

    path = processor.export(results, filename="result.json")

    with open(path, encoding="utf-8") as result_file:
        exported = json.load(result_file)
    sample = exported["samples"]["task"][0]
    assert exported["average_scores"] == {"accuracy": 1.0}
    assert exported["category"] == "category"
    assert exported["task"] == "task-id"
    assert exported["pool_file"] == "pools/source.json"
    assert sample["scores"] == {"accuracy": 1.0}
    assert sample["arguments"][0][2] == {"keep": True}
    assert sample["doc"] == {"text": "مرحبا"}


def test_export_rejects_missing_aggregate_metrics(tmp_path):
    processor = make_processor(tmp_path)

    with pytest.raises(RuntimeError, match="no aggregate metric"):
        processor.export({"results": {"task": {}}}, filename="result.json")

    assert not (tmp_path / "result.json").exists()


def test_export_validates_aggregate_metrics_before_question_aggregation(
    monkeypatch, tmp_path
):
    aggregation = Mock()
    monkeypatch.setattr(
        "src.processors.result_processing.get_metric_aggregation", aggregation
    )

    with pytest.raises(RuntimeError, match="no aggregate metric"):
        make_processor(tmp_path).export(
            {
                "results": {"task": {}},
                "samples": {"task": [{"metrics": ["llm_as_judge"]}]},
            },
            filename="result.json",
        )

    aggregation.assert_not_called()


def test_average_scores_rejects_empty_task_even_when_another_task_is_valid(tmp_path):
    with pytest.raises(RuntimeError, match="has no aggregate metric values"):
        make_processor(tmp_path)._calculate_average_scores(
            {
                "results": {
                    "valid": {"accuracy,none": 1.0},
                    "missing": {"accuracy_stderr,none": 0.0},
                }
            }
        )


@pytest.mark.parametrize("value", [None, "bad", {"rouge1": 1.0}])
def test_average_scores_rejects_unsupported_values_with_valid_peer(
    tmp_path, value
):
    with pytest.raises(RuntimeError, match="Unsupported aggregate metric value"):
        make_processor(tmp_path)._calculate_average_scores(
            {
                "results": {
                    "task": {
                        "accuracy,none": 1.0,
                        "other,none": value,
                    }
                }
            }
        )
