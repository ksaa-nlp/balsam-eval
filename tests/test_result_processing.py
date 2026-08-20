import json

import numpy as np

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
            "second": {"accuracy,none": 0.12346, "invalid,none": "value"},
            "invalid": [],
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
        "metrics": ["accuracy", "missing", "broken"],
        "accuracy": ["a", "a"],
        "missing": [1],
        "broken": [2],
        "scores": {"old": 1},
    }

    def get_aggregation(name):
        if name == "accuracy":
            return lambda items: 1.0 if items == [["a", "a"]] else 0.0
        if name == "broken":
            return lambda _items: 1 / 0
        return None

    monkeypatch.setattr(
        "src.processors.result_processing.get_metric_aggregation", get_aggregation
    )

    processor._add_question_scores({"samples": {"task": [sample]}})

    assert sample["scores"] == {"accuracy": 1.0}


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
