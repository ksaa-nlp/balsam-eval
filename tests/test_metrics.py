from unittest.mock import Mock

import pytest

from src.metrics.accuracy_metric import (
    compute_accuracy,
    compute_fuzzy_accuracy,
    extract_first_word_or_line,
    normalize_text,
    process_results,
    resolve_mcq_answer,
)
from src.metrics.asr_utils import extract_text_from_prediction, process_results_asr
import src.metrics_combined as combined_module
from src.metrics_combined import (
    CombinedMetricsRegistry,
    _current_combined_process_results,
    create_combined_process_results,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Answer: A.", "A"),
        ("الإجابة: ب", "ب"),
        ("short answer!", "short answer"),
        ("first line\nsecond line", "first line"),
        ("this sentence has many words", "this sentence has many words"),
        ("  ", ""),
    ],
)
def test_extract_first_word_or_line(value, expected):
    assert extract_first_word_or_line(value) == expected


def test_normalize_text_handles_case_punctuation_and_arabic_marks():
    assert normalize_text(" Hello,   WORLD! ") == "hello world"
    assert normalize_text("مَـرْحَبًا") == "مرحبا"
    assert normalize_text("a", {"A": "Mapped Value"}) == "Mapped Value"
    assert normalize_text(None) == ""


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("B)", "beta"), ("2", "beta"), ("ب", "beta"), ("ALPHA", "alpha")],
)
def test_resolve_mcq_answer_supports_labels_and_option_text(answer, expected):
    assert resolve_mcq_answer(answer, ["alpha", "beta"]) == expected


def test_compute_accuracy_skips_empty_references_and_counts_empty_predictions():
    items = [("A", "a."), ("B", "wrong"), (None, "ignored"), ("", "ignored")]

    assert compute_accuracy(items) == 0.5
    assert compute_accuracy([("answer", "")]) == 0.0
    assert compute_accuracy([]) == 0.0


def test_compute_fuzzy_accuracy_uses_default_threshold(monkeypatch):
    monkeypatch.setattr("src.metrics.accuracy_metric.fuzz.ratio", lambda *_: 85)
    monkeypatch.setattr("src.metrics.accuracy_metric.fuzz.partial_ratio", lambda *_: 0)

    assert compute_fuzzy_accuracy([("reference", "prediction")]) == 1.0


def test_accuracy_process_results_resolves_prediction_to_option():
    doc = {"output": "beta", "mcq": ["alpha", "beta"]}

    assert process_results(doc, ["Answer: B."]) == {"accuracy": ["beta", "beta"]}


@pytest.mark.parametrize(
    ("gold", "prediction"),
    [
        ("The Colosseum", "The Colosseum (Flavian Amphitheatre) in Rome, Italy."),
        (
            "Pretoria",
            "- South Africa has three capitals.\nCurrent administrative capital: "
            "Pretoria (also called Tshwane).",
        ),
    ],
)
def test_accuracy_process_results_extracts_multiword_answer_from_verbose_response(
    gold, prediction
):
    result = process_results({"output": gold}, [prediction])

    assert result == {"accuracy": [gold, gold]}
    assert compute_accuracy([result["accuracy"]]) == 1.0


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ""),
        (7, "7"),
        ('{"text": "hello"}', "hello"),
        ('{"text": null}', ""),
        ('{"answer": "first", "other": "second"}', "first"),
        ('"quoted"', "quoted"),
        ("'single quoted'", "single quoted"),
        (" plain text ", "plain text"),
        ("[1, 2]", "[1, 2]"),
    ],
)
def test_extract_text_from_prediction(value, expected):
    assert extract_text_from_prediction(value) == expected


def test_process_results_asr_uses_first_prediction_and_dynamic_metric_name():
    assert process_results_asr(
        {"output": "reference"}, ['{"text":"prediction"}', "ignored"], "wer"
    ) == {"wer": ["reference", "prediction"]}
    assert process_results_asr({"output": "reference"}, [], "cer") == {
        "cer": ["reference", ""]
    }


@pytest.fixture(autouse=True)
def restore_combined_registry(monkeypatch):
    monkeypatch.setattr(CombinedMetricsRegistry, "_functions", {})
    monkeypatch.setattr(combined_module, "CURRENT_COMBINED_FUNCTION", None)


def test_combined_registry_registers_retrieves_and_reports_missing():
    function = Mock()
    CombinedMetricsRegistry.register("both", function)

    assert CombinedMetricsRegistry.get("both") is function
    with pytest.raises(KeyError, match=r"Available: \['both'\]"):
        CombinedMetricsRegistry.get("missing")


def test_combined_process_results_merges_metric_outputs():
    first = Mock(return_value={"accuracy": ["a", "a"]})
    second = Mock(return_value={"wer": ["x", "y"]})
    function = create_combined_process_results(
        [
            {"name": "accuracy", "process_results": first},
            {"name": "wer", "process_results": second},
            {"name": "without-processor"},
        ]
    )

    result = function({"output": "a"}, ["a"])

    assert result == {"accuracy": ["a", "a"], "wer": ["x", "y"]}
    assert repr(function) == (
        "<CombinedFunction(metrics=['accuracy', 'wer', 'without-processor'])>"
    )


def test_combined_process_results_rejects_duplicate_keys():
    function = create_combined_process_results(
        [
            {"name": "first", "process_results": lambda *_: {"score": 1}},
            {"name": "second", "process_results": lambda *_: {"score": 2}},
        ]
    )

    with pytest.raises(ValueError, match="Duplicate result keys.*second.*score"):
        function({}, [])


def test_current_combined_process_results_requires_and_delegates_function(monkeypatch):
    with pytest.raises(RuntimeError, match="No combined function registered"):
        _current_combined_process_results({}, [])

    function = Mock(return_value={"score": 1})
    monkeypatch.setattr(combined_module, "CURRENT_COMBINED_FUNCTION", function)
    assert _current_combined_process_results({"id": 1}, ["result"]) == {"score": 1}
    function.assert_called_once_with({"id": 1}, ["result"])
