"""Focused coverage for inexpensive defensive and configuration branches."""

import logging
from unittest.mock import MagicMock

import pytest

from src.adapters import utils as adapter_utils
from src.core import helpers
from src.llm_judger.generative_llm_judge import GenerativeLLMJudge
from src.llm_judger.mcq_llm_judge import MCQLLMJudge
from src.metrics import accuracy_metric
from src.metrics.accuracy_metric import AccuracyMetric
from src.metrics.asr_utils import extract_text_from_prediction
from src.metrics_registry import MetricConfig
from src.processors.result_processing import ResultProcessor


def make_processor(tmp_path):
    return ResultProcessor(
        category="category",
        task_id="task",
        source_pool_path="pool.json",
        results_dir=str(tmp_path),
    )


@pytest.mark.parametrize(
    ("destination", "expected_parent"),
    [("result.json", None), ("nested/result.json", "nested")],
)
def test_gcs_download_uses_blob_and_only_creates_present_parent(
    monkeypatch, destination, expected_parent
):
    client = MagicMock()
    monkeypatch.setattr(helpers.storage, "Client", MagicMock(return_value=client))
    makedirs = MagicMock()
    monkeypatch.setattr(helpers.os, "makedirs", makedirs)

    helpers.download_pool_file_from_gcs("bucket", "pools/input.json", destination)

    client.bucket.assert_called_once_with("bucket")
    blob = client.bucket.return_value.blob
    blob.assert_called_once_with("pools/input.json")
    blob.return_value.download_to_filename.assert_called_once_with(destination)
    if expected_parent is None:
        makedirs.assert_not_called()
    else:
        makedirs.assert_called_once_with(expected_parent, exist_ok=True)


def test_gcs_upload_sets_json_content_type(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(helpers.storage, "Client", MagicMock(return_value=client))

    helpers.upload_result_file_to_gcs("bucket", "local.json", "results/out.json")

    client.bucket.assert_called_once_with("bucket")
    blob = client.bucket.return_value.blob
    blob.assert_called_once_with("results/out.json")
    blob.return_value.upload_from_filename.assert_called_once_with(
        "local.json", content_type="application/json; charset=utf-8"
    )


def test_result_scoring_ignores_missing_and_malformed_sample_structures(tmp_path):
    processor = make_processor(tmp_path)
    sample = {"metrics": ["absent"], "scores": {"stale": True}}
    results = {
        "samples": {
            "not-a-list": "bad",
            "mixed": [None, sample],
        }
    }

    processor._add_question_scores(results)

    assert sample["scores"] == {}
    processor._add_question_scores({})
    processor._add_question_scores({"samples": []})


def test_result_averaging_ignores_missing_and_unsupported_values(tmp_path):
    processor = make_processor(tmp_path)

    assert processor._calculate_average_scores({}) == {}
    assert processor._calculate_average_scores({"results": []}) == {}
    assert processor._calculate_average_scores(
        {"results": {"task": {"rouge,none": {"rouge1": 1}, "score,none": None}}}
    ) == {}


def test_result_cleanup_handles_missing_and_irregular_arguments(tmp_path):
    processor = make_processor(tmp_path)
    missing = {"results": {}}
    sample = {
        "arguments": [["short"], ("request", {}, {"audio": [1], "keep": 2})],
        "doc": None,
    }
    results = {"samples": {"invalid": None, "task": [sample]}}

    assert processor._strip_multimodal_data(missing) is missing
    processor._strip_audio_data(results)

    assert sample["arguments"][1][2] == {"keep": 2}


@pytest.mark.parametrize(
    ("adapter", "model", "expected"),
    [
        ("local-chat-completions", "QwQ-32B", {"max_tokens": 8192}),
        ("anthropic", "extended-thinking", {"max_tokens": 8192}),
        ("google-stt", "unused", {"max_tokens": 4096}),
    ],
)
def test_adapter_edge_token_configs(monkeypatch, adapter, model, expected):
    monkeypatch.delenv("IS_REASONING", raising=False)

    assert adapter_utils.get_max_tokens_config(adapter, model) == expected


def test_unknown_thinking_adapter_has_no_special_config():
    assert adapter_utils._get_thinking_model_config("unknown", "thinking") is None


def test_anthropic_conversion_verbose_mode_reports_adapter_and_url(capsys, caplog):
    with caplog.at_level(logging.INFO, logger=adapter_utils.__name__):
        result = adapter_utils.process_adapter_and_url(
            "anthropic-chat-completions", None, verbose=True
        )

    expected_url = "https://api.anthropic.com/v1/chat/completions"
    assert result == ("local-chat-completions", expected_url)
    assert capsys.readouterr().out.splitlines() == [
        "Converting anthropic-chat-completions to local-chat-completions",
        f"Using base_url: {expected_url}",
    ]
    assert "Using base_url" in caplog.text


def test_fuzzy_accuracy_accepts_partial_match_when_full_ratio_is_low(monkeypatch):
    ratio = MagicMock(return_value=20)
    partial_ratio = MagicMock(return_value=90)
    monkeypatch.setattr(accuracy_metric.fuzz, "ratio", ratio)
    monkeypatch.setattr(accuracy_metric.fuzz, "partial_ratio", partial_ratio)

    assert accuracy_metric.compute_accuracy(
        [("long reference", "reference")], use_fuzzy=True
    ) == 1.0
    ratio.assert_called_once_with("long reference", "reference")
    partial_ratio.assert_called_once_with("long reference", "reference")


def test_fuzzy_processing_resolves_mcq_labels_and_empty_results():
    doc = {"output": "beta", "mcq": ["alpha", "beta"]}

    assert accuracy_metric.process_results_fuzzy(doc, ["Choice: B"]) == {
        "fuzzy_accuracy": ["beta", "beta"]
    }
    assert accuracy_metric.process_results_fuzzy(doc, None) == {
        "fuzzy_accuracy": ["beta", ""]
    }


def test_accuracy_metric_class_adds_mcq_template_and_generation_settings():
    metric = AccuracyMetric(MetricConfig(name="accuracy"))

    template = metric.get_doc_to_text("{{ input }}")
    assert template.startswith("{{ input }}\n{% if mcq %}")
    assert "ABCDEFGHIJKLMNOPQRSTUVWXYZ" in template
    assert metric.get_generation_kwargs() == {"do_sample": False, "until": []}


def test_asr_unparseable_double_quoted_prediction_strips_outer_quotes():
    assert extract_text_from_prediction('"unescaped "quote""') == 'unescaped "quote"'


@pytest.mark.parametrize(
    ("judge_class", "maximum", "prompt_fragment"),
    [
        (GenerativeLLMJudge, 3.0, "Score 3"),
        (MCQLLMJudge, 1.0, "Arabic–English equivalence"),
    ],
)
def test_judge_subclasses_expose_scale_and_specialized_prompt(
    judge_class, maximum, prompt_fragment
):
    judge = object.__new__(judge_class)

    assert judge.get_max_score() == maximum
    assert prompt_fragment in judge.get_evaluation_prompt()


@pytest.mark.parametrize("judge_class", [GenerativeLLMJudge, MCQLLMJudge])
def test_judge_normalization_rejects_none_and_non_numeric_objects(judge_class):
    judge = object.__new__(judge_class)

    assert judge.normalize_score(None) == 0.0
    assert judge.normalize_score(object()) == 0.0
