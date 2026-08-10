"""Exercise remaining reachable low-cost coverage edges."""

import builtins
import importlib
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import src.task as task_module
from src import cli, evaluation
from src.core import common
from src.core.config import EvalConfig
from src.metrics import accuracy_metric, bleu_metric, metrics_utils, rouge_metric
from src.processors.result_processing import ResultProcessor, _NumpyEncoder
from src.task import LMHDataset


def test_cli_module_entrypoint_invokes_main(monkeypatch):
    runner = Mock()
    monkeypatch.setattr(cli.runner, "main", runner)
    monkeypatch.setattr(sys, "argv", ["balsam-eval"])

    runpy.run_path(cli.__file__, run_name="__main__")

    runner.assert_called_once_with()


def test_failed_media_download_preserves_reference(monkeypatch, tmp_path, capsys):
    data_file = tmp_path / "data.json"
    data_file.write_text(json.dumps([{"audio": ["missing.wav"]}]), encoding="utf-8")
    client = Mock()
    client.bucket.return_value.blob.return_value.download_to_filename.side_effect = (
        RuntimeError("download failed")
    )
    monkeypatch.setattr(common.storage, "Client", Mock(return_value=client))

    common.copy_audio_to_temp(str(data_file), str(tmp_path / "temp"), bucket="bucket")

    assert json.loads(data_file.read_text(encoding="utf-8")) == [
        {"audio": ["missing.wav"]}
    ]
    assert "Could not fetch gs://bucket/missing.wav" in capsys.readouterr().out


def test_missing_local_media_without_bucket_preserves_reference(tmp_path):
    data_file = tmp_path / "data.json"
    data_file.write_text(json.dumps([{"images": ["missing.png"]}]), encoding="utf-8")

    common.copy_images_to_temp(str(data_file), str(tmp_path / "temp"))

    assert json.loads(data_file.read_text(encoding="utf-8")) == [
        {"images": ["missing.png"]}
    ]


def test_empty_config_lists(monkeypatch):
    monkeypatch.delenv("POOL_FILES", raising=False)

    assert EvalConfig._parse_csv_env("POOL_FILES") == []
    assert EvalConfig().get_evaluation_types_list() == []


def test_compatibility_patch_reraises_unrelated_value_error(monkeypatch):
    def fail(_self, *_args, **_kwargs):
        raise ValueError("different drives")

    monkeypatch.setattr(Path, "relative_to", fail)
    with evaluation._lm_eval_compatibility_patches():
        with pytest.raises(ValueError, match="different drives"):
            Path("child").relative_to("parent")
    assert Path.relative_to is fail


def test_api_key_noop_and_unknown_adapter_branches(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    evaluation._set_api_key_env("openai", None)
    evaluation._set_api_key_env("unknown", "secret")

    assert "OPENAI_API_KEY" not in evaluation.os.environ


def test_job_preserves_supplied_eos_string(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)

    job = evaluation.SingleFileEvaluationJob(
        task_name="task",
        category="category",
        task_id="id",
        source_pool_path="pool.json",
        adapter="unknown",
        model_args={"model": "model", "eos_string": "<end>"},
        result_filename="result.json",
        results_dir="results",
    )

    assert job.model_args["eos_string"] == "<end>"


def test_accuracy_scalar_conversion_and_remaining_decision_edges():
    assert accuracy_metric.extract_first_word_or_line(42) == "42"
    assert accuracy_metric.normalize_text(42) == "42"
    assert accuracy_metric.extract_first_word_or_line("Note: value") == "Note: value"
    assert accuracy_metric.resolve_mcq_answer("Z", ["one", "two"]) == "Z"
    assert accuracy_metric.resolve_mcq_answer("9", ["one", "two"]) == "9"
    assert accuracy_metric.compute_accuracy(
        [("reference", "different")], use_fuzzy=True
    ) == 0.0


def test_accuracy_registration_guards_when_already_registered():
    reloaded = importlib.reload(accuracy_metric)

    assert reloaded is accuracy_metric


def test_bleu_registration_executes_when_metric_is_absent():
    registry = bleu_metric.le_registry.METRIC_REGISTRY
    aggregation_registry = bleu_metric.le_registry.METRIC_AGGREGATION_REGISTRY
    snapshot = registry._objs.copy()
    aggregation_snapshot = aggregation_registry._objs.copy()
    registry._objs.pop("bleu", None)
    aggregation_registry._objs.pop("bleu", None)
    try:
        importlib.reload(bleu_metric)
        assert "bleu" in registry
    finally:
        registry._objs.clear()
        registry._objs.update(snapshot)
        aggregation_registry._objs.clear()
        aggregation_registry._objs.update(aggregation_snapshot)


def test_metrics_utils_handles_missing_optional_pyarabic(monkeypatch):
    real_import = builtins.__import__
    original_araby = metrics_utils.araby

    def import_without_pyarabic(name, *args, **kwargs):
        if name == "pyarabic":
            raise ImportError("optional dependency unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_pyarabic)
    try:
        importlib.reload(metrics_utils)
        assert metrics_utils.araby is None
    finally:
        metrics_utils.araby = original_araby


def test_rouge_metric_registration_and_identity_function(monkeypatch):
    registry = rouge_metric.le_registry.METRIC_REGISTRY
    snapshot = registry._objs.copy()
    registry._objs.pop("rouge", None)
    monkeypatch.setattr(rouge_metric.evaluate, "load", Mock(return_value=Mock()))
    try:
        importlib.reload(rouge_metric)
        items = [("reference", "prediction")]
        assert rouge_metric.rouge_fn(items) is items
        importlib.reload(rouge_metric)
    finally:
        registry._objs.clear()
        registry._objs.update(snapshot)


def test_numpy_encoder_delegates_unsupported_values():
    with pytest.raises(TypeError, match="not JSON serializable"):
        _NumpyEncoder().encode(object())


def test_result_cleanup_preserves_sample_without_argument_list():
    results = {"samples": {"task": [{"arguments": "scalar", "doc": {"id": 1}}]}}

    cleaned = ResultProcessor._strip_multimodal_data(results)

    assert cleaned == results


def test_csv_loader_skips_blank_data_rows(tmp_path):
    path = tmp_path / "blank-row.csv"
    path.write_text(
        "task,metric\nExample,accuracy\nid,input,output,split_type\n,,,\n1,q,a,test\n",
        encoding="utf-8",
    )

    loaded = object.__new__(LMHDataset)._load_csv(str(path))

    assert [item["id"] for item in loaded["data"]["test"]] == ["1"]


def test_xml_loader_handles_empty_metrics_and_missing_data(tmp_path):
    path = tmp_path / "minimal.xml"
    path.write_text("<dataset><metrics><metric /></metrics></dataset>", encoding="utf-8")

    loaded = object.__new__(LMHDataset)._load_xml(str(path))

    assert "metrics" not in loaded
    assert loaded["data"] == {"dev": [], "test": [], "train": []}


def test_export_data_uses_fallback_path_for_missing_audio(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    dataset = object.__new__(LMHDataset)
    dataset.data = {
        "test": [{"id": "1", "input": ["missing.wav"], "output": "answer"}]
    }
    dataset.source_directory = str(source)
    dataset.directory = str(tmp_path)
    dataset.file_name = "export"
    dataset.custom_prompt = None

    dataset._export_data("test")

    exported = json.loads((tmp_path / "export_test.json").read_text(encoding="utf-8"))
    assert exported[0]["audio"] == [str((source / "missing.wav").resolve())]
    assert exported[0]["input"] == "<audio>"


def test_export_custom_metric_without_processor_omits_wrapper(monkeypatch, tmp_path):
    metric = Mock()
    metric.config = SimpleNamespace(
        name="custom",
        aggregation_name="custom",
        higher_is_better=True,
        process_results=None,
    )
    metric.get_yaml_config.side_effect = lambda base: {
        **base,
        "metric_list": [
            {"metric": "custom", "aggregation": "custom", "higher_is_better": True}
        ],
    }
    registry = Mock()
    registry.detect_metric_type.return_value = "custom"
    registry.get.return_value = metric
    monkeypatch.setattr(task_module, "get_metrics_registry", lambda: registry)

    dataset = object.__new__(LMHDataset)
    dataset.data = {
        "test": [{"id": "1", "input": "question", "output": "answer"}],
        "dev": [],
        "train": [],
    }
    dataset.directory = str(tmp_path)
    dataset.source_directory = str(tmp_path)
    dataset.file_name = "custom"
    dataset.name = "custom"
    dataset.metadata = {}
    dataset.task_kwargs = {}
    dataset.custom_prompt = None
    dataset.metric = "custom"

    dataset.export()

    assert "process_results" not in (tmp_path / "custom.yaml").read_text(encoding="utf-8")
