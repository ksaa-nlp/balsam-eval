import json
from pathlib import Path
from unittest.mock import Mock, call

import pytest

import run
from src.core.config import EvalConfig
from src.db_operations import JobOutcome


def test_resolve_pool_files_returns_remote_sources_unchanged():
    config = EvalConfig(pool_files=["one.json", "nested/two.json"])

    assert run.resolve_pool_files(config) == ["one.json", "nested/two.json"]


def test_resolve_pool_files_discovers_only_source_json(monkeypatch, tmp_path):
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    for name in ("b.json", "a.json", "a_test.json", "b_dev.json", "notes.txt"):
        (tasks / name).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(run, "TASKS_DIR", str(tasks))

    assert run.resolve_pool_files(EvalConfig()) == [
        str(tasks / "a.json"),
        str(tasks / "b.json"),
    ]


def test_resolve_pool_files_handles_missing_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "TASKS_DIR", str(tmp_path / "missing"))

    assert run.resolve_pool_files(EvalConfig()) == []


def test_slugify_source_is_stable_and_avoids_basename_collisions():
    first = run._slugify_source("category-a/data.json")
    second = run._slugify_source("category-b/data.json")

    assert first == run._slugify_source("category-a/data.json")
    assert first.startswith("data-")
    assert first != second


def test_materialise_local_file_normalises_legacy_payload(monkeypatch, tmp_path):
    temp = tmp_path / "temp"
    source = tmp_path / "pool.json"
    source.write_text(
        json.dumps({"task": 4, "category": 9, "json": {"samples": [1]}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(run, "TEMP_DIR", str(temp))

    stem = run._materialise_pool_file(str(source), False, None)

    materialised = json.loads((temp / f"{stem}.json").read_text(encoding="utf-8"))
    assert materialised == {"samples": [1], "task": 4, "category": 9}


def test_materialise_remote_downloads_and_requires_bucket(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "TEMP_DIR", str(tmp_path))
    download = Mock(
        side_effect=lambda **kwargs: Path(kwargs["dest_path"]).write_text(
            '{"samples": []}', encoding="utf-8"
        )
    )
    monkeypatch.setattr(run, "download_pool_file_from_gcs", download)

    stem = run._materialise_pool_file("pools/data.json", True, "bucket")

    download.assert_called_once_with(
        bucket="bucket",
        object_path="pools/data.json",
        dest_path=str(tmp_path / f"{stem}.json"),
    )
    with pytest.raises(ValueError, match="GCLOUD_BUCKET"):
        run._materialise_pool_file("pools/data.json", True, None)


def test_evaluate_one_file_exports_media_and_runs_job(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "TEMP_DIR", str(tmp_path))
    monkeypatch.setattr(run, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(run, "_materialise_pool_file", lambda *_args: "pool-stem")
    for split in ("test", "dev"):
        (tmp_path / f"exported_{split}.json").write_text("[]", encoding="utf-8")

    dataset = Mock()
    dataset.file_name = "exported"
    dataset.name = "generated-task"
    dataset.category_id = "cat"
    dataset.task_id = "task"
    dataset_type = Mock(return_value=dataset)
    monkeypatch.setattr(run, "LMHDataset", dataset_type)
    images = Mock()
    audio = Mock()
    monkeypatch.setattr(run, "copy_images_to_temp", images)
    monkeypatch.setattr(run, "copy_audio_to_temp", audio)
    job = Mock(return_value="/results/result.json")
    job_type = Mock(return_value=job)
    monkeypatch.setattr(run, "_create_evaluation_job", job_type)
    config = EvalConfig(bucket="media-bucket", category_id="fallback")

    result = run._evaluate_one_file(
        source="remote/pool.json",
        is_remote=True,
        config=config,
        processed_adapter="adapter",
        model_args={"model": "m"},
    )

    assert result == ("/results/result.json", "pool-stem.json")
    dataset_type.assert_called_once_with("pool-stem", directory=str(tmp_path))
    dataset.export.assert_called_once_with()
    expected_splits = [str(tmp_path / f"exported_{name}.json") for name in ("test", "dev")]
    assert images.call_args_list == [
        call(path, str(tmp_path), bucket="media-bucket") for path in expected_splits
    ]
    assert audio.call_args_list == [
        call(path, str(tmp_path), bucket="media-bucket") for path in expected_splits
    ]
    job_type.assert_called_once_with(
        task_name="generated-task",
        category="cat",
        task_id="task",
        source_pool_path="remote/pool.json",
        adapter="adapter",
        model_args={"model": "m"},
        model=None,
        batch_size=8,
        bootstrap_iters=100000,
        result_filename="pool-stem.json",
        results_dir=str(tmp_path / "results"),
    )
    job.assert_called_once_with()


def test_try_finalize_skips_local_and_localhost(monkeypatch):
    finalize = Mock()
    monkeypatch.setattr(run, "finalize_job", finalize)

    assert run._try_finalize(EvalConfig(), JobOutcome.SUCCEEDED)
    assert run._try_finalize(EvalConfig(api_host="http://localhost:8000"), JobOutcome.FAILED)
    finalize.assert_not_called()


def test_try_finalize_truncates_error_and_swallows_backend_failure(monkeypatch):
    finalize = Mock(side_effect=RuntimeError("offline"))
    monkeypatch.setattr(run, "finalize_job", finalize)
    config = EvalConfig(
        api_host="https://api.example/", finalize_token="token", job_id="7"
    )

    assert not run._try_finalize(config, JobOutcome.FAILED, "x" * 5000)
    assert len(finalize.call_args.kwargs["error"]) == 4000
    assert finalize.call_args.kwargs["outcome"] is JobOutcome.FAILED


def _remote_config():
    return EvalConfig(
        api_host="https://api.example",
        finalize_token="token",
        job_id="7",
        adapter="openai",
        model_name="model",
        bucket="bucket",
        results_path="results/path/",
        pool_files=["one.json", "two.json"],
    )


def test_run_remote_evaluates_uploads_all_and_finalizes(monkeypatch):
    config = _remote_config()
    monkeypatch.setattr(run.EvalConfig, "from_env", Mock(return_value=config))
    for name in (
        "_setup_logging",
        "setup_directories",
        "copy_multimodal_utils_to_temp",
        "copy_metrics_combined_to_temp",
        "set_api_key_for_adapter",
        "_log_job_start",
        "_log_job_end",
    ):
        monkeypatch.setattr(run, name, Mock())
    monkeypatch.setattr(run, "process_adapter_and_url", Mock(return_value=("processed", "url")))
    model = Mock()
    create_model = Mock(return_value=model)
    monkeypatch.setattr(run, "_create_evaluation_model", create_model)
    evaluate = Mock(side_effect=[("/tmp/one", "one-result.json"), ("/tmp/two", "two-result.json")])
    monkeypatch.setattr(run, "_evaluate_one_file", evaluate)
    upload = Mock()
    finalize = Mock(return_value=True)
    monkeypatch.setattr(run, "upload_result_file_to_gcs", upload)
    monkeypatch.setattr(run, "_try_finalize", finalize)

    assert run._run() == 0
    create_model.assert_called_once_with("processed", config.get_model_args("url"), 8)
    assert all(item.kwargs["model"] is model for item in evaluate.call_args_list)
    assert upload.call_args_list == [
        call(bucket="bucket", local_path="/tmp/one", object_path="results/path/one-result.json"),
        call(bucket="bucket", local_path="/tmp/two", object_path="results/path/two-result.json"),
    ]
    finalize.assert_called_once_with(config, JobOutcome.SUCCEEDED)


def test_run_reports_no_files_and_evaluation_failure(monkeypatch):
    config = EvalConfig(model_name="model", adapter="adapter")
    monkeypatch.setattr(run.EvalConfig, "from_env", Mock(return_value=config))
    for name in (
        "_setup_logging",
        "setup_directories",
        "copy_multimodal_utils_to_temp",
        "copy_metrics_combined_to_temp",
        "set_api_key_for_adapter",
        "_log_job_start",
        "_log_job_end",
    ):
        monkeypatch.setattr(run, name, Mock())
    monkeypatch.setattr(run, "process_adapter_and_url", lambda *_args: ("adapter", None))
    monkeypatch.setattr(run, "_create_evaluation_model", Mock(return_value=Mock()))
    finalize = Mock(return_value=True)
    monkeypatch.setattr(run, "_try_finalize", finalize)
    monkeypatch.setattr(run, "resolve_pool_files", Mock(return_value=[]))

    assert run._run() == 1
    finalize.assert_called_once_with(config, JobOutcome.FAILED, "No pool files to evaluate.")

    monkeypatch.setattr(run, "resolve_pool_files", Mock(return_value=["bad.json"]))
    monkeypatch.setattr(run, "_evaluate_one_file", Mock(side_effect=ValueError("broken")))
    finalize.reset_mock()
    assert run._run() == 1
    finalize.assert_called_once_with(
        config, JobOutcome.FAILED, "bad.json: broken"
    )


def test_main_exits_with_run_code_and_reports_crash(monkeypatch):
    monkeypatch.setattr(run, "load_dotenv", Mock())
    monkeypatch.setattr(run, "_run", Mock(return_value=3))
    with pytest.raises(SystemExit) as exc:
        run.main()
    assert exc.value.code == 3

    config = _remote_config()
    monkeypatch.setattr(run, "_run", Mock(side_effect=RuntimeError("boom")))
    monkeypatch.setattr(run.EvalConfig, "from_env", Mock(return_value=config))
    finalize = Mock(return_value=True)
    monkeypatch.setattr(run, "_try_finalize", finalize)
    with pytest.raises(SystemExit) as exc:
        run.main()
    assert exc.value.code == 1
    finalize.assert_called_once_with(
        config, JobOutcome.FAILED, "runner crashed: boom"
    )
