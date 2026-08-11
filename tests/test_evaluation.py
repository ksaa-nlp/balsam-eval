from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from src import evaluation


def make_job(**overrides):
    values = {
        "task_name": "generated-task",
        "category": "category",
        "task_id": "task-id",
        "source_pool_path": "pools/source.json",
        "adapter": "openai-chat-completions",
        "model_args": {"model": "model", "api_key": "secret"},
        "result_filename": "result.json",
        "results_dir": ".results",
    }
    values.update(overrides)
    return evaluation.SingleFileEvaluationJob(**values)


def test_compatibility_patches_apply_defaults_and_restore(monkeypatch, tmp_path):
    original_relative_to = Path.relative_to
    post = Mock(return_value="response")
    monkeypatch.setattr(requests, "post", post)

    with evaluation._lm_eval_compatibility_patches():
        outside = tmp_path / "outside"
        assert outside.relative_to(tmp_path / "other") == outside
        assert requests.post("https://example.test") == "response"
        requests.post("https://example.test", timeout=2)

    assert Path.relative_to is original_relative_to
    assert requests.post is post
    assert post.call_args_list[0].kwargs == {"timeout": 5000}
    assert post.call_args_list[1].kwargs == {"timeout": 2}


def test_compatibility_patches_restore_after_exception(monkeypatch):
    original_relative_to = Path.relative_to
    post = Mock()
    monkeypatch.setattr(requests, "post", post)

    with pytest.raises(RuntimeError, match="boom"):
        with evaluation._lm_eval_compatibility_patches():
            raise RuntimeError("boom")

    assert Path.relative_to is original_relative_to
    assert requests.post is post


@pytest.mark.parametrize(
    ("adapter", "name"),
    [
        ("openai", "OPENAI_API_KEY"),
        ("local-chat-completions", "OPENAI_API_KEY"),
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("gemini", "GOOGLE_API_KEY"),
        ("cohere", "CO_API_KEY"),
        ("azure-stt", "AZURE_SPEECH_KEY"),
    ],
)
def test_set_api_key_env_maps_supported_adapters(monkeypatch, adapter, name):
    monkeypatch.delenv(name, raising=False)
    evaluation._set_api_key_env(adapter, "key")
    assert evaluation.os.environ[name] == "key"


def test_job_constructor_copies_args_adds_eos_and_reads_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "env-secret")
    supplied = {"model": "model"}

    job = make_job(model_args=supplied)

    assert supplied == {"model": "model"}
    assert job.model_args == {"model": "model", "eos_string": "<|endoftext|>"}
    assert evaluation.os.environ["OPENAI_API_KEY"] == "env-secret"


@pytest.mark.parametrize(
    ("adapter", "chat_template"),
    [("openai-chat-completions", True), ("openai-asr", False)],
)
def test_run_lm_eval_builds_expected_request(monkeypatch, adapter, chat_template):
    task_manager = Mock(return_value="manager")
    simple_evaluate = Mock(return_value={"results": {}})
    max_tokens = Mock(return_value={"max_gen_toks": 99})
    monkeypatch.setattr(evaluation.lm_eval.tasks, "TaskManager", task_manager)
    monkeypatch.setattr(evaluation.lm_eval.evaluator, "simple_evaluate", simple_evaluate)
    monkeypatch.setattr(evaluation, "get_max_tokens_config", max_tokens)
    job = make_job(adapter=adapter, model_args={"model": "model"})

    assert job._run_lm_eval() == {"results": {}}
    task_manager.assert_called_once_with(
        include_path=str(Path(".temp").resolve()), include_defaults=False
    )
    simple_evaluate.assert_called_once_with(
        model=adapter,
        model_args={"model": "model", "eos_string": "<|endoftext|>"},
        tasks=["generated-task"],
        apply_chat_template=chat_template,
        task_manager="manager",
        batch_size=8,
        bootstrap_iters=100000,
        log_samples=True,
        gen_kwargs={"max_gen_toks": 99},
    )


def test_run_lm_eval_reuses_preinitialized_model(monkeypatch):
    model = Mock(batch_size=4)
    simple_evaluate = Mock(return_value={"results": {}, "config": {}})
    monkeypatch.setattr(evaluation.lm_eval.evaluator, "simple_evaluate", simple_evaluate)
    monkeypatch.setattr(evaluation, "get_max_tokens_config", Mock(return_value={}))
    job = make_job(model=model, batch_size=4, bootstrap_iters=2)

    job._run_lm_eval()

    assert simple_evaluate.call_args.kwargs["model"] is model
    assert simple_evaluate.call_args.kwargs["model_args"] is None
    assert simple_evaluate.call_args.kwargs["batch_size"] == 4
    assert simple_evaluate.call_args.kwargs["bootstrap_iters"] == 2
    config = simple_evaluate.return_value["config"]
    assert config == {
        "model": "openai-chat-completions",
        "model_args": {
            "model": "model",
            "api_key": "secret",
            "eos_string": "<|endoftext|>",
        },
    }


def test_call_sanitizes_stamps_and_exports(monkeypatch):
    results = {
        "config": {"model_args": {"api_key": "secret", "model": "m"}},
        "results": {
            "one": {"accuracy": 1},
            "two": {"task": "existing", "category": "existing"},
            "invalid": [],
        },
    }
    job = make_job()
    monkeypatch.setattr(job, "_run_lm_eval", Mock(return_value=results))
    processor = Mock()
    processor.export.return_value = "/results/result.json"
    processor_type = Mock(return_value=processor)
    monkeypatch.setattr(evaluation, "ResultProcessor", processor_type)

    assert job() == "/results/result.json"
    assert results["config"]["model_args"] == {"model": "m"}
    assert results["results"]["one"]["task"] == "task-id"
    assert results["results"]["one"]["category"] == "category"
    assert results["results"]["two"]["task"] == "existing"
    assert results["category"] == "category"
    assert results["task"] == "task-id"
    assert results["pool_file"] == "pools/source.json"
    processor_type.assert_called_once_with(
        category="category",
        task_id="task-id",
        source_pool_path="pools/source.json",
        results_dir=".results",
    )
    processor.export.assert_called_once_with(results, filename="result.json")


@pytest.mark.parametrize("empty", [None, {}])
def test_call_rejects_empty_lm_eval_result(monkeypatch, empty):
    job = make_job()
    monkeypatch.setattr(job, "_run_lm_eval", Mock(return_value=empty))

    with pytest.raises(RuntimeError, match="returned no results"):
        job()


def test_sanitize_tolerates_unexpected_config_shapes():
    for results in ({}, {"config": []}, {"config": {"model_args": "text"}}):
        evaluation.SingleFileEvaluationJob._sanitize_results(results)
