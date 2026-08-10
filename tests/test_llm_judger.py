import json
from statistics import StatisticsError
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.llm_judger import base_llm_judge as base
from src.llm_judger.generative_llm_judge import GenerativeLLMJudge
from src.llm_judger.mcq_llm_judge import MCQLLMJudge


def build_judge(cls=GenerativeLLMJudge, configs=None, **kwargs):
    judge = object.__new__(cls)
    judge.model_configs = configs or [base.ModelConfig("judge")]
    judge.model_adapters = [MagicMock() for _ in judge.model_configs]
    judge.aggregation_method = kwargs.get("aggregation_method", "mean")
    judge.threshold = kwargs.get("threshold", 0.7)
    judge.custom_prompt = kwargs.get("custom_prompt")
    return judge


def test_create_model_adapter_builds_provider_specific_parameters(monkeypatch):
    adapter = MagicMock()
    factory = MagicMock(return_value=adapter)
    get_model = MagicMock(return_value=factory)
    monkeypatch.setattr(base, "get_model", get_model)
    config = base.ModelConfig(
        name="judge",
        provider="openai",
        api_key="key",
        endpoint_url="https://host/v1/chat/completions?api-version=1",
        other={"temperature": 0},
    )
    assert base.create_model_adapter(config) is adapter
    assert adapter.api_key == "key"
    get_model.assert_called_once_with("openai")
    factory.assert_called_once_with(
        model="judge",
        base_url="https://host/v1/chat/completions?api-version=1",
        temperature=0,
    )


def test_create_model_adapter_rejects_unknown_provider():
    config = base.ModelConfig("judge")
    config.provider = "unknown"
    with pytest.raises(ValueError, match="Unsupported provider"):
        base.create_model_adapter(config)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ('```json\n{"score": 1, "explanation": "ok"}\n```', {"score": 1, "explanation": "ok"}),
        ((json.dumps({"score": 0.5, "explanation": "partial"}), {}), {"score": 0.5, "explanation": "partial"}),
        (SimpleNamespace(text='{"score": 0, "explanation": "no"}'), {"score": 0, "explanation": "no"}),
        (SimpleNamespace(content='{"score": 1, "explanation": "yes"}'), {"score": 1, "explanation": "yes"}),
    ],
)
def test_call_model_adapter_parses_common_response_shapes(response, expected):
    adapter = MagicMock()
    adapter.generate.return_value = response
    assert base.call_model_adapter_with_retry(adapter, "prompt", max_retries=1) == expected


def test_call_model_adapter_uses_lm_eval_generate_until():
    generate_until = MagicMock(
        return_value=['{"score": 1, "explanation": "ok"}']
    )
    adapter = SimpleNamespace(generate_until=generate_until)

    result = base.call_model_adapter_with_retry(adapter, "prompt", max_retries=1)

    assert result == {"score": 1, "explanation": "ok"}
    request = generate_until.call_args.args[0][0]
    assert request.args == ("prompt", {"until": [], "do_sample": False})
    generate_until.assert_called_once_with([request])


@pytest.mark.parametrize("method", ["_call", "invoke", "__call__"])
def test_call_model_adapter_routes_available_call_styles(method):
    response = '{"score": 1, "explanation": "ok"}'
    if method == "__call__":
        adapter = MagicMock(return_value=response, spec=lambda _: None)
    else:
        adapter = SimpleNamespace(**{method: MagicMock(return_value=response)})
    assert base.call_model_adapter_with_retry(adapter, "prompt", max_retries=1)["score"] == 1


@pytest.mark.parametrize(
    "response",
    [
        '{"score": true, "explanation": "bad"}',
        '{"score": 2, "explanation": "too high"}',
        '{"score": 1}',
        "not-json",
    ],
)
def test_call_model_adapter_rejects_invalid_judgments(response, monkeypatch):
    monkeypatch.setattr(base.time, "sleep", MagicMock())
    adapter = MagicMock()
    adapter.generate.return_value = response
    with pytest.raises(RuntimeError, match="unable to get a valid response"):
        base.call_model_adapter_with_retry(adapter, "prompt", max_retries=2)
    assert adapter.generate.call_count == 2


@pytest.mark.parametrize(
    ("cls", "raw", "expected"),
    [
        (MCQLLMJudge, 0, 0.0),
        (MCQLLMJudge, "1", 1.0),
        (MCQLLMJudge, 0.5, 0.0),
        (MCQLLMJudge, "bad", 0.0),
        (GenerativeLLMJudge, 0, 0.0),
        (GenerativeLLMJudge, 2, 0.66667),
        (GenerativeLLMJudge, 3, 1.0),
        (GenerativeLLMJudge, 4, 0.0),
    ],
)
def test_score_normalization(cls, raw, expected):
    assert object.__new__(cls).normalize_score(raw) == expected


@pytest.mark.parametrize(
    ("configs", "method", "threshold", "message"),
    [([], "mean", 0.5, "At least one"),
     ([base.ModelConfig("m")], "mode", 0.5, "aggregation_method"),
     ([base.ModelConfig("m")], "mean", 1.1, "threshold")],
)
def test_judge_constructor_validation(configs, method, threshold, message, monkeypatch):
    monkeypatch.setattr(base, "create_model_adapter", MagicMock())
    with pytest.raises(ValueError, match=message):
        MCQLLMJudge(configs, aggregation_method=method, threshold=threshold)


def test_judge_constructor_creates_one_adapter_per_config(monkeypatch):
    create = MagicMock(side_effect=["a", "b"])
    monkeypatch.setattr(base, "create_model_adapter", create)
    configs = [base.ModelConfig("one"), base.ModelConfig("two", provider="gemini")]
    judge = GenerativeLLMJudge(configs, aggregation_method="median", threshold=0.4)
    assert judge.model_adapters == ["a", "b"]
    assert judge.aggregation_method == "median"
    assert create.call_args_list[1].args == (configs[1],)


def test_single_model_prompt_precedence_and_normalized_result(monkeypatch):
    config = base.ModelConfig("judge", custom_prompt="config prompt")
    judge = build_judge(configs=[config], custom_prompt="instance prompt", threshold=0.6)
    call = MagicMock(return_value={"score": 2, "explanation": "close"})
    monkeypatch.setattr(base, "call_model_adapter_with_retry", call)

    result = judge._evaluate_single_model(
        "question", "reference", "given", "context", config, judge.model_adapters[0],
        custom_prompt="call prompt",
    )

    prompt = call.call_args.args[1]
    assert prompt.startswith("call prompt")
    assert "[PROMPT]\nquestion" in prompt
    assert "[CONTEXT]\ncontext" in prompt
    assert "[GROUND TRUTH]\nreference" in prompt
    assert result == {
        "model": "judge", "provider": "openai", "score": 0.66667,
        "raw_score": 2, "passed": True, "explanation": "close",
    }


@pytest.mark.parametrize(
    ("method", "expected_score", "expected_raw"),
    [("mean", 0.6, 1.5), ("median", 0.6, 1.5)],
)
def test_aggregation_ignores_none_scores_and_explains_models(method, expected_score, expected_raw):
    judge = build_judge(aggregation_method=method)
    result = judge._aggregate_model_results([
        {"model": "a", "score": 0.2, "raw_score": 1, "explanation": "low"},
        {"model": "b", "score": 1.0, "raw_score": 2, "explanation": "high"},
        {"model": "c", "score": None, "raw_score": None, "explanation": "missing"},
    ])
    assert result["overall_score"] == expected_score
    assert result["overall_raw_score"] == expected_raw
    assert "a: low" in result["aggregated_explanation"]


def test_evaluate_answer_aggregates_models_and_metadata(monkeypatch):
    configs = [base.ModelConfig("one"), base.ModelConfig("two")]
    judge = build_judge(configs=configs, threshold=0.5)
    monkeypatch.setattr(judge, "_evaluate_single_model", MagicMock(side_effect=[
        {"model": "one", "score": 0.4, "raw_score": 1, "explanation": "a"},
        {"model": "two", "score": 0.8, "raw_score": 3, "explanation": "b"},
    ]))
    result = judge.evaluate_answer("q", "r", "g", test_id="id", metadata={"group": "x"})
    assert result["overall_score"] == pytest.approx(0.6)
    assert result["overall_raw_score"] == 2
    assert result["overall_passed"] is True
    assert result["metadata"]["test_id"] == "id"
    assert result["metadata"]["group"] == "x"


def test_evaluate_batch_accepts_dict_and_dataclass(monkeypatch):
    judge = build_judge(threshold=0.5)
    evaluate = MagicMock(side_effect=[
        {"overall_score": 0.25, "overall_raw_score": 1},
        {"overall_score": 0.75, "overall_raw_score": 3},
    ])
    monkeypatch.setattr(judge, "evaluate_answer", evaluate)
    result = judge.evaluate_batch([
        {"question": "q1", "reference_answer": "r1", "given_answer": "g1"},
        base.TestCaseDict("q2", "r2", "g2", context="ctx", id="2"),
    ], show_progress=False)
    assert result["batch_statistics"] == {
        "total_test_cases": 2,
        "average_score": 0.5,
        "median_score": 0.5,
        "average_raw_score": 2,
        "median_raw_score": 2.0,
        "pass_rate": 0.5,
    }
    assert evaluate.call_args_list[1].kwargs["context"] == "ctx"


def test_evaluate_batch_rejects_unknown_case_type():
    judge = build_judge()
    with pytest.raises(TypeError, match="Unsupported test case"):
        judge.evaluate_batch([object()], show_progress=False)


def test_empty_aggregation_is_deterministic_and_empty_statistics_are_rejected():
    judge = build_judge()
    assert judge._aggregate_model_results([]) == {
        "overall_score": 0, "overall_raw_score": 0, "aggregated_explanation": "No valid scores"
    }
    with pytest.raises(StatisticsError):
        judge._calculate_batch_statistics([])


def test_save_results_writes_utf8_json(tmp_path):
    path = tmp_path / "result.json"
    build_judge().save_results({"text": "عربي"}, str(path))
    assert json.loads(path.read_text(encoding="utf-8")) == {"text": "عربي"}
