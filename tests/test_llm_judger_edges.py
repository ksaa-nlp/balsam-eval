import json
from types import SimpleNamespace
from unittest.mock import MagicMock, call

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


def test_create_model_adapter_keeps_non_openai_endpoint(monkeypatch):
    factory = MagicMock(return_value="adapter")
    monkeypatch.setattr(base, "get_model", MagicMock(return_value=factory))
    config = base.ModelConfig(
        name="judge",
        provider="gemini",
        endpoint_url="https://judge.invalid/chat/completions/",
    )

    assert base.create_model_adapter(config) == "adapter"
    factory.assert_called_once_with(
        model_name="judge", base_url="https://judge.invalid/chat/completions/"
    )


def test_create_model_adapter_omits_unset_optional_parameters(monkeypatch):
    factory = MagicMock(return_value="adapter")
    monkeypatch.setattr(base, "get_model", MagicMock(return_value=factory))

    base.create_model_adapter(base.ModelConfig(name="judge"))

    factory.assert_called_once_with(model="judge")


@pytest.mark.parametrize("method_name", ["call", "predict", "complete"])
def test_call_model_adapter_uses_fallback_methods(method_name):
    method = MagicMock(return_value='{"score": 1, "explanation": "ok"}')
    adapter = SimpleNamespace(**{method_name: method})

    result = base.call_model_adapter_with_retry(adapter, "prompt", max_retries=1)

    assert result == {"score": 1, "explanation": "ok"}
    method.assert_called_once_with("prompt")


def test_call_model_adapter_stringifies_unknown_response_objects():
    class Response:
        def __str__(self):
            return '{"score": 0.5, "explanation": "stringified"}'

    adapter = SimpleNamespace(generate=MagicMock(return_value=Response()))

    assert base.call_model_adapter_with_retry(adapter, "prompt", max_retries=1) == {
        "score": 0.5,
        "explanation": "stringified",
    }


def test_call_model_adapter_retries_call_exception_then_succeeds(monkeypatch):
    sleep = MagicMock()
    monkeypatch.setattr(base.time, "sleep", sleep)
    adapter = SimpleNamespace(
        generate=MagicMock(
            side_effect=[
                ConnectionError("offline"),
                '{"score": 1, "explanation": "recovered"}',
            ]
        )
    )

    result = base.call_model_adapter_with_retry(adapter, "prompt", max_retries=2)

    assert result["explanation"] == "recovered"
    sleep.assert_called_once_with(1)


def test_call_model_adapter_rejects_adapter_without_callable_method(monkeypatch):
    sleep = MagicMock()
    monkeypatch.setattr(base.time, "sleep", sleep)

    with pytest.raises(RuntimeError, match="unable to get a valid response"):
        base.call_model_adapter_with_retry(object(), "prompt", max_retries=1)

    sleep.assert_called_once_with(1)


def test_call_model_adapter_retries_empty_text_response(monkeypatch):
    sleep = MagicMock()
    monkeypatch.setattr(base.time, "sleep", sleep)
    adapter = SimpleNamespace(generate=MagicMock(return_value=SimpleNamespace(text=None)))

    with pytest.raises(RuntimeError, match="unable to get a valid response"):
        base.call_model_adapter_with_retry(adapter, "prompt", max_retries=2)

    assert sleep.call_args_list == [call(1), call(2)]


@pytest.mark.parametrize(
    "score",
    [-1, 4, True, float("nan"), float("inf")],
)
def test_call_model_adapter_enforces_numeric_finite_score_range(score, monkeypatch):
    monkeypatch.setattr(base.time, "sleep", MagicMock())
    response = json.dumps({"score": score, "explanation": "invalid"})
    adapter = SimpleNamespace(generate=MagicMock(return_value=response))

    with pytest.raises(RuntimeError, match="unable to get a valid response"):
        base.call_model_adapter_with_retry(
            adapter, "prompt", max_retries=1, max_score=3
        )


def test_call_model_adapter_accepts_subclass_maximum_score():
    adapter = SimpleNamespace(
        generate=MagicMock(return_value='{"score": 3, "explanation": "perfect"}')
    )

    assert base.call_model_adapter_with_retry(
        adapter, "prompt", max_retries=1, max_score=3
    )["score"] == 3


@pytest.mark.parametrize(
    ("config_prompt", "instance_prompt", "expected_start"),
    [
        ("config prompt", "instance prompt", "config prompt"),
        (None, "instance prompt", "instance prompt"),
        (None, None, "You are an impartial and expert judge"),
    ],
)
def test_single_model_prompt_fallback_precedence(
    monkeypatch, config_prompt, instance_prompt, expected_start
):
    config = base.ModelConfig("judge", custom_prompt=config_prompt)
    judge = build_judge(configs=[config], custom_prompt=instance_prompt)
    call = MagicMock(return_value={"score": 0, "explanation": "no"})
    monkeypatch.setattr(base, "call_model_adapter_with_retry", call)

    judge._evaluate_single_model(
        "question", "reference", "answer", None, config, judge.model_adapters[0]
    )

    prompt = call.call_args.args[1]
    assert prompt.startswith(expected_start)
    assert "[CONTEXT]" not in prompt
    call.assert_called_once_with(judge.model_adapters[0], prompt, max_score=1.0)


def test_single_model_handles_none_raw_score_from_adapter(monkeypatch):
    config = base.ModelConfig("judge")
    judge = build_judge(configs=[config])
    monkeypatch.setattr(
        base,
        "call_model_adapter_with_retry",
        MagicMock(return_value={"score": None, "explanation": "missing"}),
    )

    result = judge._evaluate_single_model(
        "q", "r", "a", None, config, judge.model_adapters[0]
    )

    assert result["score"] == 0
    assert result["passed"] is False


def test_single_model_reraises_adapter_failure(monkeypatch):
    config = base.ModelConfig("judge")
    judge = build_judge(configs=[config])
    monkeypatch.setattr(
        base,
        "call_model_adapter_with_retry",
        MagicMock(side_effect=RuntimeError("failed")),
    )

    with pytest.raises(RuntimeError, match="failed"):
        judge._evaluate_single_model(
            "q", "r", "a", None, config, judge.model_adapters[0]
        )


def test_aggregate_model_results_defaults_raw_score_when_absent():
    judge = build_judge()

    result = judge._aggregate_model_results(
        [{"model": "judge", "score": 0.75, "raw_score": None, "explanation": "ok"}]
    )

    assert result["overall_score"] == 0.75
    assert result["overall_raw_score"] == 0


def test_evaluate_batch_accepts_test_case_object_and_uses_progress(monkeypatch):
    class FakeLLMTestCase:
        def __init__(self):
            self.input = "question"
            self.expected_output = "reference"
            self.actual_output = "answer"
            self.context = ["first", "second"]
            self.id = "case-id"
            self.metadata = {"group": "edge"}

    judge = build_judge(threshold=0.5)
    evaluate = MagicMock(
        return_value={"overall_score": 1.0, "overall_raw_score": 3.0}
    )
    progress = MagicMock(side_effect=lambda values, **_: values)
    monkeypatch.setattr(base, "tqdm", progress)
    monkeypatch.setattr(judge, "evaluate_answer", evaluate)

    result = judge.evaluate_batch([FakeLLMTestCase()], show_progress=True)

    progress.assert_called_once()
    evaluate.assert_called_once_with(
        question="question",
        reference_answer="reference",
        given_answer="answer",
        context="first\nsecond",
        test_id="case-id",
        metadata={"group": "edge"},
    )
    assert result["batch_statistics"]["pass_rate"] == 1.0


@pytest.mark.parametrize(
    ("judge_class", "maximum", "prompt_fragment"),
    [
        (GenerativeLLMJudge, 1.0, "Decimal scores are allowed"),
        (MCQLLMJudge, 1.0, "multiple-choice question"),
    ],
)
def test_subclass_prompts_and_maximum_scores(judge_class, maximum, prompt_fragment):
    judge = object.__new__(judge_class)

    assert judge.get_max_score() == maximum
    assert prompt_fragment in judge.get_evaluation_prompt()


@pytest.mark.parametrize(
    ("judge_class", "raw_score"),
    [
        (GenerativeLLMJudge, None),
        (GenerativeLLMJudge, "invalid"),
        (GenerativeLLMJudge, -1),
        (GenerativeLLMJudge, float("nan")),
        (MCQLLMJudge, None),
        (MCQLLMJudge, "invalid"),
        (MCQLLMJudge, -1),
        (MCQLLMJudge, float("nan")),
    ],
)
def test_subclass_normalizers_reject_invalid_values(judge_class, raw_score):
    assert object.__new__(judge_class).normalize_score(raw_score) == 0.0
