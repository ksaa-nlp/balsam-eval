"""Judge failures must stop scoring and fail the evaluation job."""

from unittest.mock import Mock, patch

import pytest

from src.db_operations import JobStatus
from src.evaluation import EvaluationJob
from src.llm_judger.base_llm_judge import ModelConfig, call_model_adapter_with_retry
from src.llm_judger.generative_llm_judge import GenerativeLLMJudge
from src.llm_judger.mcq_llm_judge import MCQLLMJudge


@pytest.mark.parametrize("is_mcq", [False, True])
@pytest.mark.parametrize(
    "response",
    [
        RuntimeError("judge API unavailable"),
        "invalid JSON",
        "",
        '{"score": 1}',
        '{"score": null, "explanation": "failed"}',
        '{"score": "invalid", "explanation": "failed"}',
        '{"score": 4, "explanation": "failed"}',
        '{"score": NaN, "explanation": "failed"}',
        '{"score": 1, "explanation": null}',
    ],
)
def test_judge_failure_stops_job_and_marks_failed(is_mcq, response):
    """A judge failure prevents subsequent scoring and result export."""
    adapter = Mock()
    if isinstance(response, Exception):
        adapter.generate.side_effect = response
    else:
        adapter.generate.return_value = response

    job = EvaluationJob(
        tasks=["task"],
        model_args={},
        category_name="test",
        api_host="https://example.com",
        job_id="job",
        server_token="token",
        llm_judge_model="judge",
        llm_judge_provider="openai",
        llm_judge_api_key="key",
    )
    results = {
        "results": {"task": {}},
        "samples": {
            "task": [
                {"doc": {"input": "question", "output": "answer"},
                 "filtered_resps": ["answer"]}
                for _ in range(2)
            ]
        },
    }
    job._run_lm_eval = Mock(return_value=results)
    job.task_ops.add_task_to_results = Mock(side_effect=lambda data: data)
    job.task_ops.separate_mcq_and_generation_tasks = Mock(
        return_value=(["task"], []) if is_mcq else ([], ["task"])
    )
    job.result_processor.export_results = Mock()

    with (
        patch("src.llm_judger.base_llm_judge.create_model_adapter", return_value=adapter),
        patch("src.llm_judger.base_llm_judge.time.sleep") as sleep,
        patch("src.evaluation.update_status") as update_status,
    ):
        with pytest.raises(RuntimeError) as error:
            job()

    assert str(error.value)
    assert adapter.generate.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [1, 2]
    job.result_processor.export_results.assert_not_called()
    assert [call.kwargs["status"] for call in update_status.call_args_list] == [
        JobStatus.RUNNING,
        JobStatus.FAILED,
    ]
    assert update_status.call_args.kwargs["error_message"]


@pytest.mark.parametrize("judge_class", [GenerativeLLMJudge, MCQLLMJudge])
@pytest.mark.parametrize("raw_score", [0, 1])
def test_valid_scores_are_still_evaluated(judge_class, raw_score):
    """Valid zero and positive scores remain successful evaluations."""
    adapter = Mock()
    adapter.generate.return_value = (
        f'{{"score": {raw_score}, "explanation": "valid evaluation"}}',
        0,
    )
    with patch("src.llm_judger.base_llm_judge.create_model_adapter", return_value=adapter):
        judge = judge_class([ModelConfig(name="judge")])
    result = judge.evaluate_answer("question", "reference", "answer")

    assert result["overall_raw_score"] == raw_score
    assert result["overall_score"] == judge.normalize_score(raw_score)
    adapter.generate.assert_called_once()


@pytest.mark.parametrize(
    "first_response",
    [RuntimeError("temporary API error"), "invalid JSON",
     '{"score": null, "explanation": "failed"}'],
)
def test_judge_retry_recovers(first_response):
    """A successful retry returns a score and stops further attempts."""
    adapter = Mock()
    adapter.generate.side_effect = [
        first_response, '{"score": 1, "explanation": "recovered"}'
    ]
    with patch("src.llm_judger.base_llm_judge.time.sleep") as sleep:
        result = call_model_adapter_with_retry(adapter, "prompt", max_score=3)

    assert result == {"score": 1, "explanation": "recovered"}
    assert adapter.generate.call_count == 2
    sleep.assert_called_once_with(1)


def test_retry_exhaustion_preserves_final_error():
    """The original final exception is raised after all attempts fail."""
    adapter = Mock()
    final_error = RuntimeError("last API failure")
    adapter.generate.side_effect = [ValueError("first"), ValueError("second"), final_error]
    with patch("src.llm_judger.base_llm_judge.time.sleep"):
        with pytest.raises(RuntimeError) as error:
            call_model_adapter_with_retry(adapter, "prompt")
    assert error.value is final_error
    assert adapter.generate.call_count == 3
