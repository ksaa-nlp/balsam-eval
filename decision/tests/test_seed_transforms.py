"""The seeded System One and OpenAI Decisions transforms."""

import math

import pytest

from decisioneval.remote.contract import INVALID_CHOICE, MALFORMED, MISSING, REFUSAL, check
from decisioneval.transforms.engine import compile_expression, evaluate
from decisioneval.transforms.seeds import seed_transforms

IDS = ("A", "B", "C")
GOOD = {"A": 0.7, "B": 0.2, "C": 0.1}


def run(expression: str, data):
    return evaluate(compile_expression(expression), data)


def respond(seed, reply):
    return check(run(seed_transforms(seed)[1], {"reply": reply, "option_ids": list(IDS)}), IDS)


def oai(**answer):
    base = {"type": "choice", "name": "answer", "choice": "A", "confidence": 0.7,
            "probabilities": [{"value": k, "probability": v} for k, v in GOOD.items()]}
    return {"model": "gpt-6-luna", "answers": [{**base, **answer}]}


def s1(**answer):
    return {"answers": {"answer": {"type": "choice", "choice": "A", "probabilities": dict(GOOD), **answer}}}


@pytest.mark.parametrize("seed,reply", [("openai-decisions", oai()), ("systemone", s1())])
def test_a_clean_reply_maps_to_its_choice_and_vector(seed, reply):
    result = respond(seed, reply)
    assert (result.choice, result.probabilities, result.invalid_reason) == ("A", GOOD, None)


@pytest.mark.parametrize(
    "seed,reply",
    [
        ("openai-decisions", {"answers": [{"type": "refusal", "name": "answer"}]}),
        ("systemone", {"answers": {"answer": {"type": "refusal"}}}),
    ],
)
def test_a_refusal_is_invalid_with_no_vector(seed, reply):
    result = respond(seed, reply)
    assert result.invalid_reason == REFUSAL and result.probabilities is None


@pytest.mark.parametrize("choice", ["Z", None, 1, True])
@pytest.mark.parametrize("seed,build", [("openai-decisions", oai), ("systemone", s1)])
def test_a_choice_outside_the_ids_is_invalid(seed, build, choice):
    assert respond(seed, build(choice=choice)).invalid_reason == INVALID_CHOICE


@pytest.mark.parametrize(
    "rows",
    [
        [{"value": "A", "probability": 0.8}, {"value": "B", "probability": 0.2}],
        [{"value": "A", "probability": 0.5}, {"value": "A", "probability": 0.3}, {"value": "B", "probability": 0.2}],
        [{"value": 1, "probability": 0.7}, {"value": "B", "probability": 0.2}, {"value": "C", "probability": 0.1}],
        [{"value": "A", "probability": "0.7"}, {"value": "B", "probability": 0.2}, {"value": "C", "probability": 0.1}],
        [{"value": "A", "probability": math.nan}, {"value": "B", "probability": 0.5}, {"value": "C", "probability": 0.5}],
        [{"value": "A", "probability": 0.72}, {"value": "B", "probability": 0.2}, {"value": "C", "probability": 0.1}],
        [],
        dict(GOOD),
    ],
)
def test_a_malformed_openai_vector_is_dropped_and_the_choice_still_scores(rows):
    result = respond("openai-decisions", oai(probabilities=rows))
    assert result.choice == "A" and result.distribution_problem == MALFORMED


@pytest.mark.parametrize("seed,build", [("openai-decisions", oai), ("systemone", s1)])
def test_no_vector_is_missing(seed, build):
    reply = build()
    answer = reply["answers"][0] if seed == "openai-decisions" else reply["answers"]["answer"]
    del answer["probabilities"]
    result = respond(seed, reply)
    assert result.choice == "A" and result.distribution_problem == MISSING


def test_openai_reads_the_named_answer_before_the_first():
    reply = {"answers": [{"type": "choice", "name": "other", "choice": "C"}, oai(choice="B")["answers"][0]]}
    assert respond("openai-decisions", reply).choice == "B"


def test_openai_ignores_confidence_and_the_resolved_model():
    reply = oai()
    reply["model"] = "gpt-6-luna-2026-10-01"
    del reply["answers"][0]["confidence"]
    assert respond("openai-decisions", reply).choice == "A"


def test_system_one_reads_a_single_answer_under_any_name():
    reply = {"answers": {"q1": {"choice": "B", "probabilities": {"A": 0.1, "B": 0.8, "C": 0.1}}}}
    assert respond("systemone", reply).choice == "B"


@pytest.mark.parametrize("seed", ["openai-decisions", "systemone"])
def test_a_reply_of_another_shape_reads_as_no_choice(seed):
    result = respond(seed, {"result": {"answers": {"answer": {"choice": "A"}}}, "success": True})
    assert result.invalid_reason == INVALID_CHOICE
