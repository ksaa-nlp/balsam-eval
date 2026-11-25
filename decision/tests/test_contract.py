"""The decision contract every response transform output is held to (plan §6.4, EVALUATION.md §2)."""

import math

import pytest

from decisioneval.remote.contract import ADAPTER_ERROR, INVALID_CHOICE, MALFORMED, MISSING, REFUSAL, check

IDS = ("A", "B", "C")
GOOD = {"A": 0.7, "B": 0.2, "C": 0.1}


def answer(**fields):
    return {"status": "answer", "choice": "A", "probabilities": dict(GOOD), **fields}


def test_a_clean_answer_keeps_its_choice_and_exact_vector():
    result = check(answer(), IDS)
    assert result.choice == "A" and result.invalid_reason is None
    assert result.probabilities == GOOD and result.distribution_problem is None
    assert not result.breaker_error


@pytest.mark.parametrize(
    "output",
    [
        None,
        "A",
        [],
        {},
        {"status": "maybe", "choice": "A"},
        {"status": "answer", "choice": 1},
        {"status": "answer", "choice": "A", "probabilities": [0.5, 0.5]},
        {"status": "answer", "choice": "A", "confidence": 0.9},  # an unknown key
    ],
)
def test_output_off_the_schema_is_an_adapter_error(output):
    result = check(output, IDS)
    assert result.invalid_reason == ADAPTER_ERROR and result.choice is None
    assert result.probabilities is None and result.breaker_error


def test_a_refusal_is_invalid_without_a_vector_and_never_trips_the_breaker():
    result = check({"status": "refusal", "choice": None, "probabilities": None}, IDS)
    assert result.invalid_reason == REFUSAL and result.choice is None
    assert result.probabilities is None and result.distribution_problem == MISSING
    assert not result.breaker_error


def test_a_refusal_drops_any_vector_it_carries():
    result = check({"status": "refusal", "probabilities": dict(GOOD)}, IDS)
    assert result.invalid_reason == REFUSAL and result.probabilities is None


@pytest.mark.parametrize("choice", ["Z", "", "a", None])
def test_a_choice_outside_the_option_ids_is_invalid(choice):
    result = check(answer(choice=choice), IDS)
    assert result.invalid_reason == INVALID_CHOICE and result.choice is None
    assert not result.breaker_error


def test_a_missing_choice_key_reads_as_no_choice():
    output = answer()
    del output["choice"]
    assert check(output, IDS).invalid_reason == INVALID_CHOICE


def test_an_invalid_choice_keeps_a_clean_vector_since_the_two_are_independent():
    assert check(answer(choice="Z"), IDS).probabilities == GOOD


@pytest.mark.parametrize("probabilities", [None, "absent"])
def test_no_vector_is_missing_and_the_choice_still_scores(probabilities):
    output = answer(probabilities=probabilities)
    if probabilities == "absent":
        del output["probabilities"]
    result = check(output, IDS)
    assert result.choice == "A" and result.invalid_reason is None
    assert result.probabilities is None and result.distribution_problem == MISSING
    assert not result.breaker_error


@pytest.mark.parametrize(
    "vector",
    [
        {"A": 0.8, "B": 0.2},  # a missing key
        {"A": 0.6, "B": 0.2, "C": 0.1, "D": 0.1},  # an extra key
        {"": 0, "A": 0.7, "B": 0.2, "C": 0.1},  # the duplicate-key marker a seed emits
        {"A": "0.7", "B": 0.2, "C": 0.1},
        {"A": True, "B": False, "C": False},
        {"A": None, "B": 0.9, "C": 0.1},
        {"A": math.nan, "B": 0.5, "C": 0.5},
        {"A": math.inf, "B": 0.0, "C": 0.0},
        {"A": 1.2, "B": -0.1, "C": -0.1},
        {"A": 0.72, "B": 0.2, "C": 0.1},  # off by 0.02: never renormalised
    ],
)
def test_a_malformed_vector_is_dropped_never_repaired_and_trips_the_breaker(vector):
    result = check(answer(probabilities=vector), IDS)
    assert result.choice == "A" and result.invalid_reason is None
    assert result.probabilities is None and result.distribution_problem == MALFORMED
    assert result.breaker_error


def test_a_sum_within_tolerance_is_kept_exactly_as_sent():
    result = check(answer(probabilities={"A": 0.705, "B": 0.2, "C": 0.1}), IDS)
    assert result.probabilities == {"A": 0.705, "B": 0.2, "C": 0.1}


def test_integer_probabilities_are_numbers():
    result = check(answer(probabilities={"A": 1, "B": 0, "C": 0}), IDS)
    assert result.probabilities == {"A": 1, "B": 0, "C": 0}


def test_the_vector_is_returned_in_option_order():
    result = check(answer(probabilities={"C": 0.1, "A": 0.7, "B": 0.2}), IDS)
    assert list(result.probabilities) == ["A", "B", "C"]


def test_a_huge_integer_probability_is_malformed_not_a_crash():
    result = check(answer(probabilities={"A": 10 ** 400, "B": 0, "C": 0}), IDS)
    assert result.distribution_problem == MALFORMED


def test_a_reply_with_an_unparseable_integer_is_kept_as_text():
    from decisioneval.spec.transport import _parse

    assert "_text" in _parse(b'{"p": ' + b"9" * 5000 + b"}")
