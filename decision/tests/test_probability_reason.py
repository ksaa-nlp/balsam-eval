"""A vector the contract refused nulls its group's calibration as malformed, not missing."""

from decisioneval.metrics.probability import MALFORMED, MISSING, probability_scores
from decisioneval.models import ItemResult


def item(probabilities=None, distribution_problem=None):
    return ItemResult(
        item_id=1, dataset="d", category="c", task="t", options=("x", "y"), option_ids=("A", "B"),
        gold_answer="x", gold_option_id="A", predicted_option_id="A", predicted_answer="x",
        probabilities=probabilities, correct=True, invalid=False, latency_ms=1.0, usage=None,
        cost=None, raw_response=None, distribution_problem=distribution_problem,
    )


def test_a_refused_vector_reads_as_malformed_and_outranks_a_missing_one():
    clean = item({"A": 0.9, "B": 0.1})
    assert probability_scores([clean, item(distribution_problem="malformed")])["probability_metrics"]["reason"] == MALFORMED
    assert probability_scores([clean, item(), item(distribution_problem="malformed")])["probability_metrics"]["reason"] == MALFORMED
    assert probability_scores([clean, item(distribution_problem="missing")])["probability_metrics"]["reason"] == MISSING


def test_the_problem_is_stored_with_the_item():
    assert item(distribution_problem="malformed").to_dict()["distribution_problem"] == "malformed"
