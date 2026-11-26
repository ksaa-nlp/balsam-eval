"""Brier, ECE, AURC, coverage and the latency percentile against hand-computed values."""

from dataclasses import dataclass

import pytest

from decisioneval.metrics.probability import MALFORMED, NO_ITEMS, ece_bin_index, probability_scores
from decisioneval.metrics.summary import _percentile


@dataclass(frozen=True)
class Item:
    probabilities: dict[str, float] | None
    option_ids: tuple[str, ...]
    gold_option_id: str
    predicted_option_id: str | None
    confidence: float | None = None


def item(probabilities: dict[str, float], gold: str, predicted: str) -> Item:
    return Item(probabilities, tuple(probabilities), gold, predicted)


# Confidence 0.9 right, 0.6 wrong, 0.5 right.
ITEMS = [
    item({"A": 0.9, "B": 0.1}, "A", "A"),
    item({"A": 0.6, "B": 0.4}, "B", "A"),
    item({"A": 0.2, "B": 0.3, "C": 0.5}, "C", "C"),
]


def test_scores_match_hand_computed_values():
    scores = probability_scores(ITEMS)
    assert scores["brier_raw"] == pytest.approx((0.02 + 0.72 + 0.38) / 3)
    assert scores["brier_normalized"] == pytest.approx((0.01 + 0.36 + 0.38 / 3) / 3)
    assert scores["ece"] == pytest.approx((0.1 + 0.6 + 0.5) / 3)
    assert scores["aurc"] == pytest.approx((0 + 1 / 2 + 1 / 3) / 3)
    assert scores["coverage_at_10_risk"] == pytest.approx(1 / 3)


def test_coverage_takes_the_longest_prefix_within_the_risk_limit():
    rows = [item({"A": 0.99, "B": 0.01}, "A", "A")] * 9 + [item({"A": 0.5, "B": 0.5}, "B", "A")]
    assert probability_scores(rows)["coverage_at_10_risk"] == 1.0  # 1 error in 10 is exactly 10%


@pytest.mark.parametrize("confidence,index", [
    (0.0, 0), (0.1, 1), (0.29999, 2), (0.3, 3), (0.95, 9), (0.9, 9), (1.0, 9),
])
def test_ece_bins_have_exact_tenth_edges_and_include_one(confidence, index):
    assert ece_bin_index(confidence) == index


def test_an_empty_group_or_a_vector_off_its_sum_has_no_scores():
    assert probability_scores([])["probability_metrics"]["reason"] == NO_ITEMS
    off = item({"A": 0.7, "B": 0.2}, "A", "A")
    assert probability_scores([*ITEMS, off])["probability_metrics"]["reason"] == MALFORMED


@pytest.mark.parametrize("values,percent,expected", [
    ([], 50, None), ([7.0], 95, 7.0), ([1.0, 2.0, 3.0, 4.0], 50, 2.5), ([10.0, 20.0, 30.0], 95, 29.0),
])
def test_the_percentile_interpolates_linearly(values, percent, expected):
    assert _percentile(values, percent) == expected
