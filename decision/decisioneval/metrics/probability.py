"""Probability scores for one evaluation group.

`decision_confidence` is max(probabilities). The item field `confidence` is
provider confidence and is never read here.

Brier compares probabilities with a deterministic gold one-hot vector.
That vector is not a reference probability distribution, and probabilities
are not renormalized.

Reported Brier values are means of the per-item scores. `brier_normalized`
is the mean of `brier_raw / K`, so questions with different option counts
stay on a comparable scale. The leaderboard uses `brier_normalized` (lower
is better).

ECE uses 10 equal-width bins on [0, 1]. Bins 0..8 are [b/10, (b+1)/10).
Bin 9 is [0.9, 1.0], so confidence 1.0 is included. A confidence that
prints as an exact tenth, such as 0.3, is placed by that decimal value.
Empty bins add nothing. ECE is the count-weighted mean absolute gap
between bin accuracy and mean decision confidence.

AURC sorts by decision_confidence descending. Ties keep the earlier item.
For prefix k = 1..n, coverage is k/n and risk is errors_in_prefix / k.
AURC is the right Riemann sum

    (1/n) * sum_{k=1}^{n} risk(k)

Lower is better.

Coverage at 10% risk is the largest prefix coverage whose own risk is
<= 0.10. An earlier prefix may sit above 10% when later correct items
bring the cumulative risk back down. Higher is better.

If any item in the group lacks a valid probability vector, every
probability score in that group is null. Missing vectors and malformed
vectors are separate reasons. A malformed vector takes precedence when
both occur. Scores are not filled in.
"""

import math
from collections.abc import Sequence
from decimal import Decimal
from typing import Any, Protocol

# Same absolute tolerance the Jev adapter uses. Values are not rescaled.
PROBABILITY_SUM_TOLERANCE = 1e-2
ECE_BIN_COUNT = 10
COVERAGE_RISK_LIMIT = 0.10
MISSING = "missing_probabilities"
MALFORMED = "malformed_probabilities"
NO_ITEMS = "no_items"


class ProbabilityItem(Protocol):
    probabilities: dict[str, float] | None
    option_ids: Sequence[str]
    gold_option_id: str
    predicted_option_id: str | None
    confidence: float | None


def ece_bin_index(confidence: float, bin_count: int = ECE_BIN_COUNT) -> int:
    """Return the equal-width bin for a decision confidence in [0, 1]."""
    if confidence >= 1.0:
        return bin_count - 1
    if confidence <= 0.0:
        return 0
    scaled = Decimal(str(confidence)) * bin_count
    index = math.floor(scaled)
    if index >= bin_count:
        return bin_count - 1
    return int(index)


def probability_scores(items: Sequence[ProbabilityItem]) -> dict[str, Any]:
    """Score one group. Overall callers pass every item; dataset callers pass one dataset."""
    if not items:
        return _unavailable(NO_ITEMS)

    problems = [_probability_problem(item) for item in items]
    if any(problem == MALFORMED for problem in problems):
        return _unavailable(MALFORMED)
    if any(problem == MISSING for problem in problems):
        return _unavailable(MISSING)

    rows = [_row(item) for item in items]
    raw_scores = [_brier_raw(row) for row in rows]
    normalized_scores = [raw / row["k"] for raw, row in zip(raw_scores, rows, strict=True)]
    count = len(rows)
    return {
        "brier_raw": sum(raw_scores) / count,
        "brier_normalized": sum(normalized_scores) / count,
        "ece": _ece(rows),
        "aurc": _aurc(rows),
        "coverage_at_10_risk": _coverage_at_risk(rows, COVERAGE_RISK_LIMIT),
        "probability_metrics": {
            "status": "available",
            "reason": None,
            "items": count,
            "decision_confidence": "max(probabilities)",
            "provider_confidence_field": "confidence",
            "brier_raw": "mean of sum_k (p_k - y_k)^2 with deterministic gold one-hot y",
            "brier_normalized": "mean of brier_raw / K",
            "ece_bin_count": ECE_BIN_COUNT,
            "ece_bins": _ece_bins(rows),
            "aurc": "(1/n) * sum of prefix risks after sorting by decision_confidence",
            "coverage_at_10_risk": "longest confidence prefix whose risk is <= 0.10",
        },
    }


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "brier_raw": None,
        "brier_normalized": None,
        "ece": None,
        "aurc": None,
        "coverage_at_10_risk": None,
        "probability_metrics": {
            "status": "not_available",
            "reason": reason,
            "items": None,
            "decision_confidence": "max(probabilities)",
            "provider_confidence_field": "confidence",
            "brier_raw": "mean of sum_k (p_k - y_k)^2 with deterministic gold one-hot y",
            "brier_normalized": "mean of brier_raw / K",
            "ece_bin_count": None,
            "ece_bins": None,
            "aurc": "(1/n) * sum of prefix risks after sorting by decision_confidence",
            "coverage_at_10_risk": "longest confidence prefix whose risk is <= 0.10",
        },
    }


def _probability_problem(item: ProbabilityItem) -> str | None:
    # A vector the contract refused is stored as None, but it was malformed, not missing.
    if getattr(item, "distribution_problem", None) == "malformed":
        return MALFORMED
    probabilities = item.probabilities
    if probabilities is None:
        return MISSING
    option_ids = tuple(item.option_ids)
    if not option_ids or not isinstance(probabilities, dict):
        return MALFORMED
    if set(probabilities) != set(option_ids):
        return MALFORMED
    if item.gold_option_id not in option_ids:
        return MALFORMED
    total = 0.0
    for option_id in option_ids:
        value = probabilities[option_id]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return MALFORMED
        number = float(value)
        if not math.isfinite(number) or number < 0.0 or number > 1.0:
            return MALFORMED
        total += number
    if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
        return MALFORMED
    return None


def _row(item: ProbabilityItem) -> dict[str, Any]:
    option_ids = tuple(item.option_ids)
    probabilities = {option_id: float(item.probabilities[option_id]) for option_id in option_ids}
    return {
        "option_ids": option_ids,
        "probabilities": probabilities,
        "gold_option_id": item.gold_option_id,
        "correct": item.predicted_option_id == item.gold_option_id,
        "decision_confidence": max(probabilities.values()),
        "k": len(option_ids),
    }


def _brier_raw(row: dict[str, Any]) -> float:
    total = 0.0
    for option_id in row["option_ids"]:
        gold = 1.0 if option_id == row["gold_option_id"] else 0.0
        total += (row["probabilities"][option_id] - gold) ** 2
    return total


def _ece(rows: list[dict[str, Any]]) -> float:
    bins = _ece_bins(rows)
    count = len(rows)
    score = 0.0
    for entry in bins:
        if entry["count"] == 0:
            continue
        gap = abs(entry["accuracy"] - entry["mean_decision_confidence"])
        score += (entry["count"] / count) * gap
    return score


def _ece_bins(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: list[list[dict[str, Any]]] = [[] for _ in range(ECE_BIN_COUNT)]
    for row in rows:
        grouped[ece_bin_index(row["decision_confidence"])].append(row)
    bins: list[dict[str, Any]] = []
    for index, group in enumerate(grouped):
        lower = index / ECE_BIN_COUNT
        upper = (index + 1) / ECE_BIN_COUNT
        last = index == ECE_BIN_COUNT - 1
        if group:
            accuracy = sum(1.0 if row["correct"] else 0.0 for row in group) / len(group)
            mean_confidence = sum(row["decision_confidence"] for row in group) / len(group)
        else:
            accuracy = None
            mean_confidence = None
        bins.append(
            {
                "index": index,
                "lower": lower,
                "upper": 1.0 if last else upper,
                "includes_upper": last,
                "count": len(group),
                "accuracy": accuracy,
                "mean_decision_confidence": mean_confidence,
            }
        )
    return bins


def _ordered(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for _, row in sorted(
            enumerate(rows),
            key=lambda pair: (-pair[1]["decision_confidence"], pair[0]),
        )
    ]


def _prefix_risks(rows: list[dict[str, Any]]) -> list[float]:
    ordered = _ordered(rows)
    errors = 0
    risks: list[float] = []
    for index, row in enumerate(ordered, start=1):
        if not row["correct"]:
            errors += 1
        risks.append(errors / index)
    return risks


def _aurc(rows: list[dict[str, Any]]) -> float:
    risks = _prefix_risks(rows)
    return sum(risks) / len(risks)


def _coverage_at_risk(rows: list[dict[str, Any]], limit: float) -> float:
    risks = _prefix_risks(rows)
    best = 0
    for index, risk in enumerate(risks, start=1):
        if risk <= limit:
            best = index
    return best / len(risks)
