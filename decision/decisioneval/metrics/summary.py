"""Accuracy, invalid rate, latency percentiles, and probability scores.

P50 and P95 use `latency_ms`, which is HTTP request time and excludes retry backoff.
Probability scores are null unless every item in the group has a valid vector.
`overall` pools items with equal weight (micro). A macro mean across datasets
is not defined yet.
"""

import math
from collections import defaultdict

from decisioneval.metrics.probability import probability_scores
from decisioneval.models import ItemResult


def build_summary(items: list[ItemResult]) -> dict:
    """Build overall, per-dataset, and per-category summaries.

    Each block is ``_summarize`` on the items in that group. Category accuracy
    is correct items divided by category items. A missing latency is left out
    of P50/P95 only.
    """
    by_dataset_items: dict[str, list[ItemResult]] = defaultdict(list)
    by_category_items: dict[str, list[ItemResult]] = defaultdict(list)
    for item in items:
        by_dataset_items[item.dataset].append(item)
        by_category_items[item.category].append(item)

    return {
        "overall": _summarize(items),
        "by_dataset": {
            name: _summarize(group) for name, group in by_dataset_items.items()
        },
        "by_category": {
            name: _summarize(group) for name, group in by_category_items.items()
        },
        "probability_aggregation": {
            "overall": "micro",
            "macro_across_datasets": None,
            "macro_reason": "undefined",
        },
    }


def _summarize(items: list[ItemResult]) -> dict:
    total = len(items)
    correct = sum(1 for item in items if item.correct)
    invalid = sum(1 for item in items if item.invalid)
    latencies = [item.latency_ms for item in items if item.latency_ms is not None]
    scores = probability_scores(items)
    return {
        "total_items": total,
        "correct": correct,
        "invalid": invalid,
        "accuracy": (correct / total) if total else 0.0,
        "invalid_rate": (invalid / total) if total else 0.0,
        "p50_latency_ms": _round_ms(_percentile(latencies, 50)),
        "p95_latency_ms": _round_ms(_percentile(latencies, 95)),
        "brier_raw": scores["brier_raw"],
        "brier_normalized": scores["brier_normalized"],
        "ece": scores["ece"],
        "aurc": scores["aurc"],
        "coverage_at_10_risk": scores["coverage_at_10_risk"],
        "probability_metrics": scores["probability_metrics"],
    }


def _percentile(values: list[float], percent: float) -> float | None:
    """Linear percentile. Rank is `percent/100 * (n - 1)`."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (percent / 100) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * weight)


def _round_ms(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 3)
