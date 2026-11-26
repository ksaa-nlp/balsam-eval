"""Stored run -> the per-pool result files the backend ingests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from decisioneval.metrics.probability import probability_scores
from decisioneval.remote.normalize import PoolDocument
from decisioneval.remote.bound_adapter import BoundAdapter, mapping_origin
from decisioneval.remote.pools import PreparedPools, slug_for
from decisioneval.transforms.engine import ENGINE, ENGINE_VERSION

POOL_METRICS = (
    "accuracy",
    "invalid_rate",
    "brier_normalized",
    "ece",
    "aurc",
    "coverage_at_10_risk",
    "p50_latency_ms",
    "p95_latency_ms",
)
SAMPLE_GROUP = "decision"
META_FILE = "_decision-meta.json"
RUN_FILE = "_run.json"
INVALID_REASONS = ("refusal", "invalid_choice", "malformed_response", "adapter_error")
CALIBRATION_METRICS = frozenset({"brier_normalized", "ece", "aurc", "coverage_at_10_risk"})
CALIBRATION_SCORES = frozenset({"brier_normalized", "decision_confidence"})
DERIVED = "derived_probabilities"


def result_filename(source: str) -> str:
    return f"{slug_for(source)}.json"


def pool_results(
    pools: PreparedPools, run: dict[str, Any], calibrated: bool = True
) -> dict[str, dict[str, Any]]:
    """File name -> result document, one per pool that has scored items.

    Not `calibrated` means the adapter derived its vectors (a softmax over logits):
    they stay in the run file, and no calibration score is published from them.
    """
    by_dataset = (run.get("summary") or {}).get("by_dataset") or {}
    items_by_pool: dict[str, list[dict[str, Any]]] = {}
    for item in run.get("items") or []:
        items_by_pool.setdefault(str(item.get("dataset")), []).append(item)

    files: dict[str, dict[str, Any]] = {}
    for pool in pools.pools:
        items = items_by_pool.get(pool.source)
        if not items:
            continue
        files[result_filename(pool.source)] = pool_result(
            pool, by_dataset.get(pool.source) or {}, items, calibrated
        )
    return files


def pool_result(
    pool: PoolDocument, summary: dict[str, Any], items: list[dict[str, Any]], calibrated: bool = True
) -> dict[str, Any]:
    published = POOL_METRICS if calibrated else tuple(m for m in POOL_METRICS if m not in CALIBRATION_METRICS)
    return {
        "pool_file": pool.source,
        "category": pool.category,
        "task": pool.task,
        "average_scores": {
            metric: summary[metric]
            for metric in published
            if isinstance(summary.get(metric), (int, float))
        },
        "probability_metrics": (
            (summary.get("probability_metrics") or {}).get("status") if calibrated else DERIVED
        ),
        "samples": {SAMPLE_GROUP: [item_sample(pool, item, calibrated) for item in items]},
    }


def item_sample(pool: PoolDocument, item: dict[str, Any], calibrated: bool = True) -> dict[str, Any]:
    doc: dict[str, Any] = {"id": item.get("item_id")}
    dataset_item_id = pool.dataset_item_id(item.get("item_id"))
    if dataset_item_id is not None:
        doc["dataset_item_id"] = dataset_item_id
    return {
        "doc": doc,
        "scores": {
            name: value
            for name, value in item_scores(item).items()
            if calibrated or name not in CALIBRATION_SCORES
        },
        "model_output": item.get("predicted_answer") or item.get("predicted_option_id"),
        "predicted_option_id": item.get("predicted_option_id"),
        "gold_option_id": item.get("gold_option_id"),
        "probabilities": item.get("probabilities") if calibrated else None,
        "provider_confidence": item.get("confidence"),
        "latency_ms": item.get("latency_ms"),
    }


def item_scores(item: dict[str, Any]) -> dict[str, float]:
    """Keys are metric values whose per-question mean is the metric; ECE/AURC aren't means, so absent."""
    scores: dict[str, float] = {
        "accuracy": 1.0 if item.get("correct") else 0.0,
        "invalid_rate": 1.0 if item.get("invalid") else 0.0,
    }
    if isinstance(item.get("latency_ms"), (int, float)):
        scores["latency_ms"] = float(item["latency_ms"])
    single = probability_scores([_probability_item(item)])
    if single["probability_metrics"]["status"] == "available":
        scores["brier_normalized"] = single["brier_normalized"]
        scores["decision_confidence"] = max(item["probabilities"].values())
    return scores


def _probability_item(item: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        probabilities=item.get("probabilities"),
        option_ids=tuple(item.get("option_ids") or ()),
        gold_option_id=item.get("gold_option_id"),
        predicted_option_id=item.get("predicted_option_id"),
        confidence=item.get("confidence"),
        distribution_problem=item.get("distribution_problem"),
    )


def problem_counts(items: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Why items were invalid and why their vectors were unusable, so a null calibration says why."""
    invalid = dict.fromkeys(INVALID_REASONS, 0)
    distribution = {"missing": 0, "malformed": 0}
    for item in items:
        raw = item.get("raw_response") if isinstance(item.get("raw_response"), dict) else {}
        if item.get("invalid"):
            reason = raw.get("invalid_reason")
            invalid[reason if reason in invalid else "malformed_response"] += 1
        if item.get("distribution_problem") in distribution:
            distribution[item["distribution_problem"]] += 1
    return {"invalid_reasons": invalid, "distribution_problems": distribution}


def meta_document(
    *,
    bound: BoundAdapter,
    pools: PreparedPools,
    items: list[dict[str, Any]],
    state: str,
    image_digest: str | None = None,
    resumed_items: int = 0,
) -> dict[str, Any]:
    """What the backend stores as the job's `runner_meta`. No raw samples."""
    return {
        "state": state,
        "mapping_origin": mapping_origin(bound),
        "degraded": bound.degraded,
        "derived_probabilities": bound.derived_probabilities,
        "pools_scored": [pool.source for pool in pools.pools],
        "pools_skipped": [{"pool_file": skip.source, "reason": skip.reason} for skip in pools.skipped],
        **problem_counts(items),
        "adapter_version_id": bound.version_id,
        "adapter_content_hash": bound.content_hash,
        "engine": ENGINE,
        "engine_version": ENGINE_VERSION,
        "runner_image_digest": image_digest,
        "resumed_items": resumed_items,
        "circuit_broken": state == "circuit_broken",
    }
