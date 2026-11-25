"""Recompute a stored run from its raw item records.

One mechanism: replay every item's stored reply through the run's own
response transform, re-check the decision contract, rebuild the summary from
the replayed values, and compare. A run that cannot reproduce its stored
numbers fails.

Tolerances match the origin: metrics 1e-9, probability vectors 1e-6.
Nothing in this module edits a run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from decisioneval.metrics.summary import build_summary
from decisioneval.models import ItemResult
from decisioneval.remote.contract import Checked, check
from decisioneval.store import read_run, secret_paths

REDACTED = "[redacted]"

METRIC_TOLERANCE = 1e-9
PROBABILITY_MATCH_TOLERANCE = 1e-6

AUDITED_METRICS = (
    "total_items",
    "correct",
    "invalid",
    "accuracy",
    "invalid_rate",
    "brier_raw",
    "brier_normalized",
    "ece",
    "aurc",
    "coverage_at_10_risk",
    "p50_latency_ms",
    "p95_latency_ms",
)


@dataclass
class Finding:
    scope: str
    detail: str


@dataclass
class RunAudit:
    run_id: str
    model: str
    items: int
    findings: list[Finding] = field(default_factory=list)
    unreplayed: int = 0  # replies that held a redacted field, so their mapping can't be replayed

    @property
    def passed(self) -> bool:
        return not self.findings

    def report(self) -> str:
        head = f"{self.model} ({self.run_id}, {self.items} items) ... {'PASS' if self.passed else 'FAIL'}"
        if self.unreplayed:
            head += f" ({self.unreplayed} redacted replies checked against their stored mapping only)"
        return "\n".join([head] + [f"  {f.scope}: {f.detail}" for f in self.findings])


def audit_run_file(path: str | Path) -> RunAudit:
    return audit_document(read_run(path))


def audit_document(document: dict[str, Any]) -> RunAudit:
    """Replay one run and compare it with what it stored."""
    items = document.get("items") if isinstance(document.get("items"), list) else []
    audit = RunAudit(
        run_id=str(document.get("run_id")),
        model=str(document.get("model")),
        items=len(items),
    )

    leaked = secret_paths(document)
    if leaked:
        audit.findings.append(Finding("secrets", f"credential values present at {leaked[:3]}"))

    adapter = document.get("adapter")
    if not isinstance(adapter, dict):
        audit.findings.append(Finding("adapter", "the run names no adapter to replay"))
        return audit
    replayed = _replay_adapter_run(items, str(adapter.get("response_transform") or ""), audit)

    if audit.findings:
        return audit

    recomputed = build_summary(replayed)
    _compare_summary(document.get("summary"), recomputed, audit)
    return audit


def _replay_adapter_run(items: list[Any], transform: str, audit: RunAudit) -> list[ItemResult]:
    """Re-run the adapter's response transform on every stored reply and re-check the contract."""
    from decisioneval.remote.execute import response_input
    from decisioneval.transforms.engine import TransformError
    from decisioneval.transforms.worker import TransformWorker

    replayed: list[ItemResult] = []
    with TransformWorker() as worker:
        for record in items:
            if not isinstance(record, dict):
                audit.findings.append(Finding("item", "record is not an object"))
                continue
            where = f"{record.get('dataset')}/{record.get('item_id')}"
            raw = record.get("raw_response") if isinstance(record.get("raw_response"), dict) else {}
            option_ids = tuple(str(value) for value in record.get("option_ids") or ())
            if "mapped" in raw and _has_redacted_value(raw.get("reply")):
                audit.unreplayed += 1
                checked = check(raw["mapped"], option_ids)
            elif "mapped" in raw:
                try:
                    options = tuple(str(value) for value in record.get("options") or ())
                    again = worker.run(transform, response_input(raw.get("reply"), option_ids, options))
                except TransformError as exc:
                    audit.findings.append(Finding("transform", f"{where}: {exc}"))
                    continue
                if again != raw["mapped"]:
                    audit.findings.append(Finding("transform", f"{where}: stored mapping differs from its replay"))
                checked = check(raw["mapped"], option_ids)
            else:
                # The transform never produced output (it failed, or the reply wasn't JSON): nothing scored.
                checked = Checked(invalid_reason=raw.get("invalid_reason"), distribution_problem=raw.get("distribution_problem"))
            if checked.choice != record.get("predicted_option_id"):
                audit.findings.append(
                    Finding("choice", f"{where}: stored {record.get('predicted_option_id')!r}, replayed {checked.choice!r}")
                )
            if (checked.choice is None) != bool(record.get("invalid")):
                audit.findings.append(Finding("invalid", f"{where}: stored {record.get('invalid')}"))
            if checked.distribution_problem != record.get("distribution_problem"):
                audit.findings.append(Finding("distribution", f"{where}: stored {record.get('distribution_problem')!r}"))
            _compare_vector(where, record.get("probabilities"), checked.probabilities, audit)
            replayed.append(ItemResult.from_dict(record))
    return replayed


def _has_redacted_value(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_has_redacted_value(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_redacted_value(item) for item in value)
    return value == REDACTED


def _compare_vector(
    where: str,
    stored: Any,
    replayed: dict[str, float] | None,
    audit: RunAudit,
) -> None:
    if stored is None and replayed is None:
        return
    if stored is None or replayed is None:
        audit.findings.append(
            Finding(
                "probabilities",
                f"{where}: stored {'absent' if stored is None else 'present'}, "
                f"replayed {'absent' if replayed is None else 'present'}",
            )
        )
        return
    if set(stored) != set(replayed):
        audit.findings.append(Finding("probabilities", f"{where}: key sets differ"))
        return
    for key, value in replayed.items():
        if abs(float(stored[key]) - float(value)) > PROBABILITY_MATCH_TOLERANCE:
            audit.findings.append(
                Finding("probabilities", f"{where}: {key} stored {stored[key]!r}, replayed {value!r}")
            )
            return


def _compare_summary(stored: Any, recomputed: dict[str, Any], audit: RunAudit) -> None:
    if not isinstance(stored, dict):
        audit.findings.append(Finding("summary", "stored summary is not an object"))
        return
    for scope in ("overall",):
        _compare_block(scope, stored.get(scope), recomputed.get(scope), audit)
    for scope in ("by_dataset", "by_category"):
        stored_map = stored.get(scope) or {}
        recomputed_map = recomputed.get(scope) or {}
        if set(stored_map) != set(recomputed_map):
            audit.findings.append(Finding(scope, "key sets differ"))
            continue
        for key in sorted(recomputed_map):
            _compare_block(f"{scope}.{key}", stored_map[key], recomputed_map[key], audit)


def _compare_block(scope: str, stored: Any, recomputed: Any, audit: RunAudit) -> None:
    if not isinstance(stored, dict) or not isinstance(recomputed, dict):
        audit.findings.append(Finding(scope, "block missing"))
        return
    for metric in AUDITED_METRICS:
        if metric not in recomputed:
            continue
        if metric not in stored:
            audit.findings.append(Finding(f"{scope}.{metric}", "absent from the stored summary"))
            continue
        left, right = stored[metric], recomputed[metric]
        if left is None and right is None:
            continue
        if left is None or right is None:
            audit.findings.append(Finding(f"{scope}.{metric}", f"stored {left!r}, recomputed {right!r}"))
            continue
        if not math.isfinite(float(left)) or not math.isfinite(float(right)):
            audit.findings.append(Finding(f"{scope}.{metric}", "non-finite value"))
            continue
        if abs(float(left) - float(right)) > METRIC_TOLERANCE:
            audit.findings.append(
                Finding(f"{scope}.{metric}", f"stored {left!r}, recomputed {right!r}")
            )
