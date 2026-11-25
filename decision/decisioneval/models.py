"""Shared data types for questions, decisions, and run records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def option_id_for_index(index: int) -> str:
    """Return A, B, ... Z, AA for a zero-based option index."""
    if index < 0:
        raise ValueError(f"option index must be >= 0, got {index}")
    number = index + 1
    characters: list[str] = []
    while number:
        number, remainder = divmod(number - 1, 26)
        characters.append(chr(ord("A") + remainder))
    return "".join(reversed(characters))


@dataclass(frozen=True)
class Question:
    """One MCQ item in the internal evaluation format."""

    id: str | int
    dataset_name: str
    category: str
    task: str
    instruction: str
    input: str
    options: tuple[str, ...]
    option_ids: tuple[str, ...]
    gold_answer: str
    gold_option_id: str


@dataclass(frozen=True)
class DecisionResult:
    """What an adapter returns for one question.

    `predicted_option_id` and `predicted_answer` are both optional.
    The evaluator accepts either one and rejects the item when both are
    present and point at different options. `probabilities`, when a model
    actually returns them, are keyed by option id. This type never fills
    them in. `confidence` is the provider's own confidence, from 0 to 1 when
    the model returns it. Calibration metrics do not read it; they use
    max(probabilities).
    `latency_ms` is HTTP request time only. `elapsed_ms` adds retry backoff.
    `attempts` is how many requests were sent for this one decision.
    """

    predicted_option_id: str | None = None
    predicted_answer: str | None = None
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    raw_response: Any = None
    latency_ms: float | None = None
    usage: dict[str, Any] | None = None
    cost: float | None = None
    valid: bool = True
    model_requested: str | None = None
    model_resolved: str | None = None
    attempts: int = 1
    retried: bool = False
    elapsed_ms: float | None = None


@dataclass(frozen=True)
class ItemResult:
    """Stored outcome for one test question."""

    item_id: str | int
    dataset: str
    category: str
    task: str
    options: tuple[str, ...]
    option_ids: tuple[str, ...]
    gold_answer: str
    gold_option_id: str
    predicted_option_id: str | None
    predicted_answer: str | None
    probabilities: dict[str, float] | None
    correct: bool
    invalid: bool
    latency_ms: float | None
    usage: dict[str, Any] | None
    cost: float | None
    raw_response: Any
    # Provider-reported confidence. Calibration metrics use max(probabilities).
    confidence: float | None = None
    model_requested: str | None = None
    model_resolved: str | None = None
    attempts: int = 1
    retried: bool = False
    elapsed_ms: float | None = None
    # "malformed" when the contract dropped a vector the model did send, so calibration says why.
    distribution_problem: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "dataset": self.dataset,
            "category": self.category,
            "task": self.task,
            "options": list(self.options),
            "option_ids": list(self.option_ids),
            "gold_answer": self.gold_answer,
            "gold_option_id": self.gold_option_id,
            "predicted_option_id": self.predicted_option_id,
            "predicted_answer": self.predicted_answer,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
            "provider_confidence": self.confidence,
            "correct": self.correct,
            "invalid": self.invalid,
            "latency_ms": self.latency_ms,
            "usage": self.usage,
            "cost": self.cost,
            "raw_response": self.raw_response,
            "model_requested": self.model_requested,
            "model_resolved": self.model_resolved,
            "attempts": self.attempts,
            "retried": self.retried,
            "elapsed_ms": self.elapsed_ms,
            "distribution_problem": self.distribution_problem,
        }

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> ItemResult:
        return cls(
            item_id=record.get("item_id"),
            dataset=str(record.get("dataset")),
            category=str(record.get("category")),
            task=str(record.get("task")),
            options=tuple(record.get("options") or ()),
            option_ids=tuple(record.get("option_ids") or ()),
            gold_answer=str(record.get("gold_answer")),
            gold_option_id=str(record.get("gold_option_id")),
            predicted_option_id=record.get("predicted_option_id"),
            predicted_answer=record.get("predicted_answer"),
            probabilities=record.get("probabilities"),
            correct=bool(record.get("correct")),
            invalid=bool(record.get("invalid")),
            latency_ms=record.get("latency_ms"),
            usage=record.get("usage"),
            cost=record.get("cost"),
            raw_response=record.get("raw_response"),
            confidence=record.get("confidence"),
            model_requested=record.get("model_requested"),
            model_resolved=record.get("model_resolved"),
            attempts=int(record.get("attempts") or 1),
            retried=bool(record.get("retried")),
            elapsed_ms=record.get("elapsed_ms"),
            distribution_problem=record.get("distribution_problem"),
        )


@dataclass(frozen=True)
class ExcludedItem:
    """A test question left out of scoring, with a stable reason code."""

    item_id: str | int | None
    reason: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class Dataset:
    """A loaded MCQ dataset. Questions come from the test split only."""

    name: str
    category: str
    task: str
    source_path: str
    questions: tuple[Question, ...]
    excluded: tuple[ExcludedItem, ...] = ()


@dataclass(frozen=True)
class SkippedFile:
    path: str
    reason: str


@dataclass(frozen=True)
class LoadReport:
    datasets: tuple[Dataset, ...]
    skipped: tuple[SkippedFile, ...]
