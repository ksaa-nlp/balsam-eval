"""The decision contract every response transform output is held to.

Mirrors EVALUATION.md §2: a vector is kept exactly as sent or dropped, never
rescaled or filled, and an invalid answer and an invalid vector are independent.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from decisioneval.metrics.probability import PROBABILITY_SUM_TOLERANCE

REFUSAL = "refusal"
INVALID_CHOICE = "invalid_choice"
ADAPTER_ERROR = "adapter_error"
MISSING = "missing"
MALFORMED = "malformed"
STATUSES = frozenset({"answer", "refusal"})
OUTPUT_KEYS = frozenset({"status", "choice", "probabilities"})


@dataclass(frozen=True)
class Checked:
    choice: str | None = None
    invalid_reason: str | None = None
    probabilities: dict[str, float] | None = None
    distribution_problem: str | None = None

    @property
    def breaker_error(self) -> bool:
        """A sign the adapter is broken, not that the model declined to answer."""
        return self.invalid_reason == ADAPTER_ERROR or self.distribution_problem == MALFORMED


def check(output: Any, option_ids: Sequence[str]) -> Checked:
    """Hold one response transform output to the contract."""
    if not _on_schema(output):
        return Checked(invalid_reason=ADAPTER_ERROR, distribution_problem=MISSING)
    if output["status"] == "refusal":
        return Checked(invalid_reason=REFUSAL, distribution_problem=MISSING)

    probabilities, problem = _vector(output.get("probabilities"), option_ids)
    choice = output.get("choice")
    if choice not in option_ids:
        return Checked(invalid_reason=INVALID_CHOICE, probabilities=probabilities, distribution_problem=problem)
    return Checked(choice=choice, probabilities=probabilities, distribution_problem=problem)


def _on_schema(output: Any) -> bool:
    return (
        isinstance(output, dict)
        and set(output) <= OUTPUT_KEYS
        and output.get("status") in STATUSES
        and (output.get("choice") is None or isinstance(output["choice"], str))
        and (output.get("probabilities") is None or isinstance(output["probabilities"], dict))
    )


def _vector(value: dict[str, Any] | None, option_ids: Sequence[str]) -> tuple[dict[str, float] | None, str | None]:
    if value is None:
        return None, MISSING
    if set(value) != set(option_ids) or not all(_unit(value[key]) for key in option_ids):
        return None, MALFORMED
    if abs(sum(value.values()) - 1.0) > PROBABILITY_SUM_TOLERANCE:
        return None, MALFORMED
    return {key: value[key] for key in option_ids}, None


def _unit(value: Any) -> bool:
    # Range first: math.isfinite overflows on a huge int, which the range check already rejects.
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and 0.0 <= value <= 1.0
        and math.isfinite(value)
    )
