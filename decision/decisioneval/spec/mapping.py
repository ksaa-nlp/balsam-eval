"""Turn one raw model response into the canonical decision fields.

Everything this module returns is either copied verbatim out of the raw
response or is an option id taken from the question. No value is computed.
That is the whole reason the mapping layer is declarative: see
:mod:`decisioneval.spec.paths`.

Probability rules, in order:

1. Read the vector at ``response.probabilities.path``.
2. Re-key it onto the question's option ids. Keys are *matched*, never
   renumbered, and each value is carried across as it was found.
3. If the key set does not land exactly on the option ids, the vector is
   unavailable. Nothing is filled in and nothing is dropped to make it fit.
4. If the vector does not sum to 1 within ``PROBABILITY_SUM_TOLERANCE``, apply
   ``on_bad_sum``: ``unavailable`` keeps the choice and reports null
   probability metrics, ``reject`` marks the whole item invalid. Neither
   renormalizes. Logits reach this branch and stay unscored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from decisioneval.metrics.probability import PROBABILITY_SUM_TOLERANCE
from decisioneval.spec.model import Condition, ModelSpec, ProbabilityMap
from decisioneval.spec.paths import MISSING, get

NO_CHOICE = "no-choice"
BAD_CHOICE = "bad-choice"
NO_PROBABILITIES = "no-probabilities"
PROBABILITY_KEYS = "probability-keys"
PROBABILITY_VALUES = "probability-values"
PROBABILITY_SUM = "probability-sum"
INVALID_RULE = "invalid-rule"


@dataclass(frozen=True)
class Mapped:
    """The spec's reading of one raw response."""

    predicted_option_id: str | None = None
    probabilities: dict[str, float] | None = None
    provider_confidence: float | None = None
    valid: bool = True
    problems: tuple[str, ...] = field(default_factory=tuple)


def condition_holds(raw: Any, rule: Condition) -> bool:
    value = get(raw, rule.path)
    if rule.mode == "missing":
        return value is MISSING
    if rule.mode == "empty":
        return value is MISSING or value is None or value == "" or value == [] or value == {}
    return value is not MISSING and value == rule.equals


def map_response(
    spec: ModelSpec,
    raw: Any,
    option_ids: tuple[str, ...],
    options: tuple[str, ...],
) -> Mapped:
    """Apply `spec`'s response block to `raw` for one question."""
    rules = spec.response
    body = raw if not rules.envelope else get(raw, rules.envelope)
    if body is MISSING:
        return Mapped(valid=False, problems=(NO_CHOICE,))

    for rule in rules.invalid_when:
        if condition_holds(body, rule):
            return Mapped(valid=False, problems=(INVALID_RULE,))

    problems: list[str] = []
    choice_value = get(body, rules.choice.path)
    option_id = (
        None
        if choice_value is MISSING
        else _resolve_label(choice_value, rules.choice.kind, option_ids, options)
    )
    if choice_value is MISSING:
        problems.append(NO_CHOICE)
    elif option_id is None:
        problems.append(BAD_CHOICE)

    probabilities, prob_problems = _probabilities(rules.probabilities, body, option_ids, options)
    problems.extend(prob_problems)
    if PROBABILITY_SUM in prob_problems and rules.on_bad_sum == "reject":
        return Mapped(valid=False, problems=tuple(problems))

    confidence = None
    if rules.provider_confidence is not None:
        value = get(body, rules.provider_confidence.path)
        confidence = _number(value)

    return Mapped(
        predicted_option_id=option_id,
        probabilities=probabilities,
        provider_confidence=confidence,
        valid=option_id is not None,
        problems=tuple(problems),
    )


def _probabilities(
    rules: ProbabilityMap | None,
    body: Any,
    option_ids: tuple[str, ...],
    options: tuple[str, ...],
) -> tuple[dict[str, float] | None, list[str]]:
    if rules is None:
        return None, []
    found = get(body, rules.path)
    if found is MISSING:
        return None, [NO_PROBABILITIES]

    pairs = _pairs(rules, found, option_ids, options)
    if pairs is None:
        return None, [PROBABILITY_KEYS]

    vector: dict[str, float] = {}
    for option_id, value in pairs:
        number = _number(value)
        if number is None:
            return None, [PROBABILITY_VALUES]
        if option_id in vector:
            return None, [PROBABILITY_KEYS]
        vector[option_id] = number
    if set(vector) != set(option_ids):
        return None, [PROBABILITY_KEYS]

    total = sum(vector.values())
    if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
        # Not renormalized and not softmaxed. The policy decides what happens.
        return None, [PROBABILITY_SUM]
    return vector, []


def _pairs(
    rules: ProbabilityMap,
    found: Any,
    option_ids: tuple[str, ...],
    options: tuple[str, ...],
) -> list[tuple[str, Any]] | None:
    """Pair each raw entry with an option id, or None when it cannot be done."""
    if rules.keying == "order":
        if not isinstance(found, list) or len(found) != len(option_ids):
            return None
        return list(zip(option_ids, found, strict=True))

    if rules.keying == "pairs":
        if not isinstance(found, list):
            return None
        pairs: list[tuple[str, Any]] = []
        for entry in found:
            label = get(entry, rules.label_path or "")
            score = get(entry, rules.score_path or "")
            if label is MISSING or score is MISSING:
                return None
            option_id = _resolve_label(label, rules.label_kind, option_ids, options)
            if option_id is None:
                return None
            pairs.append((option_id, score))
        return pairs

    if not isinstance(found, dict):
        return None
    pairs = []
    for key, value in found.items():
        option_id = _resolve_label(key, rules.keying, option_ids, options)
        if option_id is None:
            return None
        pairs.append((option_id, value))
    return pairs


def _resolve_label(
    value: Any,
    kind: str,
    option_ids: tuple[str, ...],
    options: tuple[str, ...],
) -> str | None:
    """Map a raw label onto one of the question's option ids."""
    if kind in ("index0", "index1"):
        index = _integer(value)
        if index is None:
            return None
        if kind == "index1":
            index -= 1
        if not 0 <= index < len(option_ids):
            return None
        return option_ids[index]

    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    text = str(value).strip()
    if not text:
        return None

    if kind == "option_text":
        matches = [
            option_id
            for option_id, option_text in zip(option_ids, options, strict=True)
            if option_text == text
        ]
        return matches[0] if len(matches) == 1 else None

    lookup = {option_id.upper(): option_id for option_id in option_ids}
    return lookup.get(text.upper())


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text)
        except ValueError:
            return None
    return None
