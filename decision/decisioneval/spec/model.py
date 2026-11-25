"""Validate a decision-model response mapping, as the probe proposes it.

The validator's job is not only "is this well formed" but "can this spec invent
a number". Every value a spec contributes to a ``DecisionResult`` has to arrive
through :mod:`decisioneval.spec.paths`, so the validator rejects any unknown key
anywhere in the ``response`` block: a future ``transform``, ``scale`` or
``softmax`` key cannot be silently ignored, it fails the spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from decisioneval.spec.paths import PathError, parse

SPEC_VERSION = 1

CHOICE_KINDS = ("option_id", "option_text", "index0", "index1")
PROB_KEYINGS = ("option_id", "option_text", "index0", "index1", "order", "pairs")
BAD_SUM_POLICIES = ("unavailable", "reject")

class SpecError(ValueError):
    """The spec file is unusable. The message lists every problem found."""


@dataclass(frozen=True)
class ChoiceMap:
    path: str
    kind: str = "option_id"


@dataclass(frozen=True)
class ProbabilityMap:
    path: str
    keying: str = "option_id"
    label_path: str | None = None
    score_path: str | None = None
    label_kind: str = "option_id"


@dataclass(frozen=True)
class ConfidenceMap:
    path: str
    semantics: str = "unknown"


@dataclass(frozen=True)
class Condition:
    path: str
    equals: Any = None
    mode: str = "equals"  # equals | missing | empty


@dataclass(frozen=True)
class ResponseMap:
    choice: ChoiceMap
    envelope: str = ""
    probabilities: ProbabilityMap | None = None
    provider_confidence: ConfidenceMap | None = None
    invalid_when: tuple[Condition, ...] = ()
    identity: tuple[Condition, ...] = ()
    on_bad_sum: str = "unavailable"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    access_type: str
    version: str
    response: ResponseMap
    source_path: str = ""
    notes: str = ""
    provider_confidence_note: str = ""

def build_spec(document: Any, source_path: str = "") -> ModelSpec:
    """Validate an already-parsed spec document."""
    problems: list[str] = []
    spec = _build(document, source_path, problems)
    if problems:
        label = f"{source_path}: " if source_path else ""
        raise SpecError(label + "; ".join(problems))
    assert spec is not None
    return spec


def _build(document: Any, source_path: str, problems: list[str]) -> ModelSpec | None:
    if not isinstance(document, dict):
        problems.append("the spec is not a JSON object")
        return None
    _reject_unknown(document, {
        "spec_version", "name", "display_name", "access_type", "version",
        "response", "notes", "provider_confidence_note",
    }, "", problems)

    if document.get("spec_version") != SPEC_VERSION:
        problems.append(f"spec_version must be {SPEC_VERSION}")
    name = _text(document, "name", problems)
    access_type = str(document.get("access_type") or "")
    if access_type not in ("api", "local"):
        problems.append("access_type must be api or local")
    response = _response(document.get("response"), problems)
    if problems:
        return None
    assert response is not None
    return ModelSpec(
        name=name,
        display_name=str(document.get("display_name") or name),
        access_type=access_type,
        version=str(document.get("version") or "spec-1"),
        response=response,
        source_path=source_path,
        notes=str(document.get("notes") or ""),
        provider_confidence_note=str(document.get("provider_confidence_note") or ""),
    )


def _response(block: Any, problems: list[str]) -> ResponseMap | None:
    if not isinstance(block, dict):
        problems.append("response is required and must be an object")
        return None
    _reject_unknown(block, {
        "envelope", "choice", "probabilities", "provider_confidence",
        "invalid_when", "identity", "on_bad_sum",
    }, "response", problems)

    envelope = str(block.get("envelope") or "")
    if envelope:
        _check_path(envelope, "response.envelope", problems)

    choice_block = block.get("choice")
    choice: ChoiceMap | None = None
    if not isinstance(choice_block, dict):
        problems.append("response.choice is required and must be an object")
    else:
        _reject_unknown(choice_block, {"path", "kind"}, "response.choice", problems)
        path = str(choice_block.get("path") or "")
        _check_path(path, "response.choice.path", problems)
        kind = str(choice_block.get("kind") or "option_id")
        if kind not in CHOICE_KINDS:
            problems.append(f"response.choice.kind must be one of {CHOICE_KINDS}")
        choice = ChoiceMap(path=path, kind=kind)

    probabilities = _probabilities(block.get("probabilities"), problems)
    confidence = _confidence(block.get("provider_confidence"), problems)
    invalid_when = _conditions(block.get("invalid_when"), "response.invalid_when", problems)
    identity = _conditions(block.get("identity"), "response.identity", problems)

    on_bad_sum = str(block.get("on_bad_sum") or "unavailable")
    if on_bad_sum not in BAD_SUM_POLICIES:
        problems.append(f"response.on_bad_sum must be one of {BAD_SUM_POLICIES}")

    if problems or choice is None:
        return None
    return ResponseMap(
        choice=choice,
        envelope=envelope,
        probabilities=probabilities,
        provider_confidence=confidence,
        invalid_when=invalid_when,
        identity=identity,
        on_bad_sum=on_bad_sum,
    )


def _probabilities(block: Any, problems: list[str]) -> ProbabilityMap | None:
    if block is None:
        return None
    if not isinstance(block, dict):
        problems.append("response.probabilities is not an object")
        return None
    _reject_unknown(
        block,
        {"path", "keying", "label_path", "score_path", "label_kind"},
        "response.probabilities",
        problems,
    )
    path = str(block.get("path") or "")
    _check_path(path, "response.probabilities.path", problems)
    keying = str(block.get("keying") or "option_id")
    if keying not in PROB_KEYINGS:
        problems.append(f"response.probabilities.keying must be one of {PROB_KEYINGS}")
    label_path = block.get("label_path")
    score_path = block.get("score_path")
    if keying == "pairs":
        if not str(label_path or "").strip() or not str(score_path or "").strip():
            problems.append("keying=pairs requires label_path and score_path")
        else:
            _check_path(str(label_path), "response.probabilities.label_path", problems)
            _check_path(str(score_path), "response.probabilities.score_path", problems)
    label_kind = str(block.get("label_kind") or "option_id")
    if label_kind not in CHOICE_KINDS:
        problems.append(f"response.probabilities.label_kind must be one of {CHOICE_KINDS}")
    return ProbabilityMap(
        path=path,
        keying=keying,
        label_path=str(label_path) if label_path else None,
        score_path=str(score_path) if score_path else None,
        label_kind=label_kind,
    )


def _confidence(block: Any, problems: list[str]) -> ConfidenceMap | None:
    if block is None:
        return None
    if not isinstance(block, dict):
        problems.append("response.provider_confidence is not an object")
        return None
    _reject_unknown(block, {"path", "semantics"}, "response.provider_confidence", problems)
    path = str(block.get("path") or "")
    _check_path(path, "response.provider_confidence.path", problems)
    return ConfidenceMap(path=path, semantics=str(block.get("semantics") or "unknown"))


def _conditions(block: Any, label: str, problems: list[str]) -> tuple[Condition, ...]:
    if block is None:
        return ()
    if not isinstance(block, list):
        problems.append(f"{label} is not a list")
        return ()
    rules: list[Condition] = []
    for index, entry in enumerate(block):
        where = f"{label}[{index}]"
        if not isinstance(entry, dict):
            problems.append(f"{where} is not an object")
            continue
        _reject_unknown(entry, {"path", "equals", "mode"}, where, problems)
        path = str(entry.get("path") or "")
        _check_path(path, f"{where}.path", problems)
        mode = str(entry.get("mode") or ("equals" if "equals" in entry else "missing"))
        if mode not in ("equals", "missing", "empty"):
            problems.append(f"{where}.mode must be equals, missing or empty")
        rules.append(Condition(path=path, equals=entry.get("equals"), mode=mode))
    return tuple(rules)


def _text(document: dict[str, Any], key: str, problems: list[str]) -> str:
    value = str(document.get(key) or "").strip()
    if not value:
        problems.append(f"{key} is required")
    return value


def _check_path(path: str, label: str, problems: list[str]) -> None:
    try:
        parse(path)
    except PathError as exc:
        problems.append(f"{label}: {exc}")


def _reject_unknown(block: dict[str, Any], allowed: set[str], label: str, problems: list[str]) -> None:
    """Any key we do not understand is an error, never an ignored field.

    A spec that tried to smuggle a computation in (``softmax``, ``scale``,
    ``value``) fails here instead of being quietly dropped.
    """
    for key in sorted(set(block) - allowed):
        where = f"{label}.{key}" if label else key
        problems.append(f"unknown key: {where}")
