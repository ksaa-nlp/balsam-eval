"""Deterministic structural probe: propose a mapping from sample responses.

The probe walks every path in a handful of real raw responses and asks
structural questions, never semantic ones:

* which dict has a key set that lands exactly on the option ids and sums to 1?
* which list has one entry per option and sums to 1?
* which list of objects carries a label that is an option id plus a number?
* which scalar is exactly an option id, an option text, or a valid index?
* which scalar in [0, 1] sits next to the chosen option and is not in the
  distribution? (a provider confidence candidate, semantics unknown)

A candidate has to hold for *every* sample, which kills coincidences. When more
than one candidate survives, the probe reports the ambiguity instead of
picking, because a wrong guess here silently rescores a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from decisioneval.metrics.probability import PROBABILITY_SUM_TOLERANCE
from decisioneval.spec.mapping import _number, _resolve_label
from decisioneval.spec.paths import MISSING, get

CONFIDENCE_HINTS = ("confidence", "certainty", "score", "prob")


@dataclass(frozen=True)
class Sample:
    """One real raw response plus the question it answered."""

    raw: Any
    option_ids: tuple[str, ...]
    options: tuple[str, ...]
    chosen_option_id: str | None = None


@dataclass
class Candidate:
    path: str
    kind: str
    detail: dict[str, Any] = field(default_factory=dict)

    def key(self) -> tuple[str, str, str]:
        return (self.path, self.kind, str(sorted(self.detail.items())))


@dataclass
class ProbeReport:
    choice: list[Candidate] = field(default_factory=list)
    probabilities: list[Candidate] = field(default_factory=list)
    provider_confidence: list[Candidate] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ambiguous(self) -> bool:
        return len(self.choice) > 1 or len(self.probabilities) > 1

    @property
    def resolved(self) -> bool:
        return len(self.choice) == 1


def walk(node: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Every path in `node`, parents before children."""
    found: list[tuple[str, Any]] = [(prefix, node)] if prefix else []
    if isinstance(node, dict):
        for key, value in node.items():
            if "." in str(key) or "[" in str(key) or "]" in str(key):
                continue  # not addressable in the path grammar
            found.extend(walk(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(walk(value, f"{prefix}[{index}]"))
    return found


def probe(samples: list[Sample]) -> ProbeReport:
    """Propose choice / probability / confidence mappings that hold for all samples."""
    report = ProbeReport()
    if not samples:
        report.notes.append("there are no samples")
        return report

    report.probabilities = _agreed(samples, _probability_candidates)
    # A label sitting inside the distribution names a row of the distribution,
    # not the model's pick, so it is not offered as a choice path. Without this
    # a ranked list of {label, score} always reads as ambiguous.
    inside = tuple(candidate.path for candidate in report.probabilities)
    report.choice = [
        candidate
        for candidate in _agreed(samples, _choice_candidates)
        if not any(_under(candidate.path, prefix) for prefix in inside)
    ]
    report.choice = _prune_by_argmax(report, samples)
    taken = {candidate.path for candidate in report.probabilities}
    report.provider_confidence = [
        candidate
        for candidate in _agreed(samples, _confidence_candidates)
        if candidate.path not in taken
    ]

    if not report.choice:
        report.notes.append("no value matches an option id, an option text, or a valid option index")
    if len(report.choice) > 1:
        report.notes.append(
            "more than one valid choice reading: "
            + ", ".join(f"{c.path} as {c.detail.get('kind')}" for c in report.choice)
        )
    if not report.probabilities:
        report.notes.append(
            "no complete distribution summing to 1. if the model returns logits or "
            "unnormalised scores, leave probabilities out of the spec: probability "
            "metrics become null and nothing is rescaled"
        )
    if len(report.probabilities) > 1:
        report.notes.append(
            "more than one valid distribution: " + ", ".join(c.path for c in report.probabilities)
        )
    return report


def _under(path: str, prefix: str) -> bool:
    """True when `path` is `prefix` or sits inside it, on a segment boundary.

    A plain ``startswith`` would read ``out.pick`` as living inside ``out.p``.
    """
    return path == prefix or path.startswith(f"{prefix}.") or path.startswith(f"{prefix}[")


def _prune_by_argmax(report: ProbeReport, samples: list[Sample]) -> list[Candidate]:
    """Let the model's own distribution settle which choice reading is right.

    A scalar like `picked_rank: 1` reads as index0 or index1 and both are
    in range, so the payload alone cannot say which. The distribution can: the
    option it puts the most probability on is the option the model picked. This
    uses nothing but numbers the model returned, and it is what makes a
    one-based index resolvable without a human or a label.

    Candidates are kept when they agree with the argmax on every sample where
    the argmax is unique. With no distribution, or on ties, nothing is pruned.
    """
    if len(report.choice) < 2 or len(report.probabilities) != 1:
        return report.choice
    rules = report.probabilities[0]
    expected: list[str] = []
    for sample in samples:
        peak = _argmax_option(rules, sample)
        if peak is None:
            return report.choice
        expected.append(peak)

    kept: list[Candidate] = []
    for candidate in report.choice:
        agrees = True
        for sample, option_id in zip(samples, expected, strict=True):
            value = get(sample.raw, candidate.path)
            if value is MISSING:
                agrees = False
                break
            resolved = _resolve_label(
                value, candidate.detail.get("kind", "option_id"), sample.option_ids, sample.options
            )
            if resolved != option_id:
                agrees = False
                break
        if agrees:
            kept.append(candidate)
    return kept or report.choice


def _argmax_option(rules: Candidate, sample: Sample) -> str | None:
    """The single option carrying the highest probability, or None on a tie."""
    from decisioneval.spec.mapping import map_response
    from decisioneval.spec.model import build_spec

    try:
        spec = build_spec(
            {
                "spec_version": 1,
                "name": "probe-argmax",
                "access_type": "api",
                "response": {
                    "choice": {"path": rules.path},
                    "probabilities": {"path": rules.path, **rules.detail},
                },
            }
        )
    except Exception:
        return None
    mapped = map_response(spec, sample.raw, sample.option_ids, sample.options)
    if not mapped.probabilities:
        return None
    peak = max(mapped.probabilities.values())
    winners = [key for key, value in mapped.probabilities.items() if value == peak]
    return winners[0] if len(winners) == 1 else None


def _agreed(
    samples: list[Sample],
    finder: Any,
) -> list[Candidate]:
    """Keep only candidates that every sample agrees on."""
    shared: dict[tuple[str, str, str], Candidate] | None = None
    for sample in samples:
        found = {candidate.key(): candidate for candidate in finder(sample)}
        shared = found if shared is None else {k: v for k, v in shared.items() if k in found}
    return sorted((shared or {}).values(), key=lambda candidate: candidate.path)


def _probability_candidates(sample: Sample) -> list[Candidate]:
    found: list[Candidate] = []
    count = len(sample.option_ids)
    for path, node in walk(sample.raw):
        if isinstance(node, dict):
            for keying in ("option_id", "option_text", "index0", "index1"):
                if _dict_fits(node, keying, sample):
                    found.append(Candidate(path, "probabilities", {"keying": keying}))
        elif isinstance(node, list) and len(node) == count:
            numbers = [_number(value) for value in node]
            if all(value is not None for value in numbers) and _sums(numbers):
                found.append(Candidate(path, "probabilities", {"keying": "order"}))
            found.extend(_pair_candidates(path, node, sample))
    return found


def _dict_fits(node: dict[str, Any], keying: str, sample: Sample) -> bool:
    if len(node) != len(sample.option_ids):
        return False
    mapped: dict[str, float] = {}
    for key, value in node.items():
        option_id = _resolve_label(key, keying, sample.option_ids, sample.options)
        number = _number(value)
        if option_id is None or number is None or option_id in mapped:
            return False
        mapped[option_id] = number
    return set(mapped) == set(sample.option_ids) and _sums(list(mapped.values()))


def _pair_candidates(path: str, node: list[Any], sample: Sample) -> list[Candidate]:
    """A list of objects, each with a label field and a numeric field."""
    if not all(isinstance(entry, dict) for entry in node):
        return []
    first = node[0]
    label_keys = [key for key in first if isinstance(first[key], str)]
    score_keys = [key for key in first if _number(first[key]) is not None]
    found: list[Candidate] = []
    for label_key in label_keys:
        for label_kind in ("option_id", "option_text"):
            labels = [
                _resolve_label(entry.get(label_key), label_kind, sample.option_ids, sample.options)
                for entry in node
            ]
            if None in labels or len(set(labels)) != len(sample.option_ids):
                continue
            for score_key in score_keys:
                numbers = [_number(entry.get(score_key)) for entry in node]
                if any(value is None for value in numbers) or not _sums(numbers):
                    continue
                found.append(
                    Candidate(
                        path,
                        "probabilities",
                        {
                            "keying": "pairs",
                            "label_path": label_key,
                            "score_path": score_key,
                            "label_kind": label_kind,
                        },
                    )
                )
    return found


def _choice_candidates(sample: Sample) -> list[Candidate]:
    found: list[Candidate] = []
    for path, node in walk(sample.raw):
        if isinstance(node, (dict, list)):
            continue
        for kind in ("option_id", "option_text", "index0", "index1"):
            option_id = _resolve_label(node, kind, sample.option_ids, sample.options)
            if option_id is None:
                continue
            if sample.chosen_option_id is not None and option_id != sample.chosen_option_id:
                continue
            found.append(Candidate(path, "choice", {"kind": kind}))
    return found


def _confidence_candidates(sample: Sample) -> list[Candidate]:
    found: list[Candidate] = []
    for path, node in walk(sample.raw):
        number = _number(node)
        if number is None or not 0.0 <= number <= 1.0:
            continue
        leaf = path.rsplit(".", 1)[-1].lower()
        if any(hint in leaf for hint in CONFIDENCE_HINTS):
            found.append(Candidate(path, "provider_confidence", {"semantics": "unknown"}))
    return found


def _sums(numbers: list[float | None]) -> bool:
    values = [value for value in numbers if value is not None]
    if len(values) != len(numbers) or not values:
        return False
    if any(value < 0.0 or value > 1.0 for value in values):
        return False
    return abs(sum(values) - 1.0) <= PROBABILITY_SUM_TOLERANCE


def to_spec(
    name: str,
    report: ProbeReport,
    access_type: str = "api",
    display_name: str = "",
    version: str = "spec-1",
) -> dict[str, Any]:
    """Build a spec document from an unambiguous probe report."""
    # An ambiguous distribution must not quietly become "no distribution":
    # that would downgrade a calibrated model to null probability metrics.
    if not report.resolved or report.ambiguous:
        raise ValueError("the probe is ambiguous; no spec is built from an ambiguous result: " + "; ".join(report.notes))
    choice = report.choice[0]
    response: dict[str, Any] = {
        "choice": {"path": choice.path, "kind": choice.detail.get("kind", "option_id")},
    }
    if len(report.probabilities) == 1:
        candidate = report.probabilities[0]
        response["probabilities"] = {"path": candidate.path, **candidate.detail}
    if len(report.provider_confidence) == 1:
        candidate = report.provider_confidence[0]
        response["provider_confidence"] = {
            "path": candidate.path,
            "semantics": "unknown",
        }
    return {
        "spec_version": 1,
        "name": name,
        "display_name": display_name or name,
        "access_type": access_type,
        "version": version,
        "response": response,
    }
