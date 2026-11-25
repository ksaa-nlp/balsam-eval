"""Turn adapter decisions into per-item results.

The resolution rules are the scoring contract:

* an explicit option id that exists on the question is authoritative, and its
  text is only checked to confirm that option, so a duplicated option text
  does not move the choice to an earlier copy;
* without an id, the text must match exactly one option;
* gold matching is literal string equality after `strip()`;
* an invalid decision is wrong and stays in the accuracy denominator.
"""

from __future__ import annotations

from decisioneval.models import DecisionResult, ItemResult, Question


def evaluate_question(question: Question, decision: DecisionResult) -> ItemResult:
    """Resolve a decision to one option and score it against the gold text."""
    canonical_id, canonical_answer, invalid = _resolve(question, decision)
    if invalid:
        predicted_option_id = _clean(decision.predicted_option_id)
        predicted_answer = _clean(decision.predicted_answer)
        correct = False
    else:
        predicted_option_id = canonical_id
        predicted_answer = canonical_answer
        correct = canonical_answer == question.gold_answer

    probabilities = None
    if decision.probabilities is not None:
        probabilities = dict(decision.probabilities)

    return ItemResult(
        item_id=question.id,
        dataset=question.dataset_name,
        category=question.category,
        task=question.task,
        options=question.options,
        option_ids=question.option_ids,
        gold_answer=question.gold_answer,
        gold_option_id=question.gold_option_id,
        predicted_option_id=predicted_option_id,
        predicted_answer=predicted_answer,
        probabilities=probabilities,
        confidence=decision.confidence,
        correct=correct,
        invalid=invalid,
        latency_ms=decision.latency_ms,
        usage=decision.usage,
        cost=decision.cost,
        raw_response=decision.raw_response,
        model_requested=decision.model_requested,
        model_resolved=decision.model_resolved,
        attempts=decision.attempts,
        retried=decision.retried,
        elapsed_ms=decision.elapsed_ms,
    )


def _resolve(
    question: Question, decision: DecisionResult
) -> tuple[str | None, str | None, bool]:
    if not decision.valid:
        return None, None, True

    raw_id = _clean(decision.predicted_option_id)
    raw_answer = _clean(decision.predicted_answer)
    if raw_id is None and raw_answer is None:
        return None, None, True

    id_match = _match_option_id(question, raw_id) if raw_id is not None else None
    if raw_id is not None and id_match is None:
        return None, None, True

    if id_match is not None:
        answer = question.options[question.option_ids.index(id_match)]
        if raw_answer is not None and answer != raw_answer:
            return None, None, True
        return id_match, answer, False

    matched_ids = _match_answer_ids(question, raw_answer or "")
    if len(matched_ids) != 1:
        return None, None, True
    chosen = matched_ids[0]
    answer = question.options[question.option_ids.index(chosen)]
    return chosen, answer, False


def _match_option_id(question: Question, option_id: str) -> str | None:
    lookup = {existing.upper(): existing for existing in question.option_ids}
    return lookup.get(option_id.upper())


def _match_answer_ids(question: Question, answer: str) -> tuple[str, ...]:
    return tuple(
        option_id
        for option_id, option_text in zip(question.option_ids, question.options, strict=True)
        if option_text == answer
    )


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
