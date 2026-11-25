"""Turn adapter decisions into per-item results.

The resolution rules are the scoring contract:

* the decision's option id must be one of the question's ids, exactly;
* gold matching is literal string equality on that option's text;
* an invalid decision is wrong and stays in the accuracy denominator.
"""

from __future__ import annotations

from decisioneval.models import DecisionResult, ItemResult, Question


def evaluate_question(question: Question, decision: DecisionResult) -> ItemResult:
    """Resolve a decision to one option and score it against the gold text."""
    choice = decision.predicted_option_id
    if decision.valid and choice is not None and choice in question.option_ids:
        predicted_option_id, predicted_answer, invalid = choice, question.options[question.option_ids.index(choice)], False
    else:
        predicted_option_id, predicted_answer, invalid = _clean(choice), None, True
    correct = not invalid and predicted_answer == question.gold_answer

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


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
