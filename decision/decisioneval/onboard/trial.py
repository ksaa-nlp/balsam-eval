"""Run canary items through one adapter candidate and collect what came back."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from decisioneval.egress import EgressRefused
from decisioneval.onboard.canary import HELD_OUT, ORIGINAL, ROTATED, SEED_CHECK, VISIBLE, Answer, CanaryItem
from decisioneval.remote.bound_adapter import BoundAdapter, load_bound_adapter
from decisioneval.remote.contract import ADAPTER_ERROR, INVALID_CHOICE, MALFORMED
from decisioneval.remote.execute import MALFORMED_RESPONSE, RunAborted, score_question
from decisioneval.spec.transport import RedirectRefused
from decisioneval.transforms.adapter import content_hash
from decisioneval.transforms.worker import TransformWorker
from decisioneval.transport import ModelClient

URL_NOT_PUBLIC = "DECISION_URL_NOT_PUBLIC"
REDIRECT_REFUSED = "DECISION_REDIRECT_REFUSED"
ONBOARDING_FAILED = "DECISION_ONBOARDING_FAILED"
DELIVERY_UNCONFIRMED = "DECISION_DELIVERY_UNCONFIRMED"
STRUCTURE_REASONS = frozenset({ADAPTER_ERROR, INVALID_CHOICE, MALFORMED_RESPONSE})
SEED_CASES = tuple((item, ORIGINAL) for item in SEED_CHECK)
FULL_CASES = tuple((item, order) for item in VISIBLE + HELD_OUT for order in (ORIGINAL, ROTATED))
AUTH_REFUSED = frozenset({401, 403})


@dataclass
class Trial:
    answers: list[Answer] = field(default_factory=list)
    structure_errors: list[tuple[str, str]] = field(default_factory=list)  # (item id, message)
    mapped: list[dict[str, Any]] = field(default_factory=list)

    def errors(self, only: tuple[CanaryItem, ...] | None = None) -> tuple[str, ...]:
        allowed = None if only is None else {item.id for item in only}
        return tuple(message for item_id, message in self.structure_errors if allowed is None or item_id in allowed)

    @property
    def answered(self) -> int:
        return sum(answer.choice is not None for answer in self.answers)


def bind(url: str, model: str, auth: dict[str, Any], adapter: dict[str, Any]) -> BoundAdapter:
    """A candidate adapter block, hashed and loaded exactly as an eval would load it."""
    block = {key: value for key, value in adapter.items() if key not in ("version_id", "content_hash")}
    block["content_hash"] = content_hash(block, auth)
    return load_bound_adapter({"url": url, "model": model, "auth": auth, "adapter": block})


async def run_trial(
    bound: BoundAdapter,
    cases: tuple[tuple[CanaryItem, str], ...],
    *,
    key: str | None,
    client: ModelClient,
    worker: TransformWorker,
) -> Trial:
    """Ask every case. Raises `RunAborted` when the model can't be reached or keeps refusing."""
    gate = asyncio.Semaphore(bound.concurrency)

    async def one(item: CanaryItem, order: str):
        async with gate:
            scored, _ = await score_question(bound, item.as_question(order), key=key, client=client, worker=worker)
        return item, order, scored

    trial = Trial()
    for item, order, scored in await asyncio.gather(*(one(item, order) for item, order in cases)):
        raw = scored.raw_response
        reason, problem = raw.get("invalid_reason"), raw.get("distribution_problem")
        if reason in STRUCTURE_REASONS or problem == MALFORMED:
            detail = raw.get("error") or (f"mapped to {raw.get('mapped')!r}" if "mapped" in raw else "")
            trial.structure_errors.append((item.id, f"{item.id}/{order}: {reason or 'malformed vector'} {detail}".strip()))
        trial.answers.append(Answer(item=item, order=order, choice=scored.predicted_option_id))
        trial.mapped.append({"item": item.id, "order": order, "reply": raw.get("reply"), "mapped": raw.get("mapped")})
    return trial


def abort_code(exc: RunAborted) -> str:
    cause = exc.__cause__
    if isinstance(cause, EgressRefused):
        return URL_NOT_PUBLIC
    if isinstance(cause, RedirectRefused):
        return REDIRECT_REFUSED
    return ONBOARDING_FAILED
