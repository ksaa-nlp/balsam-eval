"""Probe stage: the seeded protocols on the URL as given, then known versions for its host and path.

A step succeeds only when the model's replies pass the decision contract; a 2xx
alone never does. Nothing here calls an LLM.
"""

from __future__ import annotations

from typing import Any

from decisioneval.onboard.canary import SEED_CHECK, Verdict, behaviour_report, judge
from decisioneval.onboard.report import OnboardRequest, adapter_report, outcome, shape_of
from decisioneval.onboard.trial import AUTH_REFUSED, FULL_CASES, ONBOARDING_FAILED, SEED_CASES, abort_code, bind, run_trial
from decisioneval.remote.bound_adapter import AdapterConfigError
from decisioneval.remote.execute import RunAborted
from decisioneval.transforms.adapter import SEEDED, seed_adapter
from decisioneval.transforms.seeds import request_input, seed_transforms
from decisioneval.transforms.worker import TransformWorker
from decisioneval.transport import ModelClient

STAGE = "probe"


async def probe_stage(
    request: OnboardRequest, *, key: str | None, client: ModelClient, worker: TransformWorker
) -> dict[str, Any]:
    for name, seed_auth in SEEDED:
        if seed_auth["kind"] != request.auth.get("kind"):
            continue
        # The credential's own placement: a custom prefix must reach the model, and be what gets bound.
        auth = request.auth
        bound = bind(request.url, request.model, auth, seed_adapter(name))
        try:
            trial = await run_trial(bound, SEED_CASES, key=key, client=client, worker=worker)
        except RunAborted as exc:
            if exc.status in AUTH_REFUSED:
                return outcome(STAGE, "failed", error_code=ONBOARDING_FAILED, error_detail=f"the model refused the key: {exc}")
            if exc.status is not None:
                continue  # this protocol's request is not what the endpoint expects
            return outcome(STAGE, "failed", error_code=abort_code(exc), error_detail=str(exc))
        if not trial.structure_errors and trial.answered:
            seed_check = Verdict(
                "pass",
                correct_original=sum(a.chosen_text == a.item.gold_text for a in trial.answers),
                item_count=len(SEED_CHECK),
                behaviour_report=behaviour_report(trial.answers),
            ).as_validation("seed_check")
            return outcome(STAGE, "active", adapter=adapter_report(
                seed_adapter(name), auth, bound.content_hash, origin="seeded", seed_name=name,
            ), shape=shape_of(request.url, _sent_body(name, request, worker), trial.mapped[0]["reply"]),
                validations=[seed_check])

    for candidate in request.candidates:
        try:
            bound = bind(request.url, request.model, candidate.get("auth") or {}, candidate.get("adapter") or {})
            trial = await run_trial(bound, FULL_CASES, key=key, client=client, worker=worker)
        except (AdapterConfigError, RunAborted):
            continue
        verdict = judge(trial.answers, trial.errors())
        if verdict.verdict == "pass":
            return outcome(STAGE, "active", adapter=adapter_report(
                candidate["adapter"], candidate["auth"], bound.content_hash, origin="probe",
            ), validations=[verdict.as_validation()])
    return outcome(STAGE, "awaiting_curl")


def _sent_body(seed: str, request: OnboardRequest, worker: TransformWorker) -> Any:
    """The body this seed sent for the first seed-check item, for the shape's request keys."""
    question = SEED_CASES[0][0].as_question(SEED_CASES[0][1])
    data = request_input(model=request.model, instruction=question.instruction, input_text=question.input,
                         options=question.options, option_ids=question.option_ids)
    return worker.run(seed_transforms(seed)[0], data)
