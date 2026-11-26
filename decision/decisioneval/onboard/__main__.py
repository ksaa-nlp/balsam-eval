"""`python -m decisioneval.onboard`: run one onboarding stage on Cloud Build and report it.

Environment (see README.md): ONBOARDING_RUN_ID, API_HOST, ONBOARD_TOKEN,
ONBOARD_STAGE (probe | synthesize), ONBOARD_REQUEST_B64, API_KEY, and for synthesize
DECISION_SPEC_LLM_API_KEY with optional DECISION_SPEC_LLM_PROVIDER, _MODEL and _BASE_URL.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from decisioneval.onboard.hierarchy import probe_stage
from decisioneval.onboard.llm import ENV_NAMES as LLM_ENV, LlmConfigError, provider_from_env
from decisioneval.onboard.report import (
    STAGES,
    OnboardConfigError,
    OnboardRequest,
    ReportError,
    load_request,
    outcome,
    post_report,
)
from decisioneval.onboard.synthesize import synthesize_stage
from decisioneval.onboard.trial import ONBOARDING_FAILED
from decisioneval.store import scrub_logs
from decisioneval.transforms.worker import TransformWorker
from decisioneval.transport import ModelClient, RetryPolicy

log = logging.getLogger("decisioneval.onboard")


async def run_stage(
    stage: str,
    request: OnboardRequest,
    *,
    key: str | None,
    llm: Any = None,
    progress: Any = lambda report: None,
    allow_private: bool = False,
) -> dict[str, Any]:
    client = ModelClient(concurrency=4, timeout_s=60.0, retry=RetryPolicy(attempts=3, backoff_s=1.0),
                         allow_private=allow_private)
    async with client:
        with TransformWorker() as worker:
            if stage == "probe":
                return await probe_stage(request, key=key, client=client, worker=worker)
            return await synthesize_stage(request, key=key, client=client, worker=worker, llm=llm, progress=progress)


def main(environ: Mapping[str, str] | None = None) -> int:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if environ is None:
        # A local .env fills gaps only; Cloud Build's env always wins.
        load_dotenv(Path.cwd() / ".env", override=False)
    environ = dict(os.environ if environ is None else environ)
    run_id, api_host, token = (environ.get(name, "").strip() for name in ("ONBOARDING_RUN_ID", "API_HOST", "ONBOARD_TOKEN"))
    stage = environ.get("ONBOARD_STAGE", "").strip()
    key = (environ.get("API_KEY") or "").strip() or None
    scrub_logs(key)
    if not (run_id and api_host and token):
        log.error("ONBOARDING_RUN_ID, API_HOST and ONBOARD_TOKEN are required")
        return 2

    def send(report: dict[str, Any]) -> None:
        post_report(api_host, token, run_id, report, key=key)

    def progress(report: dict[str, Any]) -> None:
        try:
            send(report)
        except ReportError as exc:  # progress is informational; the final report still decides the run
            log.warning("progress report not delivered: %s", exc)

    try:
        if stage not in STAGES:
            raise OnboardConfigError(f"ONBOARD_STAGE must be one of {', '.join(STAGES)}")
        request = load_request(environ.get("ONBOARD_REQUEST_B64", ""))
        # The mapping LLM is only ever handed to the synthesize stage.
        llm = provider_from_env({name: environ.get(name, "") for name in LLM_ENV}) if stage == "synthesize" else None
        report = asyncio.run(run_stage(stage, request, key=key, llm=llm, progress=progress))
    except (OnboardConfigError, LlmConfigError) as exc:
        report = outcome(stage or "probe", "failed", error_code=ONBOARDING_FAILED, error_detail=f"configuration error: {exc}")
    except Exception as exc:  # the run must still be reported, whatever happened
        log.error("onboarding crashed:\n%s", traceback.format_exc())
        report = outcome(stage or "probe", "failed", error_code=ONBOARDING_FAILED,
                         error_detail=f"onboarding crashed: {type(exc).__name__}: {exc}")
    log.info("onboarding run %s %s: %s %s", run_id, report["stage"], report["outcome"], report.get("error_code") or "")
    try:
        send(report)
    except ReportError as exc:
        log.error("%s", exc)
        return 3
    return 0 if report["outcome"] != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())
