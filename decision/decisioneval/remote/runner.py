"""Run one platform job end to end through its bound adapter version, then report it."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
import traceback
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from decisioneval.datasets import load_paths
from decisioneval.metrics.summary import build_summary
from decisioneval.models import ItemResult, Question
from decisioneval.remote.checkpoint import Checkpoint, item_key
from decisioneval.remote.bound_adapter import API_KEY_ENV, BoundAdapter, load_bound_adapter, mapping_origin, require_key
from decisioneval.remote.config import LOCAL_TASKS_DIR, ConfigError, RemoteConfig, load_config
from decisioneval.remote.execute import CircuitBroken, RunAborted, execute
from decisioneval.remote.finalize import FinalizeError, finalize
from decisioneval.remote.pools import PreparedPools, prepare_pools
from decisioneval.remote.results import META_FILE, RUN_FILE, meta_document, pool_results
from decisioneval.remote.storage import GcsStore, LocalStore, ObjectStore, StorageError, join
from decisioneval.store import build_run, new_run_id, redact_secrets, scrub_key, scrub_logs
from decisioneval.transforms.worker import TransformWorker
from decisioneval.transport import ModelClient

log = logging.getLogger("decisioneval.remote")

ADAPTER_BROKEN = "DECISION_ADAPTER_BROKEN"
ALLOW_PRIVATE_ENV = "DECISION_EGRESS_ALLOW_PRIVATE"
IMAGE_DIGEST_ENV = "RUNNER_IMAGE_DIGEST"


@dataclass(frozen=True)
class JobReport:
    succeeded: bool
    degraded: bool = False
    error: str | None = None
    files: tuple[str, ...] = ()
    code: str | None = None
    content_hash: str | None = None


def failed(reason: str, files: tuple[str, ...] = (), code: str | None = None) -> JobReport:
    return JobReport(succeeded=False, error=reason, files=files, code=code)


def run_job(
    config: RemoteConfig,
    store: ObjectStore,
    workdir: Path,
    *,
    environ: Mapping[str, str],
) -> JobReport:
    """Evaluate the job. Raises only on configuration and storage failures and bugs."""
    bound = load_bound_adapter(config.request_config)
    key = require_key(bound.auth, environ)
    pools = prepare_pools(store, _sources(config, store), workdir / "datasets")
    log.info("%d pool(s) to score, %d skipped", len(pools.pools), len(pools.skipped))
    for skip in pools.skipped:
        log.info("skipped pool %s: %s", skip.source, skip.reason)
    if not pools.pools:
        return failed(_no_pools_reason(pools))

    report = load_paths([str(pools.directory)])
    questions = [question for dataset in report.datasets for question in dataset.questions]
    duplicates = _duplicate_keys(questions)
    if duplicates:
        return failed(f"pool items must have unique ids within a pool; repeated: {', '.join(duplicates[:10])}")
    log.info("adapter %s (%s): %d question(s)", bound.content_hash[:12], mapping_origin(bound), len(questions))
    # Private addresses are reachable only in a local run that asks for it, never in a build.
    allow_private = not config.uses_gcs and environ.get(ALLOW_PRIVATE_ENV) == "1"
    checkpoint = Checkpoint(
        store, config.results_path, bound.content_hash, target=f"{bound.url}\n{bound.model}", key=key
    )
    resumed = {item_key(record): ItemResult.from_dict(record) for record in checkpoint.load()}
    todo = [question for question in questions if _question_key(question) not in resumed]
    if resumed:
        log.info("resuming: %d item(s) already scored by this adapter version", len(resumed))
    meta = {
        "bound": bound,
        "pools": pools,
        "image_digest": environ.get(IMAGE_DIGEST_ENV) or None,
        "resumed_items": len(resumed),
    }
    try:
        scored = asyncio.run(_score(bound, todo, key, allow_private, config.deadline_s, checkpoint))
    except CircuitBroken as exc:
        files = _write(config, store, {META_FILE: meta_document(**meta, items=[], state="circuit_broken")}, key)
        return failed(str(exc), files, code=ADAPTER_BROKEN)
    except RunAborted as exc:
        files = _write(config, store, {META_FILE: meta_document(**meta, items=[], state="aborted")}, key)
        return failed(str(exc), files)

    by_key = {**resumed, **{_question_key_of(item): item for item in scored}}
    items = [by_key[_question_key(question)] for question in questions]
    summary = build_summary(items)
    run = build_run(
        run_id=new_run_id(),
        run_type="full",
        model=config.model,
        model_version=config.model_version,
        access_type="api",
        spec_name=f"adapter:{bound.content_hash}",
        report=report,
        summary=summary,
        items=items,
        mapping_origin=mapping_origin(bound),
    )
    run["adapter"] = {
        "version_id": bound.version_id,
        "content_hash": bound.content_hash,
        "response_transform": bound.response_transform,
    }
    documents: dict[str, Any] = {
        META_FILE: meta_document(**meta, items=run["items"], state="complete"),
        RUN_FILE: run,
        **pool_results(pools, run, calibrated=not bound.derived_probabilities),
    }
    files = _write(config, store, documents, key)
    return JobReport(succeeded=True, degraded=bound.degraded, files=files, content_hash=bound.content_hash)


async def _score(
    bound: BoundAdapter,
    questions: list[Question],
    key: str | None,
    allow_private: bool,
    deadline_s: float,
    checkpoint: Checkpoint,
) -> list[ItemResult]:
    client = ModelClient(
        concurrency=bound.concurrency, timeout_s=bound.timeout_s, retry=bound.retry, allow_private=allow_private
    )
    try:
        async with asyncio.timeout(deadline_s), client:
            with TransformWorker() as worker:
                # An item the model refused is asked again on resume, since the cause may since be fixed.
                return await execute(
                    bound, questions, key=key, client=client, worker=worker,
                    on_item=lambda item: None if item.raw_response.get("http_status") else checkpoint.add(item.to_dict()),
                )
    except TimeoutError as exc:
        raise RunAborted(f"the run passed its {deadline_s:.0f}s deadline") from exc
    finally:
        checkpoint.flush()


def _duplicate_keys(questions: list[Question]) -> list[str]:
    seen: set[tuple[str, str]] = set()
    repeated: list[str] = []
    for question in questions:
        key = _question_key(question)
        if key in seen:
            repeated.append(f"{key[0]}/{key[1]}")
        seen.add(key)
    return repeated


def _question_key(question: Question) -> tuple[str, str]:
    return (question.dataset_name, str(question.id))


def _question_key_of(item: ItemResult) -> tuple[str, str]:
    return (item.dataset, str(item.item_id))


def _write(
    config: RemoteConfig, store: ObjectStore, documents: dict[str, Any], key: str | None
) -> tuple[str, ...]:
    for name, document in documents.items():
        store.write_bytes(join(config.results_path, name), _encode(scrub_key(document, key)))
    return tuple(sorted(documents))


def _sources(config: RemoteConfig, store: ObjectStore) -> tuple[str, ...]:
    if config.pool_files:
        return config.pool_files
    return tuple(path for path in store.list(LOCAL_TASKS_DIR) if path.endswith(".json"))


def _no_pools_reason(pools: PreparedPools) -> str:
    if not pools.skipped:
        return "no pool files were given"
    details = "; ".join(f"{skip.source}: {skip.reason}" for skip in pools.skipped[:10])
    return f"no scorable MCQ pool files ({len(pools.skipped)} skipped): {details}"


def _encode(document: Any) -> bytes:
    return (json.dumps(redact_secrets(document), ensure_ascii=False, indent=2) + "\n").encode()


def open_store(config: RemoteConfig) -> ObjectStore:
    return GcsStore(config.bucket) if config.bucket else LocalStore(config.local_root)


def main(environ: Mapping[str, str] | None = None) -> int:
    """Entry point for `python -m decisioneval.remote`. Returns the exit code."""
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    environ = dict(os.environ if environ is None else environ)
    scrub_logs((environ.get(API_KEY_ENV) or "").strip() or None)
    try:
        config = load_config(environ)
    except ConfigError as exc:
        log.error("configuration error: %s", exc)
        _report_unconfigured(environ, f"runner configuration error: {exc}")
        return 2

    log.info("job %s: model %s", config.job_id, config.model)
    with tempfile.TemporaryDirectory(prefix="decisioneval-") as tmp:
        report = _run_safely(config, Path(tmp), environ)
    # Error text can quote a URL that carries the key (query auth) or a reply that echoes it.
    report = replace(report, error=scrub_key(report.error, (environ.get(API_KEY_ENV) or "").strip() or None))

    log.info(
        "job %s: %s%s", config.job_id,
        "succeeded" if report.succeeded else f"failed: {report.error}",
        " (degraded: calibration unavailable)" if report.degraded else "",
    )
    if config.reports_outcome:
        try:
            finalize(
                str(config.api_host), str(config.finalize_token), config.job_id,
                succeeded=report.succeeded, error=report.error, code=report.code,
                content_hash=report.content_hash,
            )
        except FinalizeError as exc:
            log.error("%s", exc)
            return 3
    return 0 if report.succeeded else 1


def _run_safely(config: RemoteConfig, workdir: Path, environ: Mapping[str, str]) -> JobReport:
    try:
        return run_job(config, open_store(config), workdir, environ=environ)
    except ConfigError as exc:
        return failed(f"runner configuration error: {exc}")
    except StorageError as exc:
        return failed(f"storage error: {exc}")
    except Exception as exc:  # the job must still be reported, whatever happened
        log.error("runner crashed:\n%s", traceback.format_exc())
        return failed(f"runner crashed: {type(exc).__name__}: {exc}")


def _report_unconfigured(environ: Mapping[str, str], reason: str) -> None:
    """Best-effort finalize when the config itself is broken."""
    api_host, token, job_id = (
        (environ.get(name) or "").strip() for name in ("API_HOST", "FINALIZE_TOKEN", "JOB_ID")
    )
    if not (api_host and token and job_id):
        return
    try:
        finalize(api_host, token, job_id, succeeded=False, error=reason)
    except FinalizeError as exc:
        log.error("%s", exc)
