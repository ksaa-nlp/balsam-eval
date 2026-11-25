"""Runner end to end: pools in storage -> bound adapter -> result files and finalize."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from decisioneval.audit import audit_document
from decisioneval.remote import runner
from decisioneval.remote.config import ConfigError, load_config
from decisioneval.remote.results import META_FILE, POOL_METRICS, RUN_FILE, result_filename
from decisioneval.remote.storage import LocalStore
from decisioneval.transforms.adapter import content_hash, seed_adapter
from tests.conftest import OPTIONS_3, OPTIONS_4, mcq_document

POOL_A = "dev/ns/pools/cat-7/three.json"
POOL_B = "dev/ns/pools/cat-7/four.json"
POOL_GENERATIVE = "dev/ns/pools/cat-7/essay.json"
RESULTS = "dev/ns/evaluations/1/5"
LOCAL = {"API_KEY": "secret-key", "DECISION_EGRESS_ALLOW_PRIVATE": "1"}


def platform_pool(name: str, options: list[str], with_item_ids: bool) -> dict:
    """A pool file as the platform writes it: title-case header, flat rows, ids."""
    document = mcq_document(name, options)
    rows = document["data"]["test"]
    if with_item_ids:
        rows = [{**row, "dataset_item_id": 1000 + index} for index, row in enumerate(rows)]
    return {"Type Of Result": "MCQ", "category": 7, "task": 21, "data": rows}


@pytest.fixture
def bucket(tmp_path: Path) -> LocalStore:
    store = LocalStore(tmp_path / "bucket")
    store.write_bytes(POOL_A, json.dumps(platform_pool("a", OPTIONS_3, True)).encode())
    store.write_bytes(POOL_B, json.dumps({"json": platform_pool("b", OPTIONS_4, False)}).encode())
    store.write_bytes(
        POOL_GENERATIVE,
        json.dumps({"Type of result": "Generative", "data": {"test": [{"id": 1}]}}).encode(),
    )
    return store


# -- a model, and the backend's finalize, over real HTTP ------------------


class _Model(BaseHTTPRequestHandler):
    mode = "system_one"
    calls: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        if self.path.startswith("/evaluation-jobs/"):
            type(self).calls.append((self.path, self.headers["Authorization"], body))
            return self._send(200, {"ok": True})
        type(self).calls.append((self.path, self.headers.get("Authorization"), None))
        criteria = body["questions"]["answer"]["criteria"]
        ids = list(criteria)
        pick = ids[len(body["state"]) % len(ids)]
        vector = {key: (0.7 if key == pick else round(0.3 / (len(ids) - 1), 6)) for key in ids}
        vector[pick] = round(1 - sum(v for k, v in vector.items() if k != pick), 6)
        if self.mode == "down":
            return self._send(503, {"error": "down"})
        if self.mode == "refuse_one" and len(type(self).calls) == 2:
            return self._send(413, {"error": "too long"})
        if self.mode == "slow":
            time.sleep(0.5)
        if self.mode == "malformed":
            vector = {key: 0.9 for key in ids}
        if self.mode == "refusal":
            return self._send(200, {"answers": {"answer": {"type": "refusal"}}})
        self._send(200, {"answers": {"answer": {"type": "choice", "choice": pick, "probabilities": vector}}})

    def _send(self, status, payload):
        encoded = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    _Model.mode = "system_one"
    _Model.calls = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Model)
    httpd.handle_error = lambda *args: None  # a client gone past its deadline is expected noise
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}", _Model
    httpd.shutdown()


def request_config(url: str, auth=None, **changes) -> dict:
    auth = auth or {"kind": "bearer"}
    adapter = {**seed_adapter("systemone"), **changes}
    adapter["content_hash"] = content_hash(adapter, auth)
    adapter["version_id"] = 3
    return {"url": url, "model": "lab/model-1", "auth": auth, "adapter": adapter}


def job_config(url: str, job_id="5", pools=f"{POOL_A},{POOL_B},{POOL_GENERATIVE}", deadline=None, **changes):
    environ = {
        "REQUEST_CONFIG": json.dumps(request_config(f"{url}/decide", **changes)),
        "JOB_ID": job_id,
        "MODEL": "Lab Model",
        "POOL_FILES": pools,
        "RESULTS_PATH": f"{RESULTS}-{job_id}",
    }
    if deadline:
        environ["RUN_DEADLINE_S"] = str(deadline)
    return load_config(environ)


def run(bucket, tmp_path, url, environ=LOCAL, **kwargs):
    config = job_config(url, **kwargs)
    return runner.run_job(config, bucket, tmp_path / "work", environ=environ), config


def read(bucket, config, name):
    return json.loads(bucket.read_bytes(f"{config.results_path}/{name}"))


# -- the happy path ------------------------------------------------------


def test_a_seeded_adapter_produces_one_backend_result_file_per_mcq_pool(bucket, tmp_path, server):
    url, model = server
    report, config = run(bucket, tmp_path, url)

    assert report.succeeded and not report.degraded
    assert set(report.files) == {META_FILE, RUN_FILE, result_filename(POOL_A), result_filename(POOL_B)}
    result = read(bucket, config, result_filename(POOL_A))
    assert result["pool_file"] == POOL_A and (result["category"], result["task"]) == ("7", "21")
    assert set(result["average_scores"]) == set(POOL_METRICS)
    samples = result["samples"]["decision"]
    assert len(samples) == 6 and samples[0]["doc"] == {"id": "a-0", "dataset_item_id": 1000}
    assert set(samples[0]["scores"]) >= {"accuracy", "invalid_rate", "brier_normalized", "decision_confidence"}
    assert read(bucket, config, result_filename(POOL_B))["samples"]["decision"][0]["doc"] == {"id": "b-0"}
    assert all(auth == "Bearer secret-key" for _, auth, _ in model.calls)


def test_per_question_scores_average_to_the_pool_score(bucket, tmp_path, server):
    report, config = run(bucket, tmp_path, server[0])
    samples = read(bucket, config, result_filename(POOL_B))["samples"]["decision"]
    pool = read(bucket, config, result_filename(POOL_B))["average_scores"]
    for metric in ("accuracy", "invalid_rate", "brier_normalized"):
        mean = sum(sample["scores"][metric] for sample in samples) / len(samples)
        assert mean == pytest.approx(pool[metric], abs=1e-12)


def test_the_meta_file_records_the_adapter_and_why_items_failed(bucket, tmp_path, server):
    _, config = run(bucket, tmp_path, server[0])
    meta = read(bucket, config, META_FILE)
    adapter = request_config("x")["adapter"]

    assert meta["mapping_origin"] == "native" and meta["state"] == "complete"
    assert (meta["adapter_version_id"], meta["adapter_content_hash"]) == (3, adapter["content_hash"])
    assert (meta["engine"], meta["engine_version"]) == (adapter["engine"], adapter["engine_version"])
    assert meta["invalid_reasons"]["refusal"] == 0 and meta["circuit_broken"] is False
    assert meta["pools_scored"] == [POOL_A, POOL_B]
    assert [skip["pool_file"] for skip in meta["pools_skipped"]] == [POOL_GENERATIVE]
    assert not {"spec", "output_format", "registered_specs"} & set(meta)


def test_the_uploaded_run_replays_through_its_own_transform_and_carries_no_secret(bucket, tmp_path, server):
    _, config = run(bucket, tmp_path, server[0])
    audit = audit_document(read(bucket, config, RUN_FILE))
    assert audit.passed, audit.report()
    assert b"secret-key" not in bucket.read_bytes(f"{config.results_path}/{RUN_FILE}")


def test_a_non_seeded_version_is_reported_as_onboarded(bucket, tmp_path, server):
    response = seed_adapter("systemone")["response_transform"].replace('"status": "answer"', '"status":"answer"')
    _, config = run(bucket, tmp_path, server[0], response_transform=response)
    assert read(bucket, config, META_FILE)["mapping_origin"] == "onboarded"


def test_derived_probabilities_publish_no_calibration(bucket, tmp_path, server):
    report, config = run(bucket, tmp_path, server[0], derived_probabilities=True)
    assert report.succeeded
    result = read(bucket, config, result_filename(POOL_A))
    assert result["probability_metrics"] == "derived_probabilities"
    assert not {"brier_normalized", "ece", "aurc"} & set(result["average_scores"])
    assert "brier_normalized" not in result["samples"]["decision"][0]["scores"]
    assert read(bucket, config, RUN_FILE)["items"][0]["probabilities"] is not None  # kept in GCS only


def test_refusals_are_scored_wrong_and_counted(bucket, tmp_path, server):
    server[1].mode = "refusal"
    report, config = run(bucket, tmp_path, server[0])
    assert report.succeeded
    meta = read(bucket, config, META_FILE)
    assert meta["invalid_reasons"]["refusal"] == 12
    assert meta["distribution_problems"] == {"missing": 12, "malformed": 0}


# -- failure paths --------------------------------------------------------


def test_a_broken_adapter_trips_the_circuit_and_publishes_no_scores(bucket, tmp_path, server):
    server[1].mode = "malformed"
    report, config = run(bucket, tmp_path, server[0])

    assert not report.succeeded and report.code == "DECISION_ADAPTER_BROKEN"
    assert set(report.files) == {META_FILE}
    meta = read(bucket, config, META_FILE)
    assert meta["circuit_broken"] is True and meta["state"] == "circuit_broken"


def test_a_model_that_keeps_failing_aborts_with_meta_only(bucket, tmp_path, server):
    server[1].mode = "down"
    report, _ = run(bucket, tmp_path, server[0], retry={**seed_adapter("systemone")["retry"], "backoff_s": 0})
    assert not report.succeeded and "503" in report.error and set(report.files) == {META_FILE}


def test_a_private_model_address_is_refused_unless_a_local_run_allows_it(bucket, tmp_path, server):
    report, _ = run(bucket, tmp_path, server[0], environ={"API_KEY": "k"})
    assert not report.succeeded and "not public" in report.error
    assert server[1].calls == []


def test_a_tampered_adapter_is_refused_before_any_call(bucket, tmp_path, server):
    config = job_config(server[0])
    tampered = dict(config.request_config)
    tampered["adapter"] = {**tampered["adapter"], "request_transform": '{"state": state}'}
    with pytest.raises(ConfigError, match="content_hash"):
        runner.run_job(config.__class__(**{**config.__dict__, "request_config": tampered}), bucket, tmp_path, environ=LOCAL)
    assert server[1].calls == []


def test_a_job_with_no_mcq_pool_fails_before_calling_the_model(bucket, tmp_path, server):
    report, _ = run(bucket, tmp_path, server[0], pools=POOL_GENERATIVE)
    assert not report.succeeded and "no scorable MCQ pool files" in report.error
    assert server[1].calls == []


def test_the_deadline_aborts_instead_of_hanging(bucket, tmp_path, server):
    server[1].mode = "slow"
    report, _ = run(bucket, tmp_path, server[0], deadline=0.2)
    assert not report.succeeded and "deadline" in report.error


# -- main(): finalize over HTTP ------------------------------------------


def test_main_runs_in_local_mode_and_finalizes_with_the_adapter_hash(tmp_path, monkeypatch, server):
    url, model = server
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".tasks").mkdir()
    (tmp_path / ".tasks" / "three.json").write_text(json.dumps(platform_pool("a", OPTIONS_3, True)))
    config = request_config(f"{url}/decide")

    code = runner.main({
        **LOCAL,
        "REQUEST_CONFIG": json.dumps(config),
        "JOB_ID": "77",
        "API_HOST": url,
        "FINALIZE_TOKEN": "jwt-token",
    })

    assert code == 0
    finalize_calls = [call for call in model.calls if call[0].startswith("/evaluation-jobs/")]
    assert finalize_calls == [(
        "/evaluation-jobs/77/finalize",
        "Bearer jwt-token",
        {"outcome": "succeeded", "adapter_content_hash": config["adapter"]["content_hash"]},
    )]
    assert (tmp_path / ".results" / META_FILE).is_file()


def test_main_reports_a_broken_circuit_with_its_code(tmp_path, monkeypatch, server):
    url, model = server
    model.mode = "malformed"
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".tasks").mkdir()
    (tmp_path / ".tasks" / "three.json").write_text(json.dumps(platform_pool("a", OPTIONS_3, True)))

    code = runner.main({**LOCAL, "REQUEST_CONFIG": json.dumps(request_config(f"{url}/decide")),
                        "JOB_ID": "78", "API_HOST": url, "FINALIZE_TOKEN": "t"})

    assert code == 1
    body = [call for call in model.calls if call[0].startswith("/evaluation-jobs/")][0][2]
    assert body["outcome"] == "failed" and body["code"] == "DECISION_ADAPTER_BROKEN"


def test_main_reports_a_broken_configuration_as_a_failed_job(server):
    url, model = server
    code = runner.main({"API_HOST": url, "FINALIZE_TOKEN": "t", "JOB_ID": "9"})
    assert code == 2
    body = [call for call in model.calls if call[0].startswith("/evaluation-jobs/")][0][2]
    assert body["outcome"] == "failed" and "REQUEST_CONFIG" in body["error"]


# -- resuming -------------------------------------------------------------


def test_a_relaunched_job_resumes_from_its_checkpoint_and_scores_the_same(bucket, tmp_path, server):
    url, model = server
    first, config = run(bucket, tmp_path, url)
    full = read(bucket, config, RUN_FILE)["summary"]["overall"]
    calls_first = len(model.calls)

    # Drop every published file and the last chunk, as if the build died mid-run.
    chunks = sorted(path for path in bucket.list(config.results_path) if "/_checkpoint/" in path)
    for path in [*(p for p in bucket.list(config.results_path) if "/_checkpoint/" not in p), chunks[-1]]:
        (tmp_path / "bucket" / path).unlink()

    second, _ = run(bucket, tmp_path, url)
    meta = read(bucket, config, META_FILE)
    assert second.succeeded and meta["resumed_items"] == 10
    assert len(model.calls) - calls_first == 2  # only the items of the lost chunk
    assert read(bucket, config, RUN_FILE)["summary"]["overall"]["accuracy"] == full["accuracy"]
    assert audit_document(read(bucket, config, RUN_FILE)).passed


def test_a_new_adapter_version_does_not_resume_the_old_ones_items(bucket, tmp_path, server):
    url, model = server
    run(bucket, tmp_path, url)
    calls = len(model.calls)
    _, config = run(bucket, tmp_path, url, retry={**seed_adapter("systemone")["retry"], "attempts": 4})
    assert read(bucket, config, META_FILE)["resumed_items"] == 0
    assert len(model.calls) - calls == 12


def test_a_query_key_never_reaches_the_reported_error_or_any_file(bucket, tmp_path, server):
    url, model = server
    model.mode = "down"
    auth = {"kind": "query", "name": "api_key"}
    report, config = run(bucket, tmp_path, url, environ={**LOCAL, "API_KEY": "SUPERSECRET123"}, auth=auth,
                         retry={**seed_adapter("systemone")["retry"], "backoff_s": 0})
    assert not report.succeeded and "503" in report.error
    for path in bucket.list(config.results_path):
        assert b"SUPERSECRET123" not in bucket.read_bytes(path)


def test_main_scrubs_the_key_from_the_finalize_error(tmp_path, monkeypatch, server):
    url, model = server
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".tasks").mkdir()
    (tmp_path / ".tasks" / "three.json").write_text(json.dumps(platform_pool("a", OPTIONS_3, True)))
    config = request_config("http://127.0.0.1:9/decide", auth={"kind": "query", "name": "api_key"})

    code = runner.main({**LOCAL, "API_KEY": "SUPERSECRET123", "REQUEST_CONFIG": json.dumps(config),
                        "JOB_ID": "80", "API_HOST": url, "FINALIZE_TOKEN": "t"})

    body = [call for call in model.calls if call[0].startswith("/evaluation-jobs/")][0][2]
    assert code == 1 and "SUPERSECRET123" not in json.dumps(body) and "<API_KEY>" in body["error"]


def test_another_model_on_the_same_protocol_never_resumes_these_items(bucket, tmp_path, server):
    url, model = server
    run(bucket, tmp_path, url)
    calls = len(model.calls)
    other = job_config(url)
    other_config = dict(other.request_config, model="lab/model-2")
    report = runner.run_job(other.__class__(**{**other.__dict__, "request_config": other_config}), bucket,
                            tmp_path / "work2", environ=LOCAL)
    assert report.succeeded and read(bucket, other, META_FILE)["resumed_items"] == 0
    assert len(model.calls) - calls == 12


def test_two_questions_with_one_id_fail_the_job_before_any_call(bucket, tmp_path, server):
    url, model = server
    pool = platform_pool("a", OPTIONS_3, True)
    pool["data"][1]["id"] = pool["data"][0]["id"]
    bucket.write_bytes(POOL_A, json.dumps(pool).encode())
    report, _ = run(bucket, tmp_path, url)
    assert not report.succeeded and "unique ids" in report.error and model.calls == []


def test_audit_checks_a_redacted_reply_against_its_stored_mapping_without_failing(bucket, tmp_path, server):
    _, config = run(bucket, tmp_path, server[0])
    document = read(bucket, config, RUN_FILE)
    document["items"][0]["raw_response"]["reply"]["answers"]["answer"]["token"] = "[redacted]"
    audit = audit_document(document)
    assert audit.passed and audit.unreplayed == 1


def test_an_item_the_model_refused_is_not_checkpointed(bucket, tmp_path, server):
    url, model = server
    model.mode = "refuse_one"
    _, config = run(bucket, tmp_path, url)
    chunks = [p for p in bucket.list(config.results_path) if "/_checkpoint/" in p]
    saved = b"".join(bucket.read_bytes(p) for p in chunks).decode()
    assert saved.count("\n") == 11 and "http_status" not in saved
