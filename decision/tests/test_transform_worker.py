"""The sandboxed transform worker."""

import pytest

from decisioneval.transforms.engine import TransformError
from decisioneval.transforms.worker import MAX_OUTPUT_BYTES, TransformWorker


@pytest.fixture
def worker():
    with TransformWorker() as running:
        yield running


def test_a_transform_runs_in_the_worker(worker):
    assert worker.run('{"s": state & "؟"}', {"state": "هل"}) == {"s": "هل؟"}


def test_the_worker_is_reused_across_calls(worker):
    worker.run("a", {"a": 1})
    pid = worker._process.pid
    assert worker.run("a + 1", {"a": 1}) == 2
    assert worker._process.pid == pid


def test_a_refused_expression_fails_in_the_worker_too(worker):
    with pytest.raises(TransformError, match="now"):
        worker.run("$now()", {})


def test_an_evaluation_error_comes_back_and_the_worker_survives(worker):
    with pytest.raises(TransformError, match="boom"):
        worker.run('$error("boom")', {})
    assert worker.run("a", {"a": 3}) == 3


def test_a_runaway_transform_is_killed_and_the_next_call_gets_a_fresh_worker():
    with TransformWorker(timeout_s=1.0) as worker:
        worker.run("a", {"a": 1})
        first = worker._process.pid
        with pytest.raises(TransformError, match="ran past"):
            worker.run("$count($map([1..10000000], function($x){ $x * $x }))", {})
        assert worker.run("a", {"a": 2}) == 2
        assert worker._process.pid != first


def test_an_oversized_output_is_refused(worker):
    with pytest.raises(TransformError, match=str(MAX_OUTPUT_BYTES)):
        worker.run("$join($map([1..20000], function($x){ \"xxxxx\" }), \"\")", {})


def test_nan_in_a_reply_reaches_the_contract_rather_than_crashing(worker):
    result = worker.run("p", {"p": float("nan")})
    assert result != result


def test_the_worker_holds_no_environment(monkeypatch):
    monkeypatch.setenv("API_KEY", "s3cret")
    with TransformWorker() as fresh:
        fresh.run("a", {"a": 1})
        environ = open(f"/proc/{fresh._process.pid}/environ").read() if __import__("os").path.exists("/proc") else None
        if environ is None:  # macOS: ask ps for the child's environment
            import subprocess
            environ = subprocess.run(["ps", "eww", "-o", "command=", "-p", str(fresh._process.pid)],
                                     capture_output=True, text=True).stdout
        assert "s3cret" not in environ


@pytest.mark.skipif(not __import__("sys").platform.startswith("linux"), reason="/proc and prctl are Linux-only")
def test_a_child_cannot_read_the_parents_environment_once_a_worker_started(monkeypatch):
    import subprocess
    import sys

    monkeypatch.setenv("API_KEY", "sk-test-SECRETVALUE123")
    with TransformWorker() as fresh:
        fresh.run("a", {"a": 1})
        probe = subprocess.run(
            [sys.executable, "-c", "import os; print(open('/proc/%d/environ' % os.getppid(), 'rb').read())"],
            capture_output=True, text=True,
        )
    assert "SECRETVALUE" not in probe.stdout


def test_a_closed_worker_refuses_calls_instead_of_respawning():
    worker = TransformWorker()
    assert worker.run("1 + 1", {}) == 2
    worker.close()
    with pytest.raises(TransformError, match="closed"):
        worker.run("1 + 1", {})
    assert worker._process is None
