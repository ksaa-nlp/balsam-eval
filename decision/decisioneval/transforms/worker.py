"""Transforms run in a long-lived child process, one JSON line in and one out.

The child caps its own address space, and the parent kills and restarts it when
a call runs past its timeout, so a runaway expression costs one item, not the job.
"""

from __future__ import annotations

import hashlib
import json
import select
import subprocess
import sys
import threading
from typing import Any

from decisioneval.transforms.engine import TransformError

CALL_TIMEOUT_S = 1.0
MAX_OUTPUT_BYTES = 64 * 1024
MEMORY_LIMIT_BYTES = 1024 * 1024 * 1024
MAX_CACHED = 64


class TransformWorker:
    """The parent side. A lock serialises calls, so threads may share one worker."""

    def __init__(self, timeout_s: float = CALL_TIMEOUT_S):
        self.timeout_s = timeout_s
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._closed = False

    def run(self, expression: str, data: Any) -> Any:
        """Evaluate `expression` on `data`. Raises `TransformError` on any failure."""
        with self._lock:
            return self._call(expression, data)

    def _call(self, expression: str, data: Any) -> Any:
        process = self._start()
        assert process.stdin is not None and process.stdout is not None
        try:
            process.stdin.write(json.dumps({"expression": expression, "data": data}, ensure_ascii=False) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._kill()
            raise TransformError(f"transform worker closed: {exc}") from exc
        ready, _, _ = select.select([process.stdout], [], [], self.timeout_s)
        if not ready:
            self._kill()
            raise TransformError(f"transform ran past {self.timeout_s:g}s")
        line = process.stdout.readline()
        if not line:
            self._kill()
            raise TransformError("transform worker died, likely over its memory limit")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise TransformError(str(reply.get("error")))
        return reply.get("result")

    def close(self) -> None:
        # Under the lock, so a thread still queued after a cancelled run can't respawn a child nobody closes.
        with self._lock:
            self._closed = True
            process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()

    def __enter__(self) -> TransformWorker:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _start(self) -> subprocess.Popen[str]:
        if self._closed:
            raise TransformError("transform worker is closed")
        if self._process is None or self._process.poll() is not None:
            _hide_own_environment()
            self._process = subprocess.Popen(
                [sys.executable, "-I", "-X", "utf8", "-m", "decisioneval.transforms.worker"],
                env={},  # the sandbox needs nothing from the environment, least of all its keys
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        return self._process

    def _kill(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            process.kill()
            process.wait()


PR_SET_DUMPABLE = 4


def _hide_own_environment() -> None:
    """A same-user child could read this process's /proc/<pid>/environ, keys included; non-dumpable forbids it."""
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        ctypes.CDLL(None, use_errno=True).prctl(PR_SET_DUMPABLE, 0, 0, 0, 0)
    except (OSError, AttributeError):
        pass  # no prctl: the empty child environment still holds


def _limit_memory() -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    except (ImportError, ValueError, OSError):
        pass  # macOS may refuse RLIMIT_AS; the build runs on Linux, where it holds


def _serve() -> None:
    from decisioneval.transforms.engine import compile_expression, evaluate

    _limit_memory()
    cache: dict[str, Any] = {}
    for line in sys.stdin:
        try:
            request = json.loads(line)
            key = hashlib.sha256(request["expression"].encode("utf-8")).hexdigest()
            if key not in cache:
                if len(cache) >= MAX_CACHED:
                    cache.clear()
                cache[key] = compile_expression(request["expression"])
            text = json.dumps({"ok": True, "result": evaluate(cache[key], request["data"])}, ensure_ascii=False)
            if len(text.encode("utf-8")) > MAX_OUTPUT_BYTES:
                text = json.dumps({"ok": False, "error": f"transform output is over {MAX_OUTPUT_BYTES} bytes"})
        except Exception as exc:  # every failure goes back as one line; the loop must survive it
            text = json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]}, ensure_ascii=False)
        sys.stdout.write(text + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    _serve()
