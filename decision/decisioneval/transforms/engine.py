"""The pinned JSONata engine. Its name and version are part of every adapter's hash."""

from __future__ import annotations

from importlib.metadata import version
from typing import Any

import jsonata

from decisioneval.transforms.lint import lint

ENGINE = "jsonata-python"
ENGINE_VERSION = version(ENGINE)
MAX_EXPRESSION_BYTES = 4096


class TransformError(Exception):
    """A transform could not be compiled or evaluated."""


def compile_expression(expression: str) -> jsonata.Jsonata:
    """Parse and lint one transform. Raises `TransformError` with every problem."""
    if not isinstance(expression, str) or not expression.strip():
        raise TransformError("a transform must be a non-empty string")
    if len(expression.encode("utf-8")) > MAX_EXPRESSION_BYTES:
        raise TransformError(f"a transform must be at most {MAX_EXPRESSION_BYTES} bytes")
    try:
        compiled = jsonata.Jsonata(expression)
    except Exception as exc:
        raise TransformError(f"transform does not parse: {exc}") from exc
    problems = lint(compiled.ast)
    if problems:
        raise TransformError("; ".join(problems))
    return compiled


def evaluate(compiled: jsonata.Jsonata, data: Any) -> Any:
    try:
        return compiled.evaluate(data)
    except Exception as exc:
        raise TransformError(f"transform failed: {exc}") from exc
