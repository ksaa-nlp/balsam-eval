"""A deliberately tiny read-only path language for response mapping specs.

Grammar: ``key`` ``.`` ``key`` ``[`` integer ``]``, nothing else. There is no
arithmetic, no function call, no wildcard and no way to write a literal value.
A spec can therefore only *point at* numbers that already exist inside the
model's raw response; it can never produce one. That is what keeps invariant 1
("nothing invents a number") true by construction rather than by review.

Key segments may contain any character except ``.``, ``[`` and ``]``, so Arabic
option text and keys such as ``@cf/cloudflare/clef`` work without escaping.
"""

from __future__ import annotations

import re
from typing import Any

MISSING = object()
"""Returned by :func:`get` when the path does not exist. ``None`` is a value."""

_SEGMENT = re.compile(r"\[(-?\d+)\]|([^.\[\]]+)")


class PathError(ValueError):
    """The path text is not valid in this grammar."""


def parse(path: str) -> tuple[str | int, ...]:
    """Split `path` into string keys and integer indices."""
    text = str(path).strip()
    if not text:
        raise PathError("empty path")
    steps: list[str | int] = []
    position = 0
    while position < len(text):
        if text[position] == ".":
            position += 1
            continue
        match = _SEGMENT.match(text, position)
        if match is None:
            raise PathError(f"invalid path at position {position}: {path!r}")
        index, key = match.groups()
        steps.append(int(index) if index is not None else key)
        position = match.end()
    if not steps:
        raise PathError(f"invalid path: {path!r}")
    return tuple(steps)


def get(node: Any, path: str) -> Any:
    """Read `path` out of `node`, or return :data:`MISSING`."""
    current = node
    for step in parse(path):
        if isinstance(step, int):
            if not isinstance(current, list) or not -len(current) <= step < len(current):
                return MISSING
            current = current[step]
            continue
        if not isinstance(current, dict) or step not in current:
            return MISSING
        current = current[step]
    return current
