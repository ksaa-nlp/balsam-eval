"""Static checks on a transform's JSONata AST, before it ever runs.

A transform must be a pure function of its input: no clocks, randomness, eval
or regex, and no recursion. Lambdas may only be passed straight to the
higher-order built-ins and no name can hold a lambda, so nothing can call itself.
"""

from __future__ import annotations

from typing import Any

from jsonata.parser import Parser

BANNED_NAMES = frozenset({"eval", "now", "millis", "random", "match"})
LAMBDA_TAKERS = frozenset({"map", "filter", "reduce", "sort", "each"})
RESERVED = BANNED_NAMES | LAMBDA_TAKERS


def lint(ast: Any) -> list[str]:
    """Problems with the expression, empty when it is allowed."""
    problems: list[str] = []
    passed_lambdas: set[int] = set()
    for node in _nodes(ast):
        kind = node.type
        if kind == "regex":
            problems.append("regex literals are not allowed")
        elif kind == "variable" and node.value in BANNED_NAMES:
            problems.append(f"${node.value} is not allowed")
        elif kind == "bind" and _name(node.lhs) in RESERVED:
            # Rebinding $map to another built-in would let a lambda be collected as data.
            problems.append(f"${node.lhs.value} can't be redefined")
        elif kind == "lambda" and any(_name(arg) in RESERVED for arg in node.arguments or ()):
            problems.append("a function parameter can't be named after a built-in function")
        elif kind == "apply" and _is_call_to_taker(node.rhs) and len(node.rhs.arguments or ()) == 1:
            if _is_lambda(node.rhs.arguments[0]):  # `xs ~> $map(function…)`: the lambda is the function slot
                passed_lambdas.add(id(node.rhs.arguments[0]))
        elif kind == "partial":
            # `$p := $map(?, function($x){ $p($x) })` would let a lambda reach itself.
            problems.append("partial application is not allowed")
        elif _is_call_to_taker(node):
            # Only the function slot: a lambda passed as data could be handed itself and recurse.
            arguments = node.arguments or ()
            if len(arguments) > 1 and _is_lambda(arguments[1]):
                passed_lambdas.add(id(arguments[1]))
    for node in _nodes(ast):
        # The parser wraps tail calls in thunk lambdas of its own; those are not user code.
        if _is_lambda(node) and not node.thunk and id(node) not in passed_lambdas:
            problems.append("a function may only be the second argument of $map, $filter, $reduce, $sort or $each")
    return sorted(set(problems))


def _is_call_to_taker(node: Any) -> bool:
    return isinstance(node, Parser.Symbol) and node.type == "function" and _name(node.procedure) in LAMBDA_TAKERS


def _is_lambda(node: Any) -> bool:
    return isinstance(node, Parser.Symbol) and node.type == "lambda"


def _name(node: Any) -> str | None:
    return node.value if isinstance(node, Parser.Symbol) and node.type == "variable" else None


def _nodes(root: Any) -> list[Any]:
    found: list[Any] = []
    seen: set[int] = set()
    stack = [root]
    while stack:
        value = stack.pop()
        if isinstance(value, (list, tuple)):
            stack.extend(value)
            continue
        if not isinstance(value, Parser.Symbol) or id(value) in seen:
            continue
        seen.add(id(value))
        found.append(value)
        stack.extend(child for key, child in vars(value).items() if key != "_outer_instance")
    return found
