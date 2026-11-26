"""Compile a probe-found response mapping to a JSONata response transform.

The probe already proved the paths on the model's real replies; compiling them
deterministically means most formats never reach the LLM. Output follows the
decision contract: a vector whose labels don't resolve to distinct option ids
becomes the never-an-option key "", so the contract calls it malformed.
"""

from __future__ import annotations

import json
from typing import Any

from decisioneval.spec.paths import parse

RANGE = "[0..($count($$.option_ids) - 1)]"
MATCH = {
    "option_id": '$type($v) in ["string", "number"] and $uppercase($$.option_ids[$i]) = $uppercase($trim($string($v)))',
    "option_text": '$type($v) in ["string", "number"] and $$.options[$i] = $trim($string($v))',
    "index0": '$type($v) in ["string", "number"] and $trim($string($v)) = $string($i)',
    "index1": '$type($v) in ["string", "number"] and $trim($string($v)) = $string($i + 1)',
}


class CompileError(ValueError):
    """The mapping uses something this compiler does not translate."""


def path(text: str) -> str:
    """A spec path relative to the reply, as a JSONata path from the transform input."""
    out = "reply"
    for step in parse(text):
        if isinstance(step, int):
            out += f"[{step}]"
        elif "`" in step:
            raise CompileError(f"a key holding a backtick can't be addressed: {step!r}")
        else:
            out += f".`{step}`"
    return out


def resolve(value: str, kind: str) -> str:
    """JSONata giving the option id `value` names under `kind`, or null."""
    if kind not in MATCH:
        raise CompileError(f"unknown label kind {kind!r}")
    return (
        f"($v := {value}; $m := $filter({RANGE}, function($i) {{ {MATCH[kind]} }}); "
        f"$count($m) = 1 ? $$.option_ids[$m] : null)"
    )


def compile_response(response: dict[str, Any]) -> str:
    """JSONata for a spec response block (envelope, choice, probabilities, invalid_when)."""
    if response.get("envelope"):
        raise CompileError("compile mappings found on the bare reply, without an envelope")
    choice = response["choice"]
    lines = [f"$choice := {resolve(path(choice['path']), choice.get('kind', 'option_id'))};"]
    probabilities = response.get("probabilities")
    vector = "null"
    if probabilities:
        lines.append(f"$rows := {_rows(probabilities)};")
        lines.append(
            '$clean := $count($rows) > 0 and $count($rows[$type(id) = "string"]) = $count($rows)'
            " and $count($distinct($rows.id)) = $count($rows);"
        )
        vector = '($clean ? $merge($map($rows, function($r) { {$r.id: $r.p} })) : {"": 0})'
    refusal = " or ".join(_condition(rule) for rule in response.get("invalid_when") or ()) or "false"
    lines.append(
        f'{refusal} ? {{"status": "refusal", "choice": null, "probabilities": null}}'
        f' : {{"status": "answer", "choice": $choice, "probabilities": {vector}}}'
    )
    return "(\n  " + "\n  ".join(lines) + "\n)\n"


def _rows(rules: dict[str, Any]) -> str:
    found = path(rules["path"])
    keying = rules.get("keying", "option_id")
    if keying == "pairs":
        label = ".".join(f"`{key}`" for key in parse(rules["label_path"]))
        score = ".".join(f"`{key}`" for key in parse(rules["score_path"]))
        kind = rules.get("label_kind", "option_id")
        return f"$map({found}, function($row) {{ {{\"id\": {resolve(f'$row.{label}', kind)}, \"p\": $row.{score}}} }})"
    if keying == "order":
        return (
            f"($list := {found}; $count($list) = $count($$.option_ids)"
            f" ? $map({RANGE}, function($i) {{ {{\"id\": $$.option_ids[$i], \"p\": $list[$i]}} }}) : [])"
        )
    return f"$each({found}, function($p, $k) {{ {{\"id\": {resolve('$k', keying)}, \"p\": $p}} }})"


def _condition(rule: dict[str, Any]) -> str:
    target = path(rule["path"])
    mode = rule.get("mode", "equals")
    if mode == "missing":
        return f"$not($exists({target}))"
    if mode == "empty":
        return f'($not($exists({target})) or {target} = null or {target} = "")'
    return f"({target} = {json.dumps(rule.get('equals'), ensure_ascii=False)})"
