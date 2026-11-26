"""The transform linter and engine: only pure, non-recursive JSONata runs."""

import pytest

from decisioneval.transforms.engine import ENGINE, ENGINE_VERSION, TransformError, compile_expression, evaluate


@pytest.mark.parametrize(
    "expression",
    [
        "$now()",
        "$millis()",
        "$random()",
        '$eval("1+1")',
        "$match(s, /a+/)",
        '$replace(s, /a/, "b")',
        '$split(s, /,/)',
        "$contains(s, /x/)",
        "/abc/",
        "($f := function($x){ $f($x) }; $f(1))",
        "($g := function($x){ $x }; $g(1))",
        "function($x){ $x }(1)",
        "a ~> function($x){ $x }",
        "[function($x){ $x }]",
        '{"f": function($x){ $x }}',
        "($e := $eval; $e(\"1\"))",
        "$map(?, function($x){ $x })",
        "$map(function($s){ $s($s) }, function($f){ $f($f) })",
        "$filter(function($s){ $s($s) }, function($f){ $f($f) })",
        "$reduce([1], function($a, $b){ $a }, function($f){ $f($f) })",
        "($map := $append; $fs := $map([], function($f){ $f($f) }); $fs[0]($fs[0]))",
        "$map([$append], function($map){ ($g := $map([], function($f){ $f($f) }); $g[0]($g[0])) })",
        "($sort := $append; $sort([], function($f){ $f($f) }))",
        "($eval := 1; $eval)",
        "($p := $map(?, function($x){ $p($x) }); $p([1]))",
    ],
)
def test_impure_or_recursive_transforms_are_refused(expression):
    with pytest.raises(TransformError):
        compile_expression(expression)


@pytest.mark.parametrize(
    "expression,data,expected",
    [
        ("$map(a, function($x){ $x * 2 })", {"a": [1, 2]}, [2, 4]),
        ("$filter(a, function($x){ $x > 1 })", {"a": [1, 2, 3]}, [2, 3]),
        ("$reduce(a, function($s, $x){ $s + $x })", {"a": [1, 2, 3]}, 6),
        ("$sort(a, function($l, $r){ $l > $r })", {"a": [3, 1, 2]}, [1, 2, 3]),
        ('$each(o, function($v, $k){ $k })', {"o": {"a": 1}}, "a"),
        ("$map(a, function($x){ $string($x) })", {"a": [1, 2]}, ["1", "2"]),  # a tail call inside a lambda
        ('$replace(s, "a", "b")', {"s": "aa"}, "bb"),
        ('$split(s, ",")', {"s": "a,b"}, ["a", "b"]),
        ('$contains(s, "x")', {"s": "xy"}, True),
        ("($n := 2; n * $n)", {"n": 3}, 6),
        ("a ~> $map(function($x){ $x + 1 })", {"a": [1, 2]}, [2, 3]),
        ("a ~> $filter(function($x){ $x > 1 })", {"a": [1, 2]}, 2),
    ],
)
def test_pure_transforms_compile_and_run(expression, data, expected):
    assert evaluate(compile_expression(expression), data) == expected


def test_an_oversized_or_empty_expression_is_refused():
    with pytest.raises(TransformError, match="4096"):
        compile_expression('"' + "x" * 4100 + '"')
    with pytest.raises(TransformError):
        compile_expression("  ")


def test_a_syntax_error_is_a_transform_error():
    with pytest.raises(TransformError, match="parse"):
        compile_expression("{ a: ")


def test_an_evaluation_error_is_a_transform_error():
    with pytest.raises(TransformError, match="failed"):
        evaluate(compile_expression('$error("boom")'), {})


def test_the_engine_is_pinned():
    assert (ENGINE, ENGINE_VERSION) == ("jsonata-python", "0.7.2")
