"""Probe-found mappings compiled to JSONata read every reply exactly as the spec engine does."""

import pytest

from decisioneval.onboard.compile import CompileError, compile_response
from decisioneval.remote.contract import MALFORMED, check
from decisioneval.spec.mapping import map_response
from decisioneval.spec.model import build_spec
from decisioneval.spec.probe import Sample, probe, to_spec
from decisioneval.transforms.engine import compile_expression, evaluate
from tests.conftest import FakeModel

QUESTIONS = [
    (("A", "B"), ("نعم", "لا")),
    (("A", "B", "C"), ("أزرق", "أسود", "أحمر")),
    (("A", "B", "C", "D"), ("آسيا", "أوروبا", "أفريقيا", "أستراليا")),
    (("A", "B", "C", "D", "E"), ("الأحمر", "البنفسجي", "البرتقالي", "الأخضر", "الأسود")),
]


def replies(shape: str, offset: int):
    model = FakeModel(shape)
    return [
        (ids, texts, model.payload(list(ids), list(texts), (index + offset) % len(ids)))
        for index, (ids, texts) in enumerate(QUESTIONS)
    ]


def compiled_for(shape: str):
    samples = [Sample(raw=raw, option_ids=ids, options=texts) for ids, texts, raw in replies(shape, 0)]
    document = to_spec("probe", probe(samples))
    return document, compile_expression(compile_response(document["response"]))


@pytest.mark.parametrize("shape", ["system_one", "index_keyed", "ranked_pairs", "logits_only"])
def test_the_compiled_transform_agrees_with_the_spec_engine(shape):
    document, transform = compiled_for(shape)
    spec = build_spec(document)
    for ids, texts, raw in replies(shape, 1):
        expected = map_response(spec, raw, ids, texts)
        got = check(evaluate(transform, {"reply": raw, "option_ids": list(ids), "options": list(texts)}), ids)
        assert got.choice == expected.predicted_option_id
        assert got.probabilities == expected.probabilities


def test_a_logits_only_model_compiles_to_a_choice_only_transform():
    document, transform = compiled_for("logits_only")
    assert "probabilities" not in document["response"]
    ids, texts, raw = replies("logits_only", 0)[1]
    assert evaluate(transform, {"reply": raw, "option_ids": list(ids), "options": list(texts)})["probabilities"] is None


def test_an_unresolvable_label_in_the_vector_is_malformed_not_dropped():
    _, transform = compiled_for("ranked_pairs")
    reply = {"best": "نعم", "ranking": [{"label": "نعم", "score": 0.7}, {"label": "غير", "score": 0.3}]}
    result = check(evaluate(transform, {"reply": reply, "option_ids": ["A", "B"], "options": ["نعم", "لا"]}), ("A", "B"))
    assert result.choice == "A" and result.distribution_problem == MALFORMED


def test_refusal_rules_compile_to_a_refusal_status():
    response = {"choice": {"path": "out.pick", "kind": "option_id"},
                "invalid_when": [{"path": "out.kind", "equals": "refusal"}, {"path": "out.pick", "mode": "missing"}]}
    transform = compile_expression(compile_response(response))
    run = lambda reply: evaluate(transform, {"reply": reply, "option_ids": ["A", "B"], "options": ["x", "y"]})
    assert run({"out": {"kind": "refusal", "pick": "A"}})["status"] == "refusal"
    assert run({"out": {}})["status"] == "refusal"
    assert run({"out": {"pick": "b"}}) == {"status": "answer", "choice": "B", "probabilities": None}


def test_odd_keys_and_numeric_ids_are_addressed():
    response = {"choice": {"path": "result.@cf/clef.choice", "kind": "option_id"},
                "probabilities": {"path": "result.@cf/clef.p", "keying": "option_id"}}
    transform = compile_expression(compile_response(response))
    reply = {"result": {"@cf/clef": {"choice": 2, "p": {"1": 0.25, "2": 0.75}}}}
    out = evaluate(transform, {"reply": reply, "option_ids": ["1", "2"], "options": ["32", "42"]})
    assert out == {"status": "answer", "choice": "2", "probabilities": {"1": 0.25, "2": 0.75}}


def test_a_backtick_key_is_refused():
    with pytest.raises(CompileError):
        compile_response({"choice": {"path": "a`b", "kind": "option_id"}})
