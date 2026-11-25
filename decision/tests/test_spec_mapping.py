"""The path language, the response mapping and the probe.

The eight synthetic models in :data:`SHAPES` are the generalisation set: none
of them resembles any of the twelve models in ``results/``, and every one is
onboarded through the probe and evaluated without a line of new Python.
"""


import pytest

from decisioneval.evaluator import evaluate_question
from decisioneval.models import DecisionResult, Question
from decisioneval.spec.mapping import map_response
from decisioneval.spec.model import SpecError, build_spec
from decisioneval.spec.paths import MISSING, PathError, get, join, parse
from decisioneval.spec.probe import Sample, probe, to_spec

OPTION_IDS = ("A", "B", "C")
OPTIONS = ("نعم", "لا", "ربما")


def question(option_ids=OPTION_IDS, options=OPTIONS, gold="نعم") -> Question:
    return Question(
        id="1",
        dataset_name="d",
        category="c",
        task="t",
        instruction="س",
        input="ن",
        options=options,
        option_ids=option_ids,
        gold_answer=gold,
        gold_option_id=option_ids[options.index(gold)],
    )


def spec(response: dict, name: str = "synthetic"):
    return build_spec(
        {
            "spec_version": 1,
            "name": name,
            "access_type": "api",
            "version": "v1",
            "response": response,
        }
    )


# --------------------------------------------------------------------------
# the path language


def test_parse_reads_keys_and_indices():
    assert parse("a.b[0].c") == ("a", "b", 0, "c")
    assert parse("@cf/cloudflare/clef") == ("@cf/cloudflare/clef",)
    assert parse("probs.A: نعم") == ("probs", "A: نعم")


def test_parse_rejects_empty_and_garbage():
    for bad in ("", "   ", "[", "a.[x]"):
        with pytest.raises(PathError):
            parse(bad)


def test_get_returns_missing_rather_than_none_for_absent_paths():
    node = {"a": {"b": None}, "l": [1, 2]}
    assert get(node, "a.b") is None
    assert get(node, "a.zz") is MISSING
    assert get(node, "l[1]") == 2
    assert get(node, "l[9]") is MISSING
    assert get(node, "l.x") is MISSING


def test_join_prefixes_without_breaking_indices():
    assert join("response", "answers.answer") == "response.answers.answer"
    assert join("response", "[0].choice") == "response[0].choice"
    assert join("", "a") == "a"


def test_path_language_has_no_way_to_write_a_value():
    """Only keys and indices parse, so a spec cannot carry a literal number."""
    for attempt in ("0.99", "sum(a)", "a+1", "a|b"):
        try:
            steps = parse(attempt)
        except PathError:
            continue
        # Anything that does parse is read as plain keys, never evaluated.
        assert all(isinstance(step, (str, int)) for step in steps)
        assert get({}, attempt) is MISSING


# --------------------------------------------------------------------------
# the eight synthetic shapes (Phase 4 generalisation set)

SHAPES: dict[str, dict] = {
    # (b) probabilities as a dict keyed by zero-based index
    "index-keyed": {
        "raw": {"out": {"pick": 0, "p": {"0": 0.7, "1": 0.2, "2": 0.1}}},
        "response": {
            "envelope": "out",
            "choice": {"path": "pick", "kind": "index0"},
            "probabilities": {"path": "p", "keying": "index0"},
        },
        "expect": {"A": 0.7, "B": 0.2, "C": 0.1},
        "choice": "A",
    },
    # (b) bare list in option order, one-based choice index
    "bare-list": {
        "raw": {"rank": 2, "scores": [0.1, 0.8, 0.1]},
        "response": {
            "choice": {"path": "rank", "kind": "index1"},
            "probabilities": {"path": "scores", "keying": "order"},
        },
        "expect": {"A": 0.1, "B": 0.8, "C": 0.1},
        "choice": "B",
    },
    # (b) ranked list of {label, score}
    "ranked-pairs": {
        "raw": {
            "best": "ربما",
            "ranking": [
                {"label": "ربما", "score": 0.6},
                {"label": "نعم", "score": 0.3},
                {"label": "لا", "score": 0.1},
            ],
        },
        "response": {
            "choice": {"path": "best", "kind": "option_text"},
            "probabilities": {
                "path": "ranking",
                "keying": "pairs",
                "label_path": "label",
                "score_path": "score",
                "label_kind": "option_text",
            },
        },
        "expect": {"A": 0.3, "B": 0.1, "C": 0.6},
        "choice": "C",
    },
    # (d) choice only, no distribution
    "choice-only": {
        "raw": {"verdict": {"label": "B"}},
        "response": {"choice": {"path": "verdict.label", "kind": "option_id"}},
        "expect": None,
        "choice": "B",
    },
    # (e) logits only. The spec does not point at them at all.
    "logits-only": {
        "raw": {"argmax": "C", "logits": {"A": -2.4, "B": -1.1, "C": 3.8}},
        "response": {"choice": {"path": "argmax", "kind": "option_id"}},
        "expect": None,
        "choice": "C",
    },
    # (b) probabilities keyed by option text
    "text-keyed": {
        "raw": {"answer": "لا", "dist": {"نعم": 0.25, "لا": 0.6, "ربما": 0.15}},
        "response": {
            "choice": {"path": "answer", "kind": "option_text"},
            "probabilities": {"path": "dist", "keying": "option_text"},
        },
        "expect": {"A": 0.25, "B": 0.6, "C": 0.15},
        "choice": "B",
    },
    # (c) answer buried two envelopes deep, OpenAI-chat style
    "double-envelope": {
        "raw": {
            "choices": [
                {"message": {"decision": {"pick": "A", "p": {"A": 0.5, "B": 0.3, "C": 0.2}}}}
            ]
        },
        "response": {
            "envelope": "choices[0].message.decision",
            "choice": {"path": "pick", "kind": "option_id"},
            "probabilities": {"path": "p", "keying": "option_id"},
        },
        "expect": {"A": 0.5, "B": 0.3, "C": 0.2},
        "choice": "A",
    },
    # (f) refusal
    "refusal": {
        "raw": {"status": "refused", "reason": "policy"},
        "response": {
            "choice": {"path": "pick", "kind": "option_id"},
            "invalid_when": [{"path": "status", "equals": "refused"}],
        },
        "expect": None,
        "choice": None,
    },
}


@pytest.mark.parametrize("label", sorted(SHAPES))
def test_every_synthetic_shape_maps_with_no_new_python(label):
    shape = SHAPES[label]
    mapped = map_response(spec(shape["response"]), shape["raw"], OPTION_IDS, OPTIONS)
    assert mapped.predicted_option_id == shape["choice"]
    assert mapped.probabilities == shape["expect"]
    assert mapped.valid is (shape["choice"] is not None)


@pytest.mark.parametrize("label", sorted(SHAPES))
def test_every_synthetic_shape_evaluates_end_to_end(label):
    """The evaluator scores a spec-mapped decision exactly as a hand-written one."""
    shape = SHAPES[label]
    mapped = map_response(spec(shape["response"]), shape["raw"], OPTION_IDS, OPTIONS)
    item = evaluate_question(
        question(),
        DecisionResult(
            predicted_option_id=mapped.predicted_option_id,
            probabilities=mapped.probabilities,
            confidence=mapped.provider_confidence,
            raw_response=shape["raw"],
            valid=mapped.valid,
        ),
    )
    assert item.invalid is (shape["choice"] is None)
    assert item.probabilities == shape["expect"]
    if shape["choice"] is not None:
        assert item.correct is (shape["choice"] == "A")


def test_logits_shape_reports_null_probability_metrics_not_a_softmax():
    shape = SHAPES["logits-only"]
    mapped = map_response(spec(shape["response"]), shape["raw"], OPTION_IDS, OPTIONS)
    assert mapped.probabilities is None
    assert mapped.predicted_option_id == "C"


def test_pointing_at_logits_is_refused_rather_than_rescaled():
    """A spec may point at logits; they then fail the value rules and stay unscored."""
    pointed = spec(
        {
            "choice": {"path": "argmax", "kind": "option_id"},
            "probabilities": {"path": "logits", "keying": "option_id"},
        }
    )
    mapped = map_response(pointed, SHAPES["logits-only"]["raw"], OPTION_IDS, OPTIONS)
    assert mapped.probabilities is None
    assert mapped.predicted_option_id == "C"


def test_on_bad_sum_reject_marks_the_item_invalid():
    rejecting = spec(
        {
            "choice": {"path": "pick", "kind": "option_id"},
            "probabilities": {"path": "p", "keying": "option_id"},
            "on_bad_sum": "reject",
        }
    )
    raw = {"pick": "A", "p": {"A": 0.9, "B": 0.9, "C": 0.9}}
    mapped = map_response(rejecting, raw, OPTION_IDS, OPTIONS)
    assert mapped.valid is False
    assert mapped.probabilities is None


# --------------------------------------------------------------------------
# negative cases that must be rejected


def test_probabilities_that_do_not_sum_to_one_are_not_renormalized():
    mapped = map_response(
        spec(
            {
                "choice": {"path": "pick", "kind": "option_id"},
                "probabilities": {"path": "p", "keying": "option_id"},
            }
        ),
        {"pick": "A", "p": {"A": 0.5, "B": 0.25, "C": 0.1}},
        OPTION_IDS,
        OPTIONS,
    )
    assert mapped.probabilities is None
    assert mapped.predicted_option_id == "A"


def test_list_positions_claimed_as_option_ids_are_rejected():
    """Keys "0","1","2" declared as option_id do not become A,B,C."""
    mapped = map_response(
        spec(
            {
                "choice": {"path": "pick", "kind": "option_id"},
                "probabilities": {"path": "p", "keying": "option_id"},
            }
        ),
        {"pick": "A", "p": {"0": 0.7, "1": 0.2, "2": 0.1}},
        OPTION_IDS,
        OPTIONS,
    )
    assert mapped.probabilities is None


def test_a_partial_vector_is_unavailable_and_never_filled_in():
    mapped = map_response(
        spec(
            {
                "choice": {"path": "pick", "kind": "option_id"},
                "probabilities": {"path": "p", "keying": "option_id"},
            }
        ),
        {"pick": "A", "p": {"A": 0.6, "B": 0.4}},
        OPTION_IDS,
        OPTIONS,
    )
    assert mapped.probabilities is None


def test_unknown_response_keys_are_rejected_not_ignored():
    for smuggled in ({"softmax": True}, {"scale": 2}, {"value": 0.5}):
        with pytest.raises(SpecError, match="unknown key"):
            build_spec(
                {
                    "spec_version": 1,
                    "name": "x",
                    "access_type": "api",
                    "response": {"choice": {"path": "a"}, **smuggled},
                }
            )


def test_a_confidence_scalar_cannot_be_declared_as_a_distribution():
    with pytest.raises(SpecError, match="keying"):
        build_spec(
            {
                "spec_version": 1,
                "name": "x",
                "access_type": "api",
                "response": {
                    "choice": {"path": "a"},
                    "probabilities": {"path": "conf", "keying": "scalar"},
                },
            }
        )


def test_auth_cannot_carry_an_inline_secret():
    with pytest.raises(SpecError, match="token"):
        build_spec(
            {
                "spec_version": 1,
                "name": "x",
                "access_type": "api",
                "request": {"url": "https://e", "auth": {"kind": "bearer", "token": "sk-live"}},
                "response": {"choice": {"path": "a"}},
            }
        )


def test_build_spec_raises_with_every_problem_listed():
    with pytest.raises(SpecError) as excinfo:
        build_spec({"spec_version": 9, "access_type": "weird", "response": {}})
    message = str(excinfo.value)
    assert "spec_version" in message and "access_type" in message


def samples_for(label: str, chosen: bool = True) -> list[Sample]:
    shape = SHAPES[label]
    return [
        Sample(
            raw=shape["raw"],
            option_ids=OPTION_IDS,
            options=OPTIONS,
            chosen_option_id=shape["choice"] if chosen else None,
        )
    ]


@pytest.mark.parametrize(
    "label",
    ["index-keyed", "bare-list", "ranked-pairs", "text-keyed", "double-envelope"],
)
def test_probe_finds_the_distribution_of_a_shape_it_has_never_seen(label):
    report = probe(samples_for(label))
    assert len(report.probabilities) >= 1
    assert report.resolved
    document = to_spec(label, report)
    mapped = map_response(build_spec(document), SHAPES[label]["raw"], OPTION_IDS, OPTIONS)
    assert mapped.probabilities == SHAPES[label]["expect"]
    assert mapped.predicted_option_id == SHAPES[label]["choice"]


def test_probe_finds_no_distribution_for_logits_and_says_why():
    report = probe(samples_for("logits-only"))
    assert report.probabilities == []
    assert any("logits" in note for note in report.notes)


def test_probe_reports_ambiguity_instead_of_guessing():
    """Two different dicts both sum to 1 on the option ids, so neither wins."""
    raw = {"pick": "A", "p": {"A": 0.6, "B": 0.4, "C": 0.0}, "q": {"A": 0.5, "B": 0.5, "C": 0.0}}
    report = probe([Sample(raw=raw, option_ids=OPTION_IDS, options=OPTIONS, chosen_option_id="A")])
    assert len(report.probabilities) == 2
    assert report.ambiguous
    with pytest.raises(ValueError):
        to_spec("ambiguous", report)


def test_probe_requires_every_sample_to_agree():
    good = {"pick": "A", "p": {"A": 0.6, "B": 0.4, "C": 0.0}}
    coincidence = {
        "pick": "A",
        "p": {"A": 0.6, "B": 0.4, "C": 0.0},
        "q": {"A": 1.0, "B": 0.0, "C": 0.0},
    }
    one = Sample(raw=good, option_ids=OPTION_IDS, options=OPTIONS, chosen_option_id="A")
    two = Sample(raw=coincidence, option_ids=OPTION_IDS, options=OPTIONS, chosen_option_id="A")
    assert [candidate.path for candidate in probe([one, two]).probabilities] == ["p"]
