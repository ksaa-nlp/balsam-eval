"""Canary suite v1: a held-out delivery gate in both option orders."""

import pytest

from decisioneval.onboard.canary import (
    GATE,
    HELD_OUT,
    ITEMS,
    SEED_CHECK,
    SUITE_VERSION,
    VISIBLE,
    Answer,
    judge,
    rotated,
    tail_probability,
)


def test_the_suite_is_24_items_six_per_option_count_with_a_stratified_split():
    assert SUITE_VERSION == "canary-v1" and len(ITEMS) == 24
    for count in (2, 3, 4, 5):
        group = [item for item in ITEMS if len(item.options) == count]
        assert len(group) == 6
        assert [item.id for item in group if item in VISIBLE] == [f"{count}.{i}" for i in (1, 3, 5)]
        assert [item.id for item in group if item in HELD_OUT] == [f"{count}.{i}" for i in (2, 4, 6)]
    assert [item.id for item in SEED_CHECK] == ["2.1", "3.2", "4.3", "5.4"]


def test_two_items_use_numeric_labels_and_every_gold_is_one_of_its_options():
    assert [item.id for item in ITEMS if item.option_ids[0] == "1"] == ["4.1", "4.5"]
    for item in ITEMS:
        assert item.gold in item.option_ids and len(set(item.options)) == len(item.options)


@pytest.mark.parametrize("item", ITEMS, ids=lambda item: item.id)
def test_the_rotation_moves_every_option_and_keeps_the_gold_text(item):
    turned = rotated(item)
    assert turned.option_ids == item.option_ids
    assert all(a != b for a, b in zip(item.options, turned.options))
    assert sorted(turned.options) == sorted(item.options)
    assert turned.gold_text == item.gold_text


def test_the_gate_maths_matches_the_chance_tails():
    assert tail_probability(9) == pytest.approx(0.0021, abs=5e-5)
    assert tail_probability(8) == pytest.approx(0.0121, abs=5e-5)
    assert GATE == 9


def answers_from(picker):
    return [Answer(item=item, order=order, choice=picker(item if order == "original" else rotated(item)))
            for item in HELD_OUT + VISIBLE for order in ("original", "rotated")]


def gold_picker(item):
    return item.option_ids[item.options.index(item.gold_text)]


def test_a_correct_model_passes_with_a_tiny_p_value():
    verdict = judge(answers_from(gold_picker))
    assert verdict.verdict == "pass"
    assert (verdict.correct_original, verdict.correct_rotated, verdict.item_count) == (12, 12, 12)
    assert verdict.p_value < 1e-6


def test_an_always_first_picker_fails_the_gate():
    verdict = judge(answers_from(lambda item: item.option_ids[0]))
    assert verdict.verdict == "delivery_fail"
    assert (verdict.correct_original, verdict.correct_rotated) == (5, 4)


def test_a_transform_that_drops_the_question_answers_blind_and_fails():
    verdict = judge(answers_from(lambda item: item.option_ids[len(item.options) // 2]))
    assert verdict.verdict == "delivery_fail"


def test_one_order_below_the_gate_fails_even_if_the_other_passes():
    original = {item.id: item.options for item in ITEMS}

    def picker(item):
        return gold_picker(item) if item.options == original[item.id] else item.option_ids[0]

    verdict = judge(answers_from(picker))
    assert verdict.correct_original == 12 and verdict.correct_rotated < GATE
    assert verdict.verdict == "delivery_fail"


def test_the_behaviour_report_counts_positions_disagreements_and_misses():
    verdict = judge(answers_from(lambda item: item.option_ids[0]))
    report = verdict.behaviour_report
    assert report["position_pick_rate"] == {"1": 1.0}
    right = sum(gold_picker(i) == i.option_ids[0] for i in HELD_OUT + VISIBLE) + sum(
        gold_picker(rotated(i)) == i.option_ids[0] for i in HELD_OUT + VISIBLE)
    assert len(report["misses"]) + report["held_out_misses"] == 48 - right
    assert len(report["rotation_disagreements"]) + report["held_out_rotation_disagreements"] == 24
    held = {item.id for item in HELD_OUT}
    assert not any(row["item"] in held for row in report["misses"] + report["rotation_disagreements"])
    assert all(row["original"] != row["rotated"] for row in report["rotation_disagreements"])


def test_a_refusal_is_a_miss_with_no_choice():
    answers = answers_from(gold_picker)
    answers[0] = Answer(item=answers[0].item, order="original", choice=None)  # a held-out item
    verdict = judge(answers)
    assert verdict.correct_original == 11
    assert verdict.behaviour_report["held_out_misses"] == 1 and verdict.behaviour_report["misses"] == []
