"""The MCQ pool loader keeps what it can score and says why it left the rest out."""

import json

import pytest

from decisioneval.datasets import DatasetValidationError, NotMCQDataset, load_dataset
from tests.conftest import mcq_document


def write(tmp_path, document, *, bom=False):
    path = tmp_path / "pool.json"
    path.write_text(("﻿" if bom else "") + json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return path


def with_items(*items):
    document = mcq_document("pool", ["a", "b"], count=0)
    document["data"]["test"] = list(items)
    return document


def row(item_id, **changes):
    return {"id": item_id, "input": ["q"], "mcq": ["x", "y", "z"], "output": "y", **changes}


def test_a_bom_file_loads_and_gold_gets_its_option_id(tmp_path):
    dataset = load_dataset(write(tmp_path, with_items(row(1)), bom=True))
    [question] = dataset.questions
    assert (question.gold_answer, question.gold_option_id) == ("y", "B")


@pytest.mark.parametrize("changes,reason", [
    ({"mcq": ["y", "x", "y"]}, "ambiguous_gold"),
    ({"output": "w"}, "gold_not_in_options"),
    ({"output": ""}, "invalid_output"),
    ({"input": ["clip.wav"]}, "invalid_item"),
    ({"mcq": ["only"]}, "invalid_item"),
])
def test_an_unscorable_item_is_excluded_with_its_reason(tmp_path, changes, reason):
    dataset = load_dataset(write(tmp_path, with_items(row(1), row(2, **changes))))
    assert [question.id for question in dataset.questions] == [1]
    assert [(item.item_id, item.reason) for item in dataset.excluded] == [(2, reason)]


@pytest.mark.parametrize("item_id", [True, None, 1.5])
def test_an_unusable_id_is_excluded(tmp_path, item_id):
    dataset = load_dataset(write(tmp_path, with_items(row(1), row(item_id))))
    assert len(dataset.questions) == 1 and dataset.excluded[0].reason == "invalid_item"


def test_a_non_mcq_file_or_an_empty_test_split_is_refused(tmp_path):
    with pytest.raises(NotMCQDataset):
        load_dataset(write(tmp_path, {**with_items(row(1)), "Type of result": "Generative"}))
    with pytest.raises(DatasetValidationError, match="test split"):
        load_dataset(write(tmp_path, with_items()))
