"""The runner's inputs: the platform's pool files, request config and env."""

from __future__ import annotations

import base64
import json

import pytest

from decisioneval.remote.config import ConfigError, load_config
from decisioneval.remote.normalize import DEFAULT_INSTRUCTION, NotMcqPool, normalize_pool
from decisioneval.remote.storage import LocalStore, StorageError

ITEM = {
    "id": "q-1",
    "instruction": "اختر",
    "input": "نص",
    "mcq": ["نعم", "لا"],
    "output": "نعم",
    "dataset_item_id": 991,
}


# -- pool files ----------------------------------------------------------


def test_a_platform_pool_with_a_title_case_header_and_flat_rows_is_normalised():
    raw = {
        "Type Of Result": "MCQ",
        "category": 7,
        "task": 11,
        "data": [ITEM, {**ITEM, "id": "q-dev", "split_type": "dev"}],
    }

    pool = normalize_pool(raw, "dev/ns/pools/cat-7/a.json")

    assert pool.document["name"] == "dev/ns/pools/cat-7/a.json"
    assert pool.category == "7" and pool.task == "11"
    test_items = pool.document["data"]["test"]
    assert [item["id"] for item in test_items] == ["q-1"]  # the dev row is not scored
    assert test_items[0]["input"] == ["نص"]  # a bare string input becomes a list
    assert "dataset_item_id" not in test_items[0]
    assert pool.dataset_item_id("q-1") == 991


def test_a_legacy_pool_nested_under_json_takes_category_and_task_from_outside():
    raw = {
        "category": 3,
        "task": 4,
        "json": {"type_of_result": "mcq", "data": {"test": [ITEM], "dev": []}},
    }

    pool = normalize_pool(raw, "p.json")

    assert (pool.category, pool.task) == ("3", "4")
    assert len(pool.document["data"]["test"]) == 1


@pytest.mark.parametrize("instruction", [None, "", "   "])
def test_an_item_without_an_instruction_gets_the_default_one(instruction):
    item = {key: value for key, value in ITEM.items() if key != "instruction"}
    if instruction is not None:
        item["instruction"] = instruction

    pool = normalize_pool({"type_of_result": "mcq", "data": [item]}, "p.json")

    assert pool.document["data"]["test"][0]["instruction"] == DEFAULT_INSTRUCTION


@pytest.mark.parametrize(
    "raw, reason",
    [
        ({"Type of result": "Generative", "data": {"test": [ITEM]}}, "not MCQ"),
        ({"data": {"test": [ITEM]}}, "missing"),
        ({"Type of result": "MCQ", "data": {"dev": [ITEM]}}, "test split"),
        ([ITEM], "not a JSON object"),
    ],
)
def test_pools_a_decision_model_cannot_answer_are_rejected_with_a_reason(raw, reason):
    with pytest.raises(NotMcqPool, match=reason):
        normalize_pool(raw, "p.json")


# -- environment ---------------------------------------------------------


def test_the_request_config_is_read_from_base64():
    encoded = base64.b64encode(json.dumps({"url": "https://m"}).encode()).decode()
    config = load_config({"REQUEST_CONFIG_B64": encoded, "JOB_ID": "42"})
    assert config.request_config == {"url": "https://m"}
    assert config.job_id == "42"
    assert not config.uses_gcs and not config.reports_outcome


def test_a_bucket_without_its_paths_is_a_configuration_error():
    with pytest.raises(ConfigError, match="POOL_FILES, RESULTS_PATH"):
        load_config({"REQUEST_CONFIG": "{}", "GCLOUD_BUCKET": "b"})


@pytest.mark.parametrize(
    "environ",
    [
        {},
        {"REQUEST_CONFIG_B64": "%%%"},
        {"REQUEST_CONFIG": "[1]"},
        {"REQUEST_CONFIG": "{}", "RUN_DEADLINE_S": "-1"},
    ],
)
def test_a_broken_environment_is_a_configuration_error(environ):
    with pytest.raises(ConfigError):
        load_config(environ)


# -- local storage -------------------------------------------------------


def test_local_storage_refuses_to_overwrite_with_if_absent_and_to_escape_its_root(tmp_path):
    store = LocalStore(tmp_path)
    assert store.write_bytes("a/b.json", b"1", if_absent=True)
    assert not store.write_bytes("a/b.json", b"2", if_absent=True)
    assert store.read_bytes("a/b.json") == b"1"
    assert store.list("a") == ["a/b.json"]
    with pytest.raises(StorageError):
        store.read_bytes("../outside.json")
