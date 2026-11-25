"""Adapter content hashes, and the seed rows the backend migration copies."""

import json

import pytest

from decisioneval import cli
from decisioneval.transforms.adapter import SEEDED, content_hash, seed_adapter, seed_hashes, seed_sql

BEARER = {"kind": "bearer"}


def test_ids_are_not_hashed_and_none_valued_auth_keys_are_dropped():
    adapter = seed_adapter("systemone")
    with_ids = {**adapter, "version_id": 12, "content_hash": "x"}
    assert content_hash(with_ids, {"kind": "bearer", "name": None}) == content_hash(adapter, BEARER)


def test_key_order_and_float_spelling_do_not_change_the_hash():
    adapter = seed_adapter("openai-decisions")
    reordered = dict(reversed(list(adapter.items())))
    reordered["retry"] = {**adapter["retry"], "backoff_s": 1}
    assert content_hash(reordered, BEARER) == content_hash(adapter, BEARER)


@pytest.mark.parametrize(
    "change",
    [
        {"request_transform": "{}"},
        {"engine_version": "0.7.3"},
        {"degraded": True},
        {"retry": {"attempts": 4}},
        {"interaction": {"pattern": "gradio-call", "api_name": "/predict"}},
    ],
)
def test_any_content_change_changes_the_hash(change):
    adapter = seed_adapter("systemone")
    assert content_hash({**adapter, **change}, BEARER) != content_hash(adapter, BEARER)


def test_where_the_key_goes_is_part_of_the_hash():
    adapter = seed_adapter("systemone")
    assert content_hash(adapter, {"kind": "header", "name": "x-api-key"}) != content_hash(adapter, BEARER)


def test_seed_sql_inserts_every_seed_once_with_its_hash(capsys):
    cli.main(["adapters", "seed-sql"])
    printed = capsys.readouterr().out
    assert printed == seed_sql()
    assert printed.count("'seeded'") == len(SEEDED)
    for value in seed_hashes().values():
        assert f"'{value}'" in printed
    assert printed.rstrip().endswith('ON CONFLICT ("content_hash") DO NOTHING;')


def test_request_config_prints_a_config_the_runner_accepts(capsys):
    from decisioneval.remote.bound_adapter import load_bound_adapter, mapping_origin

    cli.main(["adapters", "request-config", "openai-decisions", "--url", "https://x.example/v1/decisions",
              "--model", "gpt-6-luna", "--auth", "bearer"])
    bound = load_bound_adapter(json.loads(capsys.readouterr().out))
    assert mapping_origin(bound) == "native" and bound.model == "gpt-6-luna"


def test_an_empty_prefix_hashes_like_none_except_for_bearer():
    from decisioneval.transforms.adapter import content_hash, seed_adapter

    adapter = seed_adapter("systemone")
    header = {"kind": "header", "name": "X-Key"}
    assert content_hash(adapter, {**header, "prefix": ""}) == content_hash(adapter, header)
    assert content_hash(adapter, {"kind": "bearer", "prefix": ""}) != content_hash(adapter, {"kind": "bearer"})
