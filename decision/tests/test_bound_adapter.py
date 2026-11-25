"""The Phase 1 request config: hash verification, engine pin, and key placement."""

import copy

import pytest

from decisioneval.remote.bound_adapter import AdapterConfigError, load_bound_adapter, place_key, require_key
from decisioneval.transforms.adapter import content_hash, seed_adapter
from decisioneval.transforms.engine import TransformError


def config(seed="openai-decisions", auth=None, **adapter_changes):
    auth = auth or {"kind": "bearer"}
    adapter = {**seed_adapter(seed), **adapter_changes}
    adapter["content_hash"] = content_hash(adapter, auth)
    adapter["version_id"] = 7
    return {"url": "https://api.openai.com/v1/decisions", "model": "gpt-6-luna", "auth": auth, "adapter": adapter}


def test_a_seeded_config_loads_with_its_retry_and_concurrency():
    bound = load_bound_adapter(config())
    assert (bound.version_id, bound.concurrency, bound.timeout_s) == (7, 4, 60.0)
    assert bound.retry.attempts == 3 and 429 in bound.retry.on_status
    assert bound.content_hash == config()["adapter"]["content_hash"]


def test_an_edited_transform_fails_the_hash_check():
    tampered = config()
    tampered["adapter"]["request_transform"] = '{"model": model}'
    with pytest.raises(AdapterConfigError, match="content_hash"):
        load_bound_adapter(tampered)


def test_moving_the_key_without_rehashing_fails_the_hash_check():
    moved = config()
    moved["auth"] = {"kind": "query", "name": "key"}
    with pytest.raises(AdapterConfigError, match="content_hash"):
        load_bound_adapter(moved)


def test_a_version_validated_on_another_engine_is_refused():
    with pytest.raises(AdapterConfigError, match="validated on"):
        load_bound_adapter(config(engine_version="0.6.0"))


def test_a_transform_the_linter_refuses_is_refused_even_with_a_good_hash():
    with pytest.raises(AdapterConfigError, match="now"):
        load_bound_adapter(config(request_transform="$now()"))


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda c: c.update(url="ftp://x"), "url"),
        (lambda c: c.update(model=None), "model"),
        (lambda c: c.update(extra=1), "unsupported request config keys"),
        (lambda c: c["adapter"].update(extra=1), "unsupported adapter keys"),
        (lambda c: c.update(auth={"kind": "cookie"}), "auth.kind"),
        (lambda c: c.update(auth={"kind": "header"}), "needs auth.name"),
        (lambda c: c.update(auth={"kind": "body_field"}), "needs auth.path"),
    ],
)
def test_malformed_configs_are_refused(mutate, message):
    broken = copy.deepcopy(config())
    mutate(broken)
    with pytest.raises(AdapterConfigError, match=message):
        load_bound_adapter(broken)


@pytest.mark.parametrize(
    "retry,message",
    [
        ({"attempts": 0}, "attempts"),
        ({"concurrency": 64}, "concurrency"),
        ({"timeout_s": 0}, "timeout_s"),
        ({"on_status": ["429"]}, "on_status"),
        ({"surprise": 1}, "unsupported adapter retry keys"),
    ],
)
def test_out_of_range_retry_settings_are_refused(retry, message):
    with pytest.raises(AdapterConfigError, match=message):
        load_bound_adapter(config(retry={**seed_adapter("systemone")["retry"], **retry}))


def test_a_gradio_interaction_needs_an_api_name():
    assert load_bound_adapter(config(interaction={"pattern": "gradio-call", "api_name": "/predict"})).interaction["api_name"] == "/predict"
    with pytest.raises(AdapterConfigError, match="api_name"):
        load_bound_adapter(config(interaction={"pattern": "gradio-call"}))


def test_unknown_interaction_patterns_are_refused():
    with pytest.raises(AdapterConfigError, match="interaction"):
        load_bound_adapter(config(interaction={"pattern": "websocket"}))


@pytest.mark.parametrize(
    "auth,expected_url,expected_headers,expected_body",
    [
        ({"kind": "none"}, "https://m.example/v1?a=1", {}, {"q": 1}),
        ({"kind": "bearer"}, "https://m.example/v1?a=1", {"Authorization": "Bearer k3y"}, {"q": 1}),
        ({"kind": "bearer", "prefix": "Token "}, "https://m.example/v1?a=1", {"Authorization": "Token k3y"}, {"q": 1}),
        ({"kind": "header", "name": "x-api-key"}, "https://m.example/v1?a=1", {"x-api-key": "k3y"}, {"q": 1}),
        ({"kind": "query", "name": "key"}, "https://m.example/v1?a=1&key=k3y", {}, {"q": 1}),
        ({"kind": "body_field", "path": "auth.token"}, "https://m.example/v1?a=1", {}, {"q": 1, "auth": {"token": "k3y"}}),
    ],
)
def test_the_key_goes_where_auth_says_on_a_copy(auth, expected_url, expected_headers, expected_body):
    body = {"q": 1}
    assert place_key(auth, "k3y", "https://m.example/v1?a=1", body) == (expected_url, expected_headers, expected_body)
    assert body == {"q": 1}


def test_a_body_field_path_through_a_non_object_is_a_transform_error():
    with pytest.raises(TransformError):
        place_key({"kind": "body_field", "path": "q.token"}, "k", "https://m.example", {"q": 1})


def test_a_keyed_adapter_without_api_key_is_a_configuration_error():
    with pytest.raises(AdapterConfigError, match="API_KEY"):
        require_key({"kind": "bearer"}, {})
    assert require_key({"kind": "none"}, {}) is None
    assert require_key({"kind": "header", "name": "x"}, {"API_KEY": " k "}) == "k"


def test_a_seeded_protocol_is_native_wherever_its_key_goes():
    from decisioneval.remote.bound_adapter import mapping_origin

    assert mapping_origin(load_bound_adapter(config(auth={"kind": "none"}))) == "native"
    assert mapping_origin(load_bound_adapter(config(auth={"kind": "header", "name": "x-api-key"}))) == "native"
    assert mapping_origin(load_bound_adapter(config(response_transform='{"status": "refusal"}'))) == "onboarded"
    assert mapping_origin(load_bound_adapter(config(degraded=True))) == "degraded"


def test_a_retry_block_without_statuses_still_retries_the_usual_ones():
    bound = load_bound_adapter(config(retry={"attempts": 3}))
    assert {429, 503} <= bound.retry.on_status
