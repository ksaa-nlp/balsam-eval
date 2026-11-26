"""The key is removed by value, wherever and however it was written."""

from decisioneval.store import KEY_PLACEHOLDER, scrub_key

KEY = "sk-live+a/b=c"


def test_every_encoding_of_the_key_is_replaced_in_nested_documents():
    document = {
        "error": f"refused a 302 redirect from https://m.example/v1?key=sk-live%2Ba%2Fb%3Dc and Bearer {KEY}",
        "reply": {"you_sent": KEY, "plus": "key=sk-live%2Ba%2Fb%3Dc", "form": "key=sk-live%2Ba%2Fb%3Dc", KEY: 1},
        "list": [f"x{KEY}y", 3, None],
    }
    cleaned = scrub_key(document, KEY)
    assert "sk-live" not in repr(cleaned)
    assert cleaned["reply"]["you_sent"] == KEY_PLACEHOLDER and KEY_PLACEHOLDER in cleaned["reply"]


def test_the_query_plus_encoding_is_replaced():
    assert scrub_key("key=sk+live+now", "sk live now") == f"key={KEY_PLACEHOLDER}"


def test_the_form_a_client_sends_and_lower_case_escapes_are_replaced():
    for echoed in ("key=sk-live%2Ba/b%3Dc", "key=sk-live%2ba%2fb%3dc"):
        assert "sk-live" not in scrub_key(echoed, KEY)


def test_a_short_placeholder_key_never_rewrites_data():
    assert scrub_key({"A": 0.5, "B": 0.5, "state": "EMPTY"}, "EMPTY") == {"A": 0.5, "B": 0.5, "state": "EMPTY"}


def test_no_key_changes_nothing():
    assert scrub_key({"a": "b"}, None) == {"a": "b"}


def test_clip_scrubs_before_it_cuts():
    from decisioneval.store import clip

    text = "x" * 10 + KEY
    assert "sk-live" not in clip(text, 14, KEY) and clip(text, 14, KEY).startswith("x" * 10)


def test_log_lines_and_tracebacks_lose_the_key():
    import logging

    from decisioneval.store import KeyScrubFilter

    record = logging.LogRecord("t", logging.ERROR, __file__, 1, "crashed:\n%s", (f"Traceback ... url?key={KEY}",), None)
    KeyScrubFilter(KEY).filter(record)
    assert "sk-live" not in record.getMessage()


def test_an_exception_logged_with_exc_info_loses_the_key():
    import io
    import logging

    from decisioneval.store import KeyScrubFilter

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(KeyScrubFilter(KEY))
    logger = logging.getLogger("test.scrub.exc_info")
    logger.addHandler(handler)
    try:
        raise RuntimeError(f"GET https://m.example/v1?key={KEY} failed")
    except RuntimeError:
        logger.exception("call failed")
    finally:
        logger.removeHandler(handler)
    assert "call failed" in stream.getvalue() and "RuntimeError" in stream.getvalue()
    assert "sk-live" not in stream.getvalue()
