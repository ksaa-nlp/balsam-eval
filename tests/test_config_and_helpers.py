import pytest

from src.core.config import EvalConfig
from src.core.helpers import normalize_string, sanitize_config_name


REMOTE_FIELDS = {
    "api_host": "https://backend.example",
    "finalize_token": "token",
    "job_id": "12",
    "adapter": "openai",
    "model_name": "model",
    "bucket": "bucket",
    "results_path": "results",
    "pool_files": ["pool.json"],
}


def test_from_env_parses_scalar_and_csv_values(monkeypatch):
    monkeypatch.setenv("MODEL", "gpt-test")
    monkeypatch.setenv("ADAPTER", "openai")
    monkeypatch.setenv("POOL_FILES", " one.json, ,two.json ")
    monkeypatch.setenv("JUDGE_MODEL", "judge-a,judge-b")

    config = EvalConfig.from_env()

    assert config.model_name == "gpt-test"
    assert config.adapter == "openai"
    assert config.pool_files == ["one.json", "two.json"]
    assert config.llm_judge == ["judge-a", "judge-b"]


@pytest.mark.parametrize("field", REMOTE_FIELDS)
def test_validate_remote_reports_each_missing_field(field):
    values = REMOTE_FIELDS.copy()
    values[field] = [] if field == "pool_files" else ""
    config = EvalConfig(**values)

    with pytest.raises(ValueError):
        config.validate_remote()


def test_validate_remote_accepts_complete_configuration():
    EvalConfig(**REMOTE_FIELDS).validate_remote()


@pytest.mark.parametrize(
    ("model", "adapter", "message"),
    [("", "openai", "MODEL is required"), ("model", "", "ADAPTER is required")],
)
def test_validate_local_requires_model_and_adapter(model, adapter, message):
    with pytest.raises(ValueError, match=message):
        EvalConfig(model_name=model, adapter=adapter).validate_local()


def test_model_args_include_only_configured_values():
    config = EvalConfig(
        model_name="model", base_url="https://default", api_key="secret"
    )

    assert config.get_model_args() == {
        "model": "model",
        "base_url": "https://default",
        "api_key": "secret",
    }
    assert config.get_model_args("https://override") == {
        "model": "model",
        "base_url": "https://override",
        "api_key": "secret",
    }


def test_evaluation_types_are_trimmed_and_empty_values_removed():
    config = EvalConfig(evaluation_types=" mcq, , generative ")

    assert config.get_evaluation_types_list() == ["mcq", "generative"]


def test_remote_detection_uses_any_remote_coordinate():
    assert not EvalConfig(model_name="m", adapter="a").is_remote_job()
    assert EvalConfig(job_id="1").is_remote_job()
    assert EvalConfig(pool_files=["pool.json"]).is_remote_job()


def test_normalize_string_normalizes_and_makes_path_safe():
    assert normalize_string("  Ａ File.Name/Part\x00 ") == "a_file_name_part"
    assert len(normalize_string("x" * 300)) == 255


def test_sanitize_config_name_replaces_forbidden_characters():
    assert sanitize_config_name(r"a<b>c:d/e\f|g?h*i") == "a_b_c_d_e_f_g_h_i"
