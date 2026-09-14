from unittest.mock import Mock

import pytest

from src import cli


def test_cli_sets_all_overrides_during_run_and_restores_environment(monkeypatch):
    monkeypatch.setenv("MODEL", "old-model")
    monkeypatch.delenv("POOL_FILES", raising=False)
    override_names = (
        "MODEL",
        "ADAPTER",
        "API_KEY",
        "BASE_URL",
        "JUDGE_MODEL",
        "JUDGE_PROVIDER",
        "JUDGE_API_KEY",
        "POOL_FILES",
    )
    previous = {name: cli.os.environ.get(name) for name in override_names}
    observed = {}

    def invoke():
        for name in override_names:
            observed[name] = cli.os.environ.get(name)

    monkeypatch.setattr(cli.runner, "main", invoke)

    cli.main(
        [
            "one.json",
            "two.json",
            "--model",
            "new-model",
            "--adapter",
            "adapter",
            "--api-key",
            "secret",
            "--base-url",
            "https://model.test",
            "--judge-model",
            "judge-a,judge-b",
            "--judge-provider",
            "openai,local",
            "--judge-api-key",
            "key-a,key-b",
        ]
    )

    assert observed == {
        "MODEL": "new-model",
        "ADAPTER": "adapter",
        "API_KEY": "secret",
        "BASE_URL": "https://model.test",
        "JUDGE_MODEL": "judge-a,judge-b",
        "JUDGE_PROVIDER": "openai,local",
        "JUDGE_API_KEY": "key-a,key-b",
        "POOL_FILES": "one.json,two.json",
    }
    assert {name: cli.os.environ.get(name) for name in override_names} == previous


def test_cli_without_options_does_not_override_existing_environment(monkeypatch):
    monkeypatch.setenv("POOL_FILES", "from-env.json")
    runner = Mock()
    monkeypatch.setattr(cli.runner, "main", runner)

    cli.main([])

    runner.assert_called_once_with()
    assert cli.os.environ["POOL_FILES"] == "from-env.json"


def test_cli_restores_environment_when_runner_raises(monkeypatch):
    monkeypatch.setenv("MODEL", "old")
    monkeypatch.setattr(cli.runner, "main", Mock(side_effect=RuntimeError("failed")))

    with pytest.raises(RuntimeError, match="failed"):
        cli.main(["--model", "temporary", "pool.json"])

    assert cli.os.environ["MODEL"] == "old"
    assert "POOL_FILES" not in cli.os.environ


def test_cli_rejects_unknown_arguments(monkeypatch):
    runner = Mock()
    monkeypatch.setattr(cli.runner, "main", runner)

    with pytest.raises(SystemExit) as exc:
        cli.main(["--unknown"])

    assert exc.value.code == 2
    runner.assert_not_called()
