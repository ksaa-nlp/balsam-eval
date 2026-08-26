# Guidelines for development & local testing

This is a mirco-service used by the [Arabic Benchmarks platform](https://benchmarks.ksaa.gov.sa) to run evaluations.
It's built over [LM Eval Harness](https://github.com/EleutherAI/lm-evaluation-harness), and supports a special format designed to integrate datasets.

This guide walks contributers through the steps needed to get the project running locally, 
and use the service to evaluate models.

## Get started

After cloning the repository, follow these steps to get your environment ready.

1. **Install dependencies**. This project uses [uv](https://docs.astral.sh/uv/)
   and requires Python 3.12 or 3.13. At the project root, run:
   ```bash
   uv sync
   ```
   This creates `.venv` and installs runtime and development dependencies from
   `uv.lock`.
2. **Set environment variables**. The evaluation requires a number of variables to be set, those can be found in `.env.example`, you will need to copy it then **modify with you own variables** (refer to (Environment variables explained)[#environment-variables-explained] section):
   ```bash
   cp .env.example .env
   ```
3. **Run the evaluation**. To run the evaluation _locally_:
   ```bash
   uv run python run.py
   ```

## Tests

Run the complete test suite with:

```bash
uv run pytest
```

Tests use [Hypothesis](https://hypothesis.readthedocs.io/) for properties that
must hold across broad input ranges. Prefer property tests for pure functions
with clear invariants, such as normalization being idempotent or sanitized
output excluding forbidden characters. Keep example-based tests for exact
business cases, integrations, and regressions.

## Environment variables explained

- `BASE_URL`: the base URL refernces your API, without any path components. For example, if you want to evaluate ChatGPT, the base URL would be `https://api.openai.com`. Not required for `aixplain` adapter.
- `ADAPTER`: select an lm-eval adapter. Project adapters are listed in the
  [README](../README.md#supported-adapters); lm-eval provides additional
  built-in adapters.
- `MODEL`: The model name provided with `model_args` alongside the `BASE_URL` of the model, in the case of OpenAI, this may be `gpt-4o`
