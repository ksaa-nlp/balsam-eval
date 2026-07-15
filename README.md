# KSAA Benchmarks Evaluation Service

This service is used to evaluate models on the [Arabic Benchmarks platform](https://benchmarks.ksaa.gov.sa) by King Salman Academy for Arabic Language (KSAA).

## CLI

Install the project and run one or more pool dataset files directly:

```bash
uv run balsam-eval \
  --model gpt-4o-mini \
  --adapter openai-chat-completions \
  --api-key "$OPENAI_API_KEY" \
  ./pool-dataset.json
```

Model and judge options can also be provided through `MODEL`, `ADAPTER`,
`API_KEY`, `BASE_URL`, `JUDGE_MODEL`, `JUDGE_PROVIDER`, and `JUDGE_API_KEY`.
When no files are supplied, the runner evaluates JSON files in `.tasks/` as
before. Run `uv run balsam-eval --help` for all options.

# Contributions

We welcome contributions form the community to enhance this service. Please read the [`CONTRIBUTING`](./docs/CONTRIBUTING.md) guide to get started.
