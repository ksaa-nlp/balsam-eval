# KSAA Benchmarks Evaluation Runner

This service evaluates models on the
[KSAA Arabic Benchmarks Platform](https://benchmarks.ksaa.gov.sa). It is invoked
both in CI/Cloud Build (remote mode) by the platform backend and locally for
development.

## Pipeline overview

The runner processes one **pool dataset file** per invocation of `lm_eval`:

```text
POOL_FILES         For each file:
 ───────────► .temp/<slug>.json
              │
              ▼
              LMHDataset.export()  →  YAML task config + test/dev JSON splits
              │
              ▼
              lm_eval.simple_evaluate(tasks=[dataset.name])
              │
              ▼
              .results/<slug>.json   (per-file result, tagged with
                                       category / task / pool_file)
              │
              ▼ (remote only)
              gs://${GCLOUD_BUCKET}/${RESULTS_PATH}/<slug>.json
              │
              ▼
              POST {API_HOST}/evaluation-jobs/{JOB_ID}/finalize
                   { "outcome": "succeeded" | "failed", "error": ... }
```

`<slug>` is a hash-suffixed stem of the input path so two pool files with the
same basename in different categories never collide.

## Operating modes

The runner picks its mode from the env. There is no flag.

* **Remote** — when `API_HOST`, `FINALIZE_TOKEN` and `JOB_ID` are all set, the
  runner downloads each path in `POOL_FILES` from `gs://${GCLOUD_BUCKET}/<path>`,
  uploads result JSONs to `gs://${GCLOUD_BUCKET}/${RESULTS_PATH}/`, and posts
  the terminal outcome to the finalize endpoint with the per-job JWT.
* **Local** — pool files are discovered from `./.tasks/*.json`. Results are
  written to `./.results/` and no network calls are made.

## Environment variables

| Variable                | Mode    | Notes                                                                   |
|-------------------------|---------|-------------------------------------------------------------------------|
| `MODEL`                 | both    | Adapter-specific model identifier (e.g. `gpt-4o-mini`).                 |
| `ADAPTER`               | both    | lm_eval adapter id (`openai-chat-completions`, `gemini`, `groq`, ...).  |
| `BASE_URL`              | both?   | Optional. Overridden for Anthropic; passed through otherwise.           |
| `API_KEY`               | both?   | Model API key. Stamped into adapter-specific env var by `common.py`.    |
| `API_HOST`              | remote  | Backend base URL (no trailing slash).                                   |
| `FINALIZE_TOKEN`        | remote  | Per-job JWT (`scope=finalize`, bound to `jobId`, 24h TTL).              |
| `JOB_ID`                | remote  | Numeric evaluation-job id.                                              |
| `EVALUATION_ID`         | remote  | Numeric evaluation id (informational).                                  |
| `CATEGORY`              | remote  | Numeric category id (fallback when the pool file lacks `category`).     |
| `BENCHMARK_ID`          | remote  | Numeric benchmark id (informational).                                   |
| `BENCHMARK_VERSION_ID`  | remote  | Numeric benchmark-version id (informational).                           |
| `GCLOUD_BUCKET`         | remote  | GCS bucket for pool files + result uploads.                             |
| `RESULTS_PATH`          | remote  | Object-path prefix for result uploads (env-prefix included by backend). |
| `POOL_FILES`            | remote  | Comma-separated GCS object paths to evaluate (one lm_eval call each).   |
| `EVALUATION_TYPES`      | remote? | Optional filter; comma-separated.                                       |
| `MODALITIES`            | remote? | Optional filter; comma-separated.                                       |
| `JUDGE_MODEL`           | both?   | Comma-separated LLM-judge model ids.                                    |
| `JUDGE_PROVIDER`        | both?   | Comma-separated providers, paired with `JUDGE_MODEL`.                   |
| `JUDGE_API_KEY`         | both?   | Comma-separated keys, paired with `JUDGE_MODEL` / `JUDGE_PROVIDER`.     |
| `IS_REASONING`          | both?   | `1` to use reasoning-model token budgets.                               |
| `MAX_TOKENS`            | both?   | Token budget when `IS_REASONING=1`.                                     |

`?` = optional. See [`src/core/config.py`](src/core/config.py) for the full
parsing rules.

## Output format

Each result JSON has the keys `lm_eval` produces plus the runner's stamped
metadata at the top level:

```jsonc
{
  "results": { "<task-config-name>": { "rouge,none": { "rougeLsum": 0.49 }, "task": "...", "category": "..." } },
  "samples": { "<task-config-name>": [ ... ] },
  "configs": { "<task-config-name>": { ... } },
  "average_scores": { "rouge": 0.4965, "rouge1": 0.49, "rouge2": 0.16, "rougeL": 0.49 },
  "category": "Question Answering",
  "task": "Answering Given Question",
  "pool_file": "evaluations/42/pool-files/dataset-7.json"
}
```

`pool_file` is the exact string the backend put in `POOL_FILES`; the backend
uses it to map results back to `pool_dataset_files.file_path`.

## Local development

1. Drop pool files under `./.tasks/` — either the legacy
   `{name, task, category, json: {...}}` envelope or the flat format the
   backend generates.
2. Export `MODEL` and `ADAPTER` (and `API_KEY` / `BASE_URL` as needed).
3. `uv run python run.py` (or `python -u run.py` inside an activated venv).
4. Result JSONs land in `./.results/`.

For GCS-backed local runs (rare), point `GOOGLE_APPLICATION_CREDENTIALS` at a
service-account key or run `gcloud auth application-default login` first.

## Cloud Build / Docker

`Dockerfile` builds the image; `cloudbuild.yaml` ships it to the project's
container registry. The backend's `EvaluationJobLauncherService` creates each
evaluation build with the runner image and the env vars listed above.

## Contributions

See [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md).
