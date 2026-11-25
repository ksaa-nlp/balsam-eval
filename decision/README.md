# decisioneval

The runner for **decision models**: non-generative models that take a multiple-choice question and return the option they picked, usually with a probability for every option. The Benchmarks Platform backend launches it as its own Cloud Build. It is separate from the balsam LLM runner in the repository root, with its own dependencies, image and build.

| Entry point | Launched for | What it does |
|---|---|---|
| `python -m decisioneval.remote` (image CMD) | an evaluation job | runs every pool question through the credential's bound adapter, scores it, writes result files to `RESULTS_PATH`, and finalizes the job |
| `python -m decisioneval.onboard` | an onboarding run | finds the adapter for an unknown API (`probe`), or writes one with an LLM (`synthesize`), then reports it to the backend |
| `python -m decisioneval` | an operator | `adapters seed-sql` (seed rows and hashes for the backend migration), `adapters request-config` (a local `REQUEST_CONFIG`), `audit <_run.json>` |

An adapter is a pair of JSONata transforms (request and response) plus where the key goes, pinned by a content hash that only Python computes. No LLM ever runs inside an evaluation.

## Environment

**Evaluation**
- `REQUEST_CONFIG_B64`: required. Base64 JSON holding `url`, `model`, `auth` and the bound `adapter` version.
- `API_KEY`: the model's credential.
- `GCLOUD_BUCKET`, `POOL_FILES` (comma-separated object paths) and `RESULTS_PATH`. Without a bucket, the runner reads `.tasks/*.json` and writes `.results/`.
- `JOB_ID`, `API_HOST` and `FINALIZE_TOKEN`: these enable `POST {API_HOST}/evaluation-jobs/{JOB_ID}/finalize`, and are required when a bucket is set.
- Optional: `MODEL`, `RUN_DEADLINE_S` and `RUNNER_IMAGE_DIGEST`.
- `DECISION_EGRESS_ALLOW_PRIVATE=1` lets a local run reach localhost. It is ignored when a bucket is set.

**Onboarding**
- `ONBOARDING_RUN_ID`, `API_HOST`, `ONBOARD_TOKEN` and `ONBOARD_STAGE` (`probe` | `synthesize`).
- `ONBOARD_REQUEST_B64`: `{url, model, auth, redacted_request?, candidates[]}`.
- `API_KEY`.

**Synthesis LLM** (the `synthesize` stage only). One generic set of variables. A local `.env` in the working directory fills in any that are unset.

| Variable | Meaning |
|---|---|
| `DECISION_SPEC_LLM_API_KEY` | the key. The backend injects it from `DECISION_MAPPING_LLM_SECRET_NAME` |
| `DECISION_SPEC_LLM_PROVIDER` | `anthropic` or `openai`. When unset, an `sk-ant-` key means Anthropic and any other key means OpenAI-compatible |
| `DECISION_SPEC_LLM_MODEL` | defaults to `claude-opus-5-5` (anthropic) or `gpt-5.1` (openai) |
| `DECISION_SPEC_LLM_BASE_URL` | optional. An OpenAI-compatible host such as `https://api.groq.com/openai/v1`, or a proxy. Anthropic appends `/v1/messages`; OpenAI appends `/chat/completions` |

The LLM only proposes transforms. Each one must pass the canary suite (at least 9 of 12 held-out questions right, in both option orders) before it is bound.

## Develop

```bash
cd decision
uv sync
uv run pytest -q
```

## Build and run locally

```bash
docker build -t decision-eval:local decision/

mkdir -p run/.tasks && cp <some MCQ pool>.json run/.tasks/
CONFIG=$(cd decision && uv run python -m decisioneval adapters request-config openai-decisions \
  --url http://host.docker.internal:8932/v1/decisions --model <model> --auth none)
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/run:/home/runner/work" \
  -e REQUEST_CONFIG="$CONFIG" -e DECISION_EGRESS_ALLOW_PRIVATE=1 decision-eval:local
cat run/.results/_decision-meta.json
(cd decision && uv run python -m decisioneval audit ../run/.results/_run.json)
```

## Cloud Build (dev)

`decision/cloudbuild-dev.yaml` builds `gcr.io/$PROJECT_ID/decision-eval-dev:latest` with `decision/` as the build context. The backend's `DECISION_EVALUATION_RUNNER_IMAGE` points at it.

Trigger settings:
- branch `^dev$`;
- config `decision/cloudbuild-dev.yaml`;
- included files `decision/**`.

The balsam image's root `.dockerignore` excludes `decision/`.
