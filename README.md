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
| `EVAL_BATCH_SIZE`       | both?   | Requests grouped per model call; defaults to `8`.                       |
| `EVAL_CONCURRENCY`      | both?   | Concurrent requests for compatible API adapters; defaults to `8`.      |
| `EVAL_BOOTSTRAP_ITERS`  | both?   | Statistical bootstrap iterations; defaults to `100000`; use `0` for speed. |

`?` = optional. See [`src/core/config.py`](src/core/config.py) for the full
parsing rules.

## Supported adapters

Chat adapters:

| Adapter | Provider/runtime | Additional configuration |
|---|---|---|
| `openai` | OpenAI chat with model-dependent image/audio input | `OPENAI_API_KEY` |
| `local-adapter` | OpenAI-compatible text/image/audio endpoint | `BASE_URL`; media support depends on server/model |
| `anthropic` | Anthropic text/image Messages | `ANTHROPIC_API_KEY`; audio is unsupported |
| `cohere` | Cohere Chat with image input | `CO_API_KEY` |
| `gemini` | Gemini/Vertex text, image, and audio | `GOOGLE_API_KEY` or application credentials |
| `groq` | Groq text/image chat | `GROQ_API_KEY`; audio uses `openai-asr` |
| `azure-openai` | Azure OpenAI text/image/audio chat | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`; media support depends on deployment |
| `aixplain` | aiXplain text/image models | `AIXPLAIN_API_KEY`; audio has no model-independent schema |
| `huggingface-chat` | Hugging Face text/image inference | `HF_TOKEN`; audio unsupported |
| `aws-bedrock` (`bedrock`) | Amazon Bedrock text/image Converse | AWS credentials and region; audio unsupported |
| `sagemaker-chat` (`sagemaker`) | Amazon SageMaker text/image/audio endpoint | AWS credentials, region, and `SAGEMAKER_ENDPOINT_NAME`; media support depends on endpoint |

ASR adapters:

| Adapter | Provider/runtime | Additional configuration |
|---|---|---|
| `openai-asr` | OpenAI-compatible transcription API | `OPENAI_API_KEY` |
| `google-stt` | Google Cloud Speech-to-Text | Application credentials |
| `azure-stt` | Azure AI Speech | `AZURE_SPEECH_KEY` |
| `hf-asr` | Hugging Face inference | `HF_TOKEN` |
| `nemo-asr` | NVIDIA NIM | `NVIDIA_API_KEY` |
| `ibm-stt` | IBM Watson Speech to Text | `IBM_API_KEY`, `IBM_STT_URL` |
| `qwen-asr` | DashScope or self-hosted Qwen ASR | `BASE_URL` |
| `cohere-asr` | Cohere Transcribe | `COHERE_API_KEY` |
| `deepgram-stt` | Deepgram | `DEEPGRAM_API_KEY` |
| `speechmatics-stt` | Speechmatics | `SPEECHMATICS_API_KEY` |
| `assemblyai-stt` | AssemblyAI | `ASSEMBLYAI_API_KEY` |
| `elevenlabs-stt` | ElevenLabs Scribe | `ELEVENLABS_API_KEY` |
| `gladia-stt` | Gladia | `GLADIA_API_KEY` |
| `revai-stt` | Rev AI | `REVAI_API_KEY` |
| `aws-transcribe` | Amazon Transcribe | AWS credentials, region, and `AWS_TRANSCRIBE_S3_BUCKET` |

`API_KEY` is accepted as the generic credential input and copied to the
provider-specific variable where applicable. Provider-specific variables are
useful when invoking adapters directly. Most ASR adapters accept
`ASR_LANGUAGE`; provider URL overrides are documented in their modules.

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

## Usage

### Run with Python

The standard runner keeps the original repository workflow:

1. Put pool dataset JSON files in `./.tasks/`.
2. Configure `MODEL` and `ADAPTER` in `./.env`.
3. Add `API_KEY` and `BASE_URL` when required by the adapter.
4. Run `python3 run.py` or `uv run python run.py`.

Example `.env`:

```dotenv
MODEL=gpt-4o-mini
ADAPTER=openai-chat-completions
API_KEY=your-model-api-key
```

The runner loads `.env` from the current working directory, discovers
`./.tasks/*.json`, and writes result files to `./.results/`.

### Run with the CLI

The CLI accepts dataset paths directly:

```bash
balsam-eval \
  --model gpt-4o-mini \
  --adapter openai-chat-completions \
  --api-key "$OPENAI_API_KEY" \
  ./pool-dataset.json
```

Pass multiple files to evaluate them in one invocation:

```bash
balsam-eval --model gpt-4o-mini --adapter openai-chat-completions \
  ./dataset-1.json ./dataset-2.json
```

`MODEL` and `ADAPTER` are required. `API_KEY` and `BASE_URL` are optional at
the runner level but may be required by the selected adapter. A dataset must
be passed to the CLI or available under `./.tasks/`.

CLI options override existing process environment variables, which override
values from `./.env`.

```text
usage: balsam-eval [-h] [--model MODEL] [--adapter ADAPTER]
                   [--api-key API_KEY] [--base-url BASE_URL]
                   [--judge-model JUDGE_MODEL]
                   [--judge-provider JUDGE_PROVIDER]
                   [--judge-api-key JUDGE_API_KEY]
                   [FILE ...]
```

### Run without cloning

Use `uvx` to install the package into a temporary isolated environment and run
it directly from GitHub:

```bash
uvx --from git+https://github.com/ksaa-nlp/balsam-eval.git balsam-eval \
  --model gpt-4o-mini \
  --adapter openai-chat-completions \
  --api-key "$OPENAI_API_KEY" \
  ./pool-dataset.json
```

After publishing the package to PyPI, use:

```bash
uvx --from balsam-lm-evaluation balsam-eval ...
```

### Optional LLM judge

Judge configuration is optional. It is needed only when a dataset uses an
LLM-as-judge metric:

```bash
balsam-eval --model gpt-4o-mini --adapter openai-chat-completions \
  --judge-model gemini-2.5-flash \
  --judge-provider gemini \
  --judge-api-key "$GEMINI_API_KEY" \
  ./pool-dataset.json
```

For multiple judges, provide comma-separated model, provider, and key values
in matching order. A single provider or API key may be shared when supported
by the judge configuration.

API keys can appear in shell history when passed as arguments. Prefer `.env`
or environment variables on shared systems.

For GCS-backed local runs, point `GOOGLE_APPLICATION_CREDENTIALS` at a
service-account key or run `gcloud auth application-default login` first.

## Cloud Build / Docker

`Dockerfile` builds the image; `cloudbuild.yaml` ships it to the project's
container registry. The backend's `EvaluationJobLauncherService` creates each
evaluation build with the runner image and the env vars listed above.

## Contributions

See [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md).
