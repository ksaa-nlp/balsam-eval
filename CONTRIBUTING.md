# Guidelines for development & local testing

This is a mirco-service used by the [Arabic Benchmarks platform](https://benchmarks.ksaa.gov.sa) to run evaluations.
It's built over [LM Eval Harness](https://github.com/EleutherAI/lm-evaluation-harness), and supports a special format designed to integrate datasets.

This guide walks contributers through the steps needed to get the project running locally, 
and use the service to evaluate models.

## Get started

After cloning the repository, follow these steps to get your environment ready.

1. **Initialize a virtual environment**. In order to keep all dependncies local for the project and avoid missing your global ones, at the root of the project, run:
   ```bash
   python3.12 -m venv venv
   ```
   This will initialize a new folder named `venv` that contains a Python 3.12 binary. To tell the IDE terminal to use this version, run:
   ```bash
   source venv/bin/activate
   ```
   Feel free to use other Python environment managers like conda.
2. **Install dependencies**. At the root of this project, run:
   ```bash
   pip install .
   ```
   This will take care of installing the main dependencies. Contributors are required to install the development dependencies as well:
   ```bash
   pip install ."[dev]"
   ```
3. **Set environment variables**. The evaluation requires a number of variables to be set, those can be found in `.env.example`, you will need to copy it then **modify with you own variables** (refer to (Environment variables explained)[#environment-variables-explained] section):
   ```bash
   cp .env.example .env
   ```
4. Run the evaluation. To run the evaluation:
   ```bash
   python3 run_local.py
   ```

## Environment variables explained

- `BASE_URL`: the base URL refernces your API, without any path components. For example, if you want to evaluate ChatGPT, the base URL would be `https://api.openai.com`. Not required for `aixplain` adapter.
- `ADAPTER`: There're 4 types of adapters supported via this service (LM Harness supports more):
  - `openai-chat-completions`: any OpenAI compatible API
  - `local-chat-completions`: local inference server with OpenAI compatible API
  - `gguf`: for locally hosted models via [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) (supports `logprobs` for `multiple_choice` tasks)
  - `aixplain`: this is a new adapter added in this service, compatible with [aiXplain API](https://aixplain.com/) (supports `logprobs` for `multiple_choice` tasks)
- `MODEL`: The model name provided with `model_args` alongside the `BASE_URL` of the model, in the case of OpenAI, this may be `gpt-4o`