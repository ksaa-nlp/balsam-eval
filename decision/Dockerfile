# The System One decision-model runner launched by the Benchmarks Platform backend,
# one Cloud Build per evaluation job. See README.md.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY decisioneval ./decisioneval
RUN pip install .

RUN useradd --create-home --uid 10001 runner
USER runner
WORKDIR /home/runner/work

CMD ["python", "-u", "-m", "decisioneval.remote"]
