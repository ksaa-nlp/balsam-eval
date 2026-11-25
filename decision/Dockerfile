# The System One decision-model runner launched by the Benchmarks Platform backend,
# one Cloud Build per evaluation job. See README.md.
FROM python:3.13-slim AS build

COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /bin/

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY decisioneval ./decisioneval
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

COPY --from=build /opt/venv /opt/venv

RUN useradd --create-home --uid 10001 runner
USER runner
WORKDIR /home/runner/work

CMD ["python", "-u", "-m", "decisioneval.remote"]
