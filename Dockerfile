FROM python:3.11-slim AS build

ENV PYTHONUNBUFFERED=1

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install build dependencies and git
RUN mkdir -p /tmp && chmod 1777 /tmp && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        git \
        zlib1g-dev && \
    apt-get purge -y --auto-remove && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Tell uv to use /opt/venv as the target environment
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

# Synchronize dependencies (uv will now populate /opt/venv directly)
RUN uv sync --frozen --no-editable --no-install-project

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Copy the populated virtual environment from the build stage
COPY --from=build /opt/venv /opt/venv

# Copy application code
COPY . .

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH"

CMD ["python", "-u", "/app/run.py"]