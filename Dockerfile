FROM python:3.13-slim AS build

ENV PYTHONUNBUFFERED=1

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Create virtual environment and install dependencies with uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv sync --frozen --no-install-project --no-dev --no-cache

# Final stage
FROM python:3.13-slim

WORKDIR /app

# Copy virtual environment from build stage
COPY --from=build /opt/venv /opt/venv

# Copy application code
COPY . .

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH"

CMD ["python", "-u", "/app/run.py"]