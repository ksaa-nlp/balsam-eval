FROM python:3.10-slim as build

ENV PYTHONUNBUFFERED=1

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install git and clean up in one layer
RUN mkdir -p /tmp && chmod 1777 /tmp && \
    apt-get update && \
    apt-get install -y --no-install-recommends git && \
    apt-get purge -y --auto-remove && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./

# Create virtual environment and install dependencies with uv
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install -e .

# Final stage
FROM python:3.10-slim

WORKDIR /app

# Copy virtual environment from build stage
COPY --from=build /opt/venv /opt/venv

# Copy application code
COPY . .

# Set environment variables
ENV API_KEY=$API_KEY \
    SERVER_TOKEN=$SERVER_TOKEN \
    PATH="/opt/venv/bin:$PATH"

CMD ["python", "-u", "/app/run.py"]