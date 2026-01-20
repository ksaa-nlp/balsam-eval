FROM python:3.10-slim as build
ENV PYTHONUNBUFFERED=1
RUN mkdir -p /tmp && chmod 1777 /tmp && apt-get update && \
    apt-get install -y --no-install-recommends git && \
    apt-get purge -y --auto-remove && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY pyproject.toml ./pyproject.toml
RUN pip install --no-cache-dir ./

FROM python:3.10-slim
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY . .

ENV API_KEY=$API_KEY
ENV SERVER_TOKEN=$SERVER_TOKEN
ENV PATH="/opt/venv/bin:$PATH"

CMD ["python", "-u", "/app/run.py"]