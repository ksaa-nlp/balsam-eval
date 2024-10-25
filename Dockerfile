FROM python:3.9-slim as build
ENV PYTHONUNBUFFERED 1
RUN mkdir /tmp && chmod 1777 /tmp && apt-get update && \
    apt-get install -y --no-install-recommends git && \
    apt-get purge -y --auto-remove && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY pyproject.toml ./pyproject.toml
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir ./
# TODO unfortunately, the following does not work since hard dependencies on
# torch -> nvidia in lm_eval imports
# RUN pip install --no-cache-dir ./ && \
#     pip uninstall \
#     nvidia-nvtx-cu12 nvidia-nvjitlink-cu12 nvidia-nccl-cu12 \
#     nvidia-curand-cu12 nvidia-cufft-cu12 nvidia-cuda-runtime-cu12 \
#     nvidia-cusparse-cu12 nvidia-cudnn-cu12 nvidia-cusolver-cu12 \
#     nvidia-cuda-nvrtc-cu12 nvidia-cuda-cupti-cu12 nvidia-cublas-cu12 -y

FROM python:3.9-slim
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY . .

ENV API_KEY $$API_KEY
ENV SERVER_TOKEN $$SERVER_TOKEN
ENV PATH="/opt/venv/bin:$PATH"
ENV GCLOUD_BUCKET=
ENV API_HOST=

CMD ["python", "-u", "/app/run.py"]