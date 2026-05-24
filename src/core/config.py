"""Configuration for the evaluation runner.

The runner is launched in two ways:

* Remote (Cloud Build): the backend sets ``API_HOST``, ``SERVER_TOKEN``,
  ``JOB_ID``, ``GCLOUD_BUCKET``, ``RESULTS_PATH`` and ``POOL_FILES`` (a
  comma-separated list of GCS object paths to evaluate).
* Local: only ``MODEL`` / ``ADAPTER`` (and optionally ``BASE_URL`` / ``API_KEY``)
  are needed. Input files are discovered from the ``.tasks/`` directory and
  result JSONs are written to ``.results/`` — no network calls are made.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalConfig:
    """Resolved environment for a single runner invocation."""

    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model_name: str = ""
    adapter: str = ""

    # Backend coordinates (remote mode only)
    api_host: Optional[str] = None
    server_token: Optional[str] = None
    job_id: Optional[str] = None
    evaluation_id: Optional[str] = None
    benchmark_id: Optional[str] = None
    benchmark_version_id: Optional[str] = None
    category_id: Optional[str] = None

    # GCS coordinates (remote mode only)
    bucket: Optional[str] = None
    results_path: Optional[str] = None
    pool_files: list[str] = field(default_factory=list)

    # Selection filters / metadata
    evaluation_types: Optional[str] = None
    modalities: Optional[str] = None
    temperature: Optional[str] = None

    # LLM-as-judge configuration
    llm_judge: list[str] = field(default_factory=list)
    llm_judge_provider: list[str] = field(default_factory=list)
    llm_judge_api_key: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "EvalConfig":
        """Build a config from process environment variables."""
        return cls(
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("API_KEY"),
            model_name=os.getenv("MODEL", ""),
            adapter=os.getenv("ADAPTER", ""),
            api_host=os.getenv("API_HOST"),
            server_token=os.getenv("SERVER_TOKEN"),
            job_id=os.getenv("JOB_ID"),
            evaluation_id=os.getenv("EVALUATION_ID"),
            benchmark_id=os.getenv("BENCHMARK_ID"),
            benchmark_version_id=os.getenv("BENCHMARK_VERSION_ID"),
            category_id=os.getenv("CATEGORY"),
            bucket=os.getenv("GCLOUD_BUCKET"),
            results_path=os.getenv("RESULTS_PATH"),
            pool_files=cls._parse_csv_env("POOL_FILES"),
            evaluation_types=os.getenv("EVALUATION_TYPES"),
            modalities=os.getenv("MODALITIES"),
            temperature=os.getenv("TEMPERATURE"),
            llm_judge=cls._parse_csv_env("JUDGE_MODEL"),
            llm_judge_provider=cls._parse_csv_env("JUDGE_PROVIDER"),
            llm_judge_api_key=cls._parse_csv_env("JUDGE_API_KEY"),
        )

    @staticmethod
    def _parse_csv_env(name: str) -> list[str]:
        raw = os.getenv(name, "")
        if not raw:
            return []
        return [v.strip() for v in raw.split(",") if v.strip()]

    def is_remote_job(self) -> bool:
        """True when the runner should report status / upload to GCS."""
        return bool(self.api_host and self.server_token and self.job_id)

    def validate_local(self) -> None:
        """Ensure the minimum env vars for a local run are set."""
        if not self.model_name:
            raise ValueError("MODEL is required")
        if not self.adapter:
            raise ValueError("ADAPTER is required")

    def validate_remote(self) -> None:
        """Ensure all backend / GCS coordinates required for a remote run are set."""
        required = [
            "api_host",
            "server_token",
            "job_id",
            "adapter",
            "model_name",
            "bucket",
            "results_path",
        ]
        missing = [attr for attr in required if not getattr(self, attr)]
        if missing:
            raise ValueError(
                "Missing required environment variables for remote run: "
                f"{', '.join(missing)}. Required: API_HOST, SERVER_TOKEN, JOB_ID, "
                "ADAPTER, MODEL, GCLOUD_BUCKET, RESULTS_PATH"
            )
        if not self.pool_files:
            raise ValueError("POOL_FILES is required for remote run")

    def get_model_args(self, base_url: Optional[str] = None) -> dict[str, str]:
        """Return the kwargs to pass to the adapter constructor."""
        args: dict[str, str] = {"model": self.model_name}
        if base_url:
            args["base_url"] = base_url
        if self.api_key:
            args["api_key"] = self.api_key
        return args

    def get_evaluation_types_list(self) -> list[str]:
        """Split the comma-separated EVALUATION_TYPES env value into a list."""
        if not self.evaluation_types:
            return []
        return [t.strip() for t in self.evaluation_types.split(",") if t.strip()]
