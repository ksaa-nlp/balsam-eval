"""Configuration for the evaluation runner.

The runner is launched in two ways:

* Remote (Cloud Build): the backend sets ``API_HOST``, ``FINALIZE_TOKEN``,
  ``JOB_ID``, ``GCLOUD_BUCKET``, ``RESULTS_PATH`` and ``POOL_FILES`` (a
  comma-separated list of GCS object paths to evaluate). ``FINALIZE_TOKEN`` is
  a per-job JWT (scope=``finalize``, ~1-week TTL) the runner sends back as a
  Bearer token when calling ``POST /evaluation-jobs/:id/finalize``.
* Local: only ``MODEL`` / ``ADAPTER`` (and optionally ``BASE_URL`` / ``API_KEY``)
  are needed. Input files are discovered from the ``.tasks/`` directory and
  result JSONs are written to ``.results/`` — no network calls are made.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class EvalConfig:
    """Resolved environment for a single runner invocation."""

    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model_name: str = ""
    adapter: str = ""

    # Backend coordinates (remote mode only)
    api_host: Optional[str] = None
    finalize_token: Optional[str] = None
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

    # Performance controls
    batch_size: int = 8
    concurrency: int = 8
    bootstrap_iters: int = 100000

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
            finalize_token=os.getenv("FINALIZE_TOKEN"),
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
            batch_size=cls._parse_positive_int_env("EVAL_BATCH_SIZE", 8),
            concurrency=cls._parse_positive_int_env("EVAL_CONCURRENCY", 8),
            bootstrap_iters=cls._parse_non_negative_int_env(
                "EVAL_BOOTSTRAP_ITERS", 100000
            ),
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

    @staticmethod
    def _parse_positive_int_env(name: str, default: int) -> int:
        value = EvalConfig._parse_non_negative_int_env(name, default)
        if value < 1:
            raise ValueError(f"{name} must be greater than zero")
        return value

    @staticmethod
    def _parse_non_negative_int_env(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return default
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if value < 0:
            raise ValueError(f"{name} must not be negative")
        return value

    def is_remote_job(self) -> bool:
        """True when any remote-job configuration has been supplied."""
        return any(
            (
                self.api_host,
                self.finalize_token,
                self.job_id,
                self.bucket,
                self.results_path,
                self.pool_files,
            )
        )

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
            "finalize_token",
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
                f"{', '.join(missing)}. Required: API_HOST, FINALIZE_TOKEN, JOB_ID, "
                "ADAPTER, MODEL, GCLOUD_BUCKET, RESULTS_PATH"
            )
        if not self.pool_files:
            raise ValueError("POOL_FILES is required for remote run")

    def get_model_args(self, base_url: Optional[str] = None) -> dict[str, Any]:
        """Return the kwargs to pass to the adapter constructor."""
        args: dict[str, Any] = {"model": self.model_name}
        resolved_base_url = base_url or self.base_url
        if resolved_base_url:
            args["base_url"] = resolved_base_url
        if self.api_key:
            args["api_key"] = self.api_key
        if self.concurrency > 1:
            args["num_concurrent"] = self.concurrency
        return args

    def get_evaluation_types_list(self) -> list[str]:
        """Split the comma-separated EVALUATION_TYPES env value into a list."""
        if not self.evaluation_types:
            return []
        return [t.strip() for t in self.evaluation_types.split(",") if t.strip()]
