"""Configuration management for evaluation jobs."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class EvalConfig:
    """Configuration for evaluation job."""

    base_url: Optional[str]
    api_key: Optional[str]
    model_name: str
    adapter: str
    server_token: Optional[str]
    api_host: Optional[str]
    user_id: Optional[str]
    benchmark_id: Optional[str]
    evaluation_types: Optional[str]
    llm_judge: Optional[str]
    llm_judge_provider: Optional[str]
    llm_judge_api_key: Optional[str]
    category_id: Optional[str]
    job_id: Optional[str]
    temperature: Optional[str]
    parallel_categories: bool = False

    @classmethod
    def from_env(cls) -> 'EvalConfig':
        """Create config from environment variables."""
        return cls(
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("API_KEY"),
            model_name=os.getenv("MODEL", ""),
            adapter=os.getenv("ADAPTER", ""),
            server_token=os.getenv("SERVER_TOKEN"),
            api_host=os.getenv("API_HOST"),
            user_id=os.getenv("USER_ID"),
            benchmark_id=os.getenv("BENCHMARK_ID"),
            evaluation_types=os.getenv("EVALUATION_TYPES"),
            llm_judge=os.getenv("JUDGE_MODEL"),
            llm_judge_provider=os.getenv("JUDGE_PROVIDER"),
            llm_judge_api_key=os.getenv("JUDGE_API_KEY"),
            category_id=os.getenv("CATEGORY"),
            job_id=os.getenv("JOB_ID"),
            temperature=os.getenv("TEMPERATURE"),
            parallel_categories=os.getenv("PARALLEL_CATEGORIES", "false").lower() in ("true", "1", "yes", "on"),
        )

    def validate_local(self) -> None:
        """Validate configuration for local execution."""
        if not self.model_name:
            raise ValueError("MODEL name is required")
        if not self.adapter:
            raise ValueError("Adapter is required")

    def validate_remote(self) -> None:
        """Validate configuration for remote execution."""
        required = ["api_host", "server_token", "category_id", "adapter", "benchmark_id"]
        missing = [attr for attr in required if not getattr(self, attr)]

        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                f"Required: API_HOST, SERVER_TOKEN, CATEGORY, ADAPTER, BENCHMARK_ID"
            )

        if not self.model_name:
            raise ValueError("MODEL name is required")

    def is_remote_job(self) -> bool:
        """Check if this is a remote evaluation job."""
        return bool(self.api_host and self.server_token and self.category_id)

    def get_model_args(self, base_url: Optional[str] = None) -> dict[str, str]:
        """Get model arguments for evaluation.

        Args:
            base_url: Optional base URL to override config

        Returns:
            Dictionary of model arguments
        """
        model_args = {"model": self.model_name}
        if base_url:
            model_args["base_url"] = base_url
        if self.api_key:
            model_args["api_key"] = self.api_key
        return model_args

    def get_evaluation_types_list(self) -> list[str]:
        """Get evaluation types as a list.

        Returns:
            List of evaluation type names
        """
        if not self.evaluation_types:
            return []
        return [t.strip() for t in self.evaluation_types.split(",")]
